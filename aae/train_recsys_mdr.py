import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_split = split_dataset(dataset)
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        if 'data_split' in result:
            data_split = result['data_split']
    dataset = make_split_dataset(data_split)
    data_loader = {'train': [], 'test': []}
    model = []
    for i in range(len(dataset)):
        data_loader_i = make_data_loader(dataset[i], cfg['model_name'])
        num_users = dataset[i]['train'].num_users['data']
        num_items = dataset[i]['train'].num_items['data']
        model_i = eval('models.{}(num_users, num_items).to(cfg["device"])'.format(cfg['model_name']))
        data_loader['train'].append(data_loader_i['train'])
        data_loader['test'].append(data_loader_i['test'])
        model.append(model_i)
    model = models.mdr(model)
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'NDCG'], 'test': ['Loss', 'NDCG']})
    else:
        raise ValueError('Not valid target mode')
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            for i in range(len(dataset)):
                model[i].load_state_dict(result['model_state_dict'][i])
            logger = result['logger']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    if cfg['world_size'] > 1:
        for i in range(len(dataset)):
            model[i] = torch.nn.DataParallel(model[i], device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        test(data_loader['test'], model, metric, logger, epoch)
        scheduler.step()
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        scheduler_state_dict = scheduler.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict,
                  'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train(data_loader, model, optimizer, metric, logger, epoch):
    logger.safe(True)
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(zip(*data_loader)):
        loss = 0
        for m in range(len(input)):
            input_m = collate(input[m])
            input_size = len(input_m[cfg['data_mode']])
            if input_size == 0:
                continue
            input_m = to_device(input_m, cfg['device'])
            output_m = model(input_m, m)
            output_m['loss'] = output_m['loss'].mean() if cfg['world_size'] > 1 else output_m['loss']
            loss += output_m['loss']
            evaluation = metric.evaluate(metric.metric_name['train'], input_m, output_m)
            logger.append(evaluation, 'train', n=input_size)
        loss = loss / len(input)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if i % int((len(data_loader[0]) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader[0]) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader[0])))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader[0])),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    model.train(False)
    with torch.no_grad():
        for i, input in enumerate(zip(*data_loader)):
            input_target_user = []
            input_target_item = []
            input_target_rating = []
            output_target_rating = []
            for m in range(len(input)):
                input_m = collate(input[m])
                input_size = len(input_m['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input_m = to_device(input_m, cfg['device'])
                output_m = model(input_m, m)
                input_target_user.append(input_m['target_user'])
                input_target_item.append(input_m['target_item'])
                input_target_rating.append(input_m['target_rating'])
                output_target_rating.append(output_m['target_rating'])
                output_m['loss'] = output_m['loss'].mean() if cfg['world_size'] > 1 else output_m['loss']
                evaluation = metric.evaluate([metric.metric_name['test'][0]], input_m, output_m)
                logger.append(evaluation, 'test', input_size)
            output = {'target_rating': torch.cat(output_target_rating)}
            input = {'target_user': torch.cat(input_target_user), 'target_item': torch.cat(input_target_item),
                     'target_rating': torch.cat(input_target_rating)}
            input_size = len(input['target_{}'.format(cfg['data_mode'])])
            if input_size == 0:
                continue
            evaluation = metric.evaluate(metric.metric_name['test'][1:], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
