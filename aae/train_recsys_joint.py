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
    data_loader = make_data_loader(dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    if cfg['model_name'] != 'base':
        optimizer = make_optimizer(model, cfg['model_name'])
        scheduler = make_scheduler(optimizer, cfg['model_name'])
    else:
        optimizer = None
        scheduler = None
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
            model.load_state_dict(result['model_state_dict'])
            if cfg['model_name'] != 'base':
                optimizer.load_state_dict(result['optimizer_state_dict'])
                scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
            data_split = result['data_split']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    if cfg['model_name'] not in ['ae']:
        local_dataset = make_split_dataset(data_split)
    else:
        local_dataset = [copy.deepcopy(dataset) for _ in range(len(data_split))]
    local_data_loader = {'test': []}
    local_model = []
    for i in range(len(local_dataset)):
        local_data_loader_i = make_data_loader(local_dataset[i], cfg['model_name'])
        local_data_loader['test'].append(local_data_loader_i['test'])
        num_users = local_dataset[i]['train'].num_users['data']
        num_items = local_dataset[i]['train'].num_items['data']
        if cfg['model_name'] == 'ae':
            local_model_i = eval(
                'models.{}(num_users, num_items, num_users, num_items).to(cfg["device"])'.format(cfg['model_name']))
        else:
            local_model_i = eval(
                'models.{}(num_users, num_items).to(cfg["device"])'.format(cfg['model_name']))
        local_model.append(local_model_i)
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        models.distribute(model, local_model, data_split)
        test(local_data_loader['test'], data_split, local_model, metric, logger, epoch)
        if scheduler is not None:
            scheduler.step()
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        if cfg['model_name'] != 'base':
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
            result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'model_state_dict': model_state_dict,
                      'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict,
                      'logger': logger}
        else:
            result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'model_state_dict': model_state_dict,
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
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = len(input[cfg['data_mode']])
        if input_size == 0:
            continue
        input = to_device(input, cfg['device'])
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        if optimizer is not None:
            optimizer.zero_grad()
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, data_split, model, metric, logger, epoch):
    logger.safe(True)
    for m in range(len(data_loader)):
        model[m].train(False)
    with torch.no_grad():
        for i, input in enumerate(zip(*data_loader)):
            input_target_user = []
            input_target_item = []
            input_target_rating = []
            output_target_rating = []
            for m in range(len(input)):
                input_m = collate(input[m])
                if cfg['model_name'] == 'ae':
                    if cfg['data_mode'] == 'user':
                        mask = torch.isin(input_m['target_item'], data_split[m])
                    else:
                        mask = torch.isin(input_m['target_user'], data_split[m])
                    if ~torch.any(mask):
                        continue
                    input_m['target_user'] = input_m['target_user'][mask]
                    input_m['target_item'] = input_m['target_item'][mask]
                    input_m['target_rating'] = input_m['target_rating'][mask]
                input_size = len(input_m['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input_m = to_device(input_m, cfg['device'])
                output_m = model[m](input_m)
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
