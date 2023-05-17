import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
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
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'NDCG'], 'test': ['Loss', 'NDCG']})
    else:
        raise ValueError('Not valid target mode')
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    data_split = result['data_split']
    dataset = make_split_dataset(data_split)
    data_loader = {'train': [], 'test': []}
    model = []
    for i in range(len(dataset)):
        data_loader_i = make_data_loader(dataset[i], cfg['model_name'])
        num_users = dataset[i]["train"].num_users['data']
        num_items = dataset[i]["train"].num_items['data']
        model_i = eval('models.{}(num_users, num_items).to(cfg["device"])'.format(cfg['model_name']))
        data_loader['train'].append(data_loader_i['train'])
        data_loader['test'].append(data_loader_i['test'])
        model.append(model_i)
    model = models.mdr(model)
    model.load_state_dict(result['model_state_dict'])
    test_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test_each_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test_each(data_loader['test'], model, metric, test_each_logger, last_epoch)
    test(data_loader['test'], model, metric, test_logger, last_epoch)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch,
              'logger': {'train': train_logger, 'test': test_logger, 'test_each': test_each_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test_each(data_loader, model, metric, each_logger, epoch):
    with torch.no_grad():
        model.train(False)
        for m in range(len(data_loader)):
            each_logger.safe(True)
            for i, input in enumerate(data_loader[m]):
                input = collate(input)
                input_size = len(input['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model(input, m)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                each_logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                             'ID: {}/{}'.format(m + 1, len(data_loader))]}
            each_logger.append(info, 'test', mean=False)
            print(each_logger.write('test', metric.metric_name['test']))
            each_logger.safe(False)
            each_logger.reset()
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
