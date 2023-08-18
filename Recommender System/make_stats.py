import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import datasets
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from utils import save, process_control, process_dataset, collate, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

if __name__ == "__main__":
    stats_path = os.path.join('res', 'stats')
    cfg['seed'] = 0
    cfg['control']['target_mode'] = 'explicit'
    data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']
    with torch.no_grad():
        for data_name in data_names:
            cfg['control']['data_name'] = data_name
            process_control()
            dataset = fetch_dataset(cfg['data_name'], verbose=False)
            stats = {'m': dataset['train'].num_users, 'n': dataset['train'].num_items}
            stats['min'] = dataset['train'].data.data.min()
            stats['max'] = dataset['train'].data.data.max()
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}.pt'.format(data_name)))
