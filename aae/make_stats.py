import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import datasets
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset
from utils import save, process_control, process_dataset, collate, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

if __name__ == "__main__":
    cfg['seed'] = 0
    data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']
    with torch.no_grad():
        for data_name in data_names:
            cfg['control']['data_name'] = data_name
            process_control()
            dataset = fetch_dataset(cfg['data_name'], verbose=False)
            stats = {'m': dataset['train'].num_users, 'n': dataset['train'].num_items}
            if hasattr(dataset['train'], 'user_profile'):
                stats['user_profile'] = {k: dataset['train'].user_profile[k].shape[-1] for k in
                                         dataset['train'].user_profile}
            if hasattr(dataset['train'], 'item_attr'):
                stats['item_attr'] = {k: dataset['train'].item_attr[k].shape[-1] for k in dataset['train'].item_attr}
            stats['sparsity'] = len(dataset['train'].data.data) / (stats['m']['data'] * stats['n']['data'])
            cfg['control']['data_mode'] = 'user'
            cfg['control']['data_split_mode'] = 'genre'
            process_control()
            dataset = fetch_dataset(cfg['data_name'], verbose=False)
            data_split = split_dataset(dataset)
            stats['n_k'] = [len(data_split[i]) for i in range(len(data_split))]
            if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                cfg['control']['data_mode'] = 'item'
                cfg['control']['data_split_mode'] = 'random-8'
                process_control()
                dataset = fetch_dataset(cfg['data_name'], verbose=False)
                data_split = split_dataset(dataset)
                stats['m_k'] = [len(data_split[i]) for i in range(len(data_split))]
            print(cfg['data_name'])
            print(stats)
