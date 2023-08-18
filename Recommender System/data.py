import copy
import torch
import numpy as np
import models
from config import cfg
from scipy.sparse import csr_matrix
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device


def fetch_dataset(data_name, model_name=None, verbose=True):
    import datasets

    model_name = cfg['model_name'] if model_name is None else model_name
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
        dataset['train'] = eval(
            'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], '
            'target_mode=cfg["target_mode"])'.format(data_name))
        dataset['test'] = eval(
            'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], '
            'target_mode=cfg["target_mode"])'.format(data_name))
        transform = InputTransform(cfg['data_mode'])
        dataset['train'].transform = transform
        dataset['test'].transform = transform
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        return {key: [b[key] for b in batch] for key in batch[0]}
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


class InputTransform(torch.nn.Module):
    def __init__(self, data_mode):
        super().__init__()
        self.data_mode = data_mode

    def forward(self, input):
        if self.data_mode == 'user':
            input['user'] = input['user'].repeat(input['item'].size(0))
            input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
        elif self.data_mode == 'item':
            input['item'] = input['item'].repeat(input['user'].size(0))
            input['target_item'] = input['target_item'].repeat(input['target_user'].size(0))
        input['size'] = torch.tensor([input['item'].size(0)])
        input['target_size'] = torch.tensor([input['target_item'].size(0)])
        if 'user_profile' in input:
            del input['user_profile']
        if 'target_user_profile' in input:
            del input['target_user_profile']
        if 'item_attr' in input:
            del input['item_attr']
        if 'target_item_attr' in input:
            del input['target_item_attr']
        else:
            raise ValueError('Not valid data mode')
        return input
