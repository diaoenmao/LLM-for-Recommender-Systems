import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MDR(nn.Module):
    def __init__(self, model, data_mode, model_name):
        super().__init__()
        model_list = []
        for m in range(len(model)):
            model_list.append(self.make_share(model[0], model[m], data_mode, model_name))
        self.model_list = nn.ModuleList(model_list)

    def make_share(self, model_0, model_m, data_mode, model_name):
        if model_name in ['mf', 'mlp']:
            if data_mode == 'user':
                model_m.user_weight = model_0.user_weight
                model_m.user_bias = model_0.user_bias
                if model_0.info_size is not None:
                    if 'user_profile' in model_0.info_size:
                        model_m.user_profile = model_0.user_profile
            elif data_mode == 'item':
                model_m.item_weight = model_0.item_weight
                model_m.item_bias = model_0.item_bias
                if model_0.info_size is not None:
                    if 'item_attr' in model_0.info_size:
                        model_m.item_attr = model_0.item_attr
            else:
                raise ValueError('Not valid data mode')
        elif model_name == 'nmf':
            if data_mode == 'user':
                model_m.user_weight_mlp = model_0.user_weight_mlp
                model_m.user_bias_mlp = model_0.user_bias_mlp
                model_m.user_weight_mf = model_0.user_weight_mf
                model_m.user_bias_mf = model_0.user_bias_mf
                if model_0.info_size is not None:
                    if 'user_profile' in model_0.info_size:
                        model_m.user_profile_mf = model_0.user_profile_mf
                        model_m.user_profile_mlp = model_0.user_profile_mlp
            elif data_mode == 'item':
                model_m.item_weight_mlp = model_0.item_weight_mlp
                model_m.item_bias_mlp = model_0.item_bias_mlp
                model_m.item_weight_mf = model_0.item_weight_mf
                model_m.item_bias_mf = model_0.item_bias_mf
                if model_0.info_size is not None:
                    if 'item_attr' in model_0.info_size:
                        model_m.item_attr_mf = model_0.item_attr_mf
                        model_m.item_attr_mlp = model_0.item_attr_mlp
            else:
                raise ValueError('Not valid data mode')
        return model_m

    def forward(self, input, m):
        output = self.model_list[m](input)
        return output


def mdr(model):
    data_mode = cfg['data_mode']
    model_name = cfg['model_name']
    model = MDR(model, data_mode, model_name)
    return model
