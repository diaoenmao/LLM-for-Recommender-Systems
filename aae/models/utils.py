import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def loss_fn(output, target, reduction='mean'):
    if cfg['target_mode'] == 'implicit':
        loss = F.binary_cross_entropy_with_logits(output, target, reduction=reduction)
    elif cfg['target_mode'] == 'explicit':
        loss = F.mse_loss(output, target, reduction=reduction)
    else:
        raise ValueError('Not valid target mode')
    return loss


def distribute(model, local_model, data_split):
    if cfg['model_name'] == 'base':
        for k, v in model.state_dict().items():
            for i in range(len(local_model)):
                local_model_state_dict_i = local_model[i].state_dict()
                local_model_state_dict_i[k].data.copy_(v[data_split[i]])
    elif cfg['model_name'] in ['mf', 'mlp', 'nmf']:
        for k, v in model.state_dict().items():
            for i in range(len(local_model)):
                local_model_state_dict_i = local_model[i].state_dict()
                if cfg['data_mode'] == 'user' and 'item' in k:
                    local_model_state_dict_i[k].data.copy_(v[data_split[i]])
                elif cfg['data_mode'] == 'item' and 'user' in k:
                    local_model_state_dict_i[k].data.copy_(v[data_split[i]])
                else:
                    local_model_state_dict_i[k].data.copy_(v)
    elif cfg['model_name'] == 'ae':
        for k, v in model.state_dict().items():
            for i in range(len(local_model)):
                local_model_state_dict_i = local_model[i].state_dict()
                local_model_state_dict_i[k].data.copy_(v)
    else:
        raise ValueError('Nto valid model')
    return
