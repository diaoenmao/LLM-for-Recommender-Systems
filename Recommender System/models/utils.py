import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def normalize(x, xmin, xmax):
    output = 2 * (x - xmin) / (xmax - xmin) - 1
    return output


def denormalize(x, xmin, xmax):
    output = ((xmax - xmin) / 2) * x + (xmax + xmin) / 2
    return output


def loss_fn(output, target, reduction='mean'):
    if cfg['target_mode'] == 'explicit':
        loss = F.mse_loss(output, target, reduction=reduction)
    elif cfg['target_mode'] == 'implicit':
        loss = F.binary_cross_entropy_with_logits(output, target, reduction=reduction)
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


def cusum_size(input, size, dim=0):
    # Compute the cumulative sums of the tensor along the specified dimension
    cumsums = torch.cumsum(input, dim=dim)
    # Get the indices for slicing based on size tensor along the specified dimension
    indices = torch.cumsum(size, dim=0) - 1
    indices[indices < 0] = 0
    # Use advanced indexing to extract the required sums from cumsums
    summed_values = torch.index_select(cumsums, dim, indices)
    # Adjust the sums to get the desired results
    shifted_summed_values = torch.roll(summed_values, shifts=1, dims=dim)
    output = summed_values - shifted_summed_values
    # Set the first value in the dimension to be the same as summed_values
    output.select(dim, 0).copy_(summed_values.select(dim, 0))
    return output