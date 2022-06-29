import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class Assist(nn.Module):
    def __init__(self, ar, ar_mode, num_organizations, aw_mode):
        super().__init__()
        self.ar_mode = ar_mode
        self.aw_mode = aw_mode
        if self.ar_mode == 'optim':
            self.assist_rate = nn.Parameter(torch.tensor(ar))
        elif self.ar_mode == 'constant':
            self.register_buffer('assist_rate', torch.tensor(ar))
        else:
            raise ValueError('Not valid ar mode')
        if self.aw_mode == 'optim':
            self.assist_weight = nn.Parameter(torch.ones(num_organizations) / num_organizations)
        elif self.aw_mode == 'constant':
            self.register_buffer('assist_weight', torch.ones(num_organizations) / num_organizations)
        else:
            raise ValueError('Not valid aw mode')

    def forward(self, input):
        output = {}
        aggregated_output = None
        weight = self.assist_weight.softmax(-1)
        for i in range(len(input['output'])):
            if aggregated_output is None:
                aggregated_output = input['output'][i] * weight[i]
            else:
                min_length = min(len(aggregated_output), len(input['output'][i]))
                if len(aggregated_output) > len(input['output'][i]):
                    aggregated_output[:min_length] += input['output'][i] * weight[i]
                else:
                    tmp_aggregated_output = aggregated_output.clone()
                    aggregated_output = (input['output'][i] * weight[i]).clone()
                    aggregated_output[:min_length] += tmp_aggregated_output
        output['target'] = input['history'] + self.assist_rate * aggregated_output
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def assist():
    ar = cfg['assist']['ar']
    ar_mode = cfg['assist']['ar_mode']
    num_organizations = cfg['num_organizations']
    aw_mode = cfg['assist']['aw_mode']
    model = Assist(ar, ar_mode, num_organizations, aw_mode)
    return model
