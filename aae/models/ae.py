import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        blocks = [nn.Linear(input_size, hidden_size[0]),
                  nn.Tanh()]
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        blocks.append(nn.Linear(hidden_size[-1], output_size))
        blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class AE(nn.Module):
    def __init__(self, encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
                 decoder_hidden_size, info_size):
        super().__init__()
        self.info_size = info_size
        if len(encoder_hidden_size) > 1:
            self.encoder = Encoder(encoder_hidden_size[0], encoder_hidden_size[1:])
            self.decoder = Decoder(decoder_hidden_size[-1], decoder_hidden_size[:-1])
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        if cfg['data_mode'] == 'user':
            self.encoder_linear = nn.Linear(encoder_num_items, encoder_hidden_size[0])
            self.decoder_linear = nn.Linear(decoder_hidden_size[-1], decoder_num_items)
        elif cfg['data_mode'] == 'item':
            self.encoder_linear = nn.Linear(encoder_num_users, encoder_hidden_size[0])
            self.decoder_linear = nn.Linear(decoder_hidden_size[-1], decoder_num_users)
        else:
            raise ValueError('Not valid data mode')
        self.dropout = nn.Dropout(p=0.5)
        if info_size is not None:
            if 'user_profile' in info_size:
                self.user_profile = Encoder(info_size['user_profile'], encoder_hidden_size)
            if 'item_attr' in info_size:
                self.item_attr = Encoder(info_size['item_attr'], encoder_hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder_linear.weight)
        if self.encoder_linear.bias is not None:
            self.encoder_linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.decoder_linear.weight)
        if self.decoder_linear.bias is not None:
            self.decoder_linear.bias.data.zero_()
        return

    def forward(self, input):
        output = {}
        if cfg['data_mode'] == 'user':
            total_user = torch.unique(torch.cat([input['user'], input['target_user']]), sorted=True)
            encoder_item_weight = self.encoder_linear.weight.t()[input['item']]
            sorted_user, sorted_user_idx = torch.sort(input['user'])
            user, user_count = torch.unique_consecutive(sorted_user, return_counts=True)
            mask = torch.isin(total_user, user)
            user_indices = torch.arange(len(user_count), device=user.device).repeat_interleave(user_count)
            x = encoder_item_weight * input['rating'].view(-1, 1)
            x_ = torch.zeros((len(total_user), encoder_item_weight.size(-1)), device=user.device)
            x_[mask] = x_[mask].index_add(0, user_indices, x[sorted_user_idx])
            x = torch.tanh(x_ + self.encoder_linear.bias)
        elif cfg['data_mode'] == 'item':
            total_item = torch.unique(torch.cat([input['item'], input['target_item']]), sorted=True)
            encoder_user_weight = self.encoder_linear.weight.t()[input['user']]
            sorted_item, sorted_item_idx = torch.sort(input['item'])
            item, item_count = torch.unique_consecutive(sorted_item, return_counts=True)
            mask = torch.isin(total_item, item)
            item_indices = torch.arange(len(item_count), device=item.device).repeat_interleave(item_count)
            x = encoder_user_weight * input['rating'].view(-1, 1)
            x_ = torch.zeros((len(total_item), encoder_user_weight.size(-1)), device=item.device)
            x_[mask] = x_[mask].index_add(0, item_indices, x[sorted_item_idx])
            x = torch.tanh(x_ + self.encoder_linear.bias)
        encoded = self.encoder(x)
        if self.info_size is not None:
            if 'user_profile' in input:
                user_profile = input['user_profile']
                user_profile = self.user_profile(user_profile)
                encoded = encoded + user_profile
            if 'item_attr' in input:
                item_attr = input['item_attr']
                item_attr = self.item_attr(item_attr)
                encoded = encoded + item_attr
        code = self.dropout(encoded)
        decoded = self.decoder(code)
        if cfg['data_mode'] == 'user':
            decoder_target_item_weight = self.decoder_linear.weight[input['target_item']]
            decoder_target_item_bias = self.decoder_linear.bias[input['target_item']]
            sorted_target_user, sorted_target_user_idx = torch.sort(input['target_user'])
            _, inverse_idx = torch.sort(sorted_target_user_idx)
            target_user, target_user_count = torch.unique_consecutive(sorted_target_user, return_counts=True)
            target_mask = torch.isin(total_user, target_user)
            x = decoded[target_mask].repeat_interleave(target_user_count, dim=0)[inverse_idx]
            x = (x * decoder_target_item_weight).sum(dim=-1) + decoder_target_item_bias
        elif cfg['data_mode'] == 'item':
            decoder_target_user_weight = self.decoder_linear.weight[input['target_user']]
            decoder_target_user_bias = self.decoder_linear.bias[input['target_user']]
            sorted_target_item, sorted_target_item_idx = torch.sort(input['target_item'])
            _, inverse_idx = torch.sort(sorted_target_item_idx)
            target_item, target_item_count = torch.unique_consecutive(sorted_target_item, return_counts=True)
            target_mask = torch.isin(total_item, target_item)
            x = decoded[target_mask].repeat_interleave(target_item_count, dim=0)[inverse_idx]
            x = (x * decoder_target_user_weight).sum(dim=-1) + decoder_target_user_bias
        output['target_rating'] = x
        if 'local' in input and input['local']:
            output['loss'] = F.mse_loss(output['target_rating'], input['target_rating'])
        else:
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        return output


def ae(encoder_num_users=None, encoder_num_items=None, decoder_num_users=None, decoder_num_items=None):
    encoder_num_users = cfg['num_users']['data'] if encoder_num_users is None else encoder_num_users
    encoder_num_items = cfg['num_items']['data'] if encoder_num_items is None else encoder_num_items
    decoder_num_users = cfg['num_users']['target'] if decoder_num_users is None else decoder_num_users
    decoder_num_items = cfg['num_items']['target'] if decoder_num_items is None else decoder_num_items
    encoder_hidden_size = cfg['ae']['encoder_hidden_size']
    decoder_hidden_size = cfg['ae']['decoder_hidden_size']
    info_size = cfg['info_size']
    model = AE(encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
               decoder_hidden_size, info_size)
    return model
