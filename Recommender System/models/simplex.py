import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn, normalize, denormalize, cusum_size
from config import cfg


class SimpleX(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.user_weight = nn.Embedding(num_users, hidden_size)
        self.item_weight = nn.Embedding(num_items, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
        return

    def user_embedding(self, user):
        embedding = self.user_weight(user)
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item)
        return embedding

    def forward(self, input):
        output = {}
        if self.training:
            user = input['user']
            target_user = input['target_user']
            item = input['item']
            target_item = input['target_item']
            rating = input['rating'].clone().detach()
            if cfg['target_mode'] == 'explicit':
                rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
            size = input['size']
            target_size = input['target_size']
        else:
            user = input['user']
            target_user = input['target_user']
            item = input['item']
            target_item = input['target_item']
            rating = input['target_rating'].clone().detach()
            if cfg['target_mode'] == 'explicit':
                rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
            size = input['size']
            target_size = input['target_size']

        if cfg['data_mode'] == 'user':
            user_embedding = self.user_embedding(user)[torch.cumsum(size, dim=0) - 1]
            item_embedding = self.item_embedding(item)
            item_embedding_cusum_size = cusum_size(item_embedding, size) / (size.unsqueeze(-1) + 1e-6)
            item_embedding_mean = item_embedding_cusum_size
            embedding = 0.5 * user_embedding + 0.5 * item_embedding_mean
            embedding = torch.repeat_interleave(embedding, target_size, dim=0)
            target_item_embedding = self.item_embedding(target_item)
            embedding = F.normalize(embedding - embedding.mean(dim=-1, keepdims=True), dim=-1)
            target_item_embedding = F.normalize(target_item_embedding -
                                                target_item_embedding.mean(dim=-1,
                                                                           keepdims=True), dim=-1)
            simplex = torch.bmm(embedding.unsqueeze(1), target_item_embedding.unsqueeze(-1)).squeeze()
        elif cfg['data_mode'] == 'item':
            item_embedding = self.item_embedding(item)[torch.cumsum(size, dim=0) - 1]
            user_embedding = self.user_embedding(user)
            user_embedding_cusum_size = cusum_size(user_embedding, size)
            user_embedding_mean = user_embedding_cusum_size / (size.unsqueeze(-1) + 1e-6)
            embedding = 0.5 * item_embedding + 0.5 * user_embedding_mean
            embedding = torch.repeat_interleave(embedding, target_size, dim=0)
            target_user_embedding = self.user_embedding(target_user)
            embedding = F.normalize(embedding - embedding.mean(dim=-1, keepdims=True), dim=-1)
            target_user_embedding = F.normalize(target_user_embedding -
                                                target_user_embedding.mean(dim=-1,
                                                                           keepdims=True), dim=-1)
            simplex = torch.bmm(embedding.unsqueeze(1), target_user_embedding.unsqueeze(-1)).squeeze()
        else:
            raise ValueError('Not valid data mode')
        simplex = simplex.view(-1)
        output['loss'] = loss_fn(simplex, rating)
        output['target_rating'] = simplex
        if cfg['target_mode'] == 'explicit':
            output['target_rating'] = denormalize(output['target_rating'], cfg['stats']['min'], cfg['stats']['max'])
        return output


def simplex():
    num_users = cfg['num_users']['data']
    num_items = cfg['num_items']['data']
    hidden_size = cfg['simplex']['hidden_size']
    model = SimpleX(num_users, num_items, hidden_size)
    return model
