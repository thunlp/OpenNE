from __future__ import print_function
import torch
import numpy as np
from .models import *
from ..utils import *

__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


class GraphFactorization(ModelWithEmbeddings):
    def __init__(self, dim=128, **kwargs):
        super(GraphFactorization, self).__init__(dim=dim, **kwargs)

    othername = 'gf'

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'epochs': 130, 'lr': 0.003, 'weight_decay': 1., 'dim': 128})
        check_range(kwargs, {'epochs': 'positive', 'lr': 'positive',
                             'weight_decay': 'positive', 'dim': 'positive'})
        return kwargs

    def build(self, graph, *, lr=0.003, **kwargs):
        self.register_buffer('adj_mat', torch.from_numpy(graph.adjmat(directed=True, weighted=True)).to(torch.float32))
        self.register_buffer('mat_mask', torch.as_tensor(self.adj_mat > 0, dtype=torch.float32))

        self.register_parameter('_embeddings', torch.nn.init.xavier_uniform_(torch.nn.Parameter(
            torch.zeros(graph.nodesize, self.dim, dtype=torch.float32), True)))
        # print(_embeddings)
        self.optimizer = torch.optim.Adam([self._embeddings], lr=lr)

    def train_model(self, graph, *, weight_decay=1., **kwargs):
        self.optimizer.zero_grad()
        cost = ((self.adj_mat - torch.mm(self._embeddings, self._embeddings.t()) * self.mat_mask) ** 2).sum() + \
               weight_decay * ((self._embeddings ** 2).sum())
        cost.backward()
        self.optimizer.step()
        self.debug_info = "cost: {}".format(float(cost))
        return self._embeddings.detach()
