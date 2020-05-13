from __future__ import print_function
import torch
import numpy as np
from .models import *
from ..utils import *

__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


class GraphFactorization(ModelWithEmbeddings):
    def __init__(self, rep_size=128):
        super(GraphFactorization, self).__init__(rep_size=rep_size)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'epoch': 120, 'learning_rate': 0.003, 'weight_decay': 1.})
        check_range(kwargs, {'epoch': (0, np.inf), 'learning_rate': (0, np.inf), 'weight_decay': (0, np.inf)})

    def build(self, graph, *, learning_rate=0.003, **kwargs):
        self.adj_mat = torch.from_numpy(graph.adjmat(directed=True, weighted=True))
        self.mat_mask = torch.as_tensor(self.adj_mat > 0, dtype=torch.float32)

        self._embeddings = torch.tensor(torch.nn.init.xavier_uniform_(torch.zeros(graph.nodesize, self.rep_size,
                                                                             dtype=torch.float32)),
                                   requires_grad=True)
        # print(_embeddings)
        self.optimizer = torch.optim.Adam([self._embeddings], lr=learning_rate)

    def get_train(self, graph, *, weight_decay=1., **kwargs):
        self.optimizer.zero_grad()
        cost = ((self.adj_mat - torch.mm(self._embeddings, self._embeddings.t()) * self.mat_mask) ** 2).sum() \
            + weight_decay * ((self._embeddings ** 2).sum())
        cost.backward()
        self.optimizer.step()
        self.debug_info = "cost: {}".format(float(cost))
        return self._embeddings.detach()
