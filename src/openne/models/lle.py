from time import time
import networkx as nx
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from ..utils import *
from .models import *
from sklearn.preprocessing import normalize

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class LLE(ModelWithEmbeddings):

    def __init__(self, dim, **kwargs):
        """ Initialize the LocallyLinearEmbedding class

        Args:
          dim: int
            dimension of the embedding
        """
        super(LLE, self).__init__(dim=dim, **kwargs)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'dim': 128, 'sparse': False})
        check_range(kwargs, {'dim': 'positive', 'sparse': [1, 0, True, False]})
        return kwargs

    def get_train(self, graph, *, sparse=False, **kwargs):
        A = graph.adjmat(directed=False, weighted=True, sparse=sparse)  # todo: check when sparse matrix is better
        normalize(A, norm='l1', axis=1, copy=False)
        if sparse:
            I_n = sp.eye(graph.nodesize())
        else:
            I_n = np.eye(graph.nodesize())  # sp.eye(graph.number_of_nodes())
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, k=self.dim + 1, which='SM')  # todo: check SM or LM
        vt = torch.tensor(vt)
        return vt.t()[:, 1:]