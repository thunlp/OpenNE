import torch
from ..utils import *
from .models import *

__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


def getLap(adj_mat):
    degree_mat = torch.diagflat(adj_mat.sum(1))
    deg_trans = torch.diagflat(torch.reciprocal(torch.sqrt(adj_mat.sum(1))))
    deg_trans[torch.isnan(deg_trans)] = 0
    L = degree_mat - adj_mat
    norm_lap_mat = torch.mm(torch.mm(deg_trans, L), deg_trans)
    return norm_lap_mat


class LaplacianEigenmaps(ModelWithEmbeddings):
    def __init__(self, dim=128, **kwargs):
        super(LaplacianEigenmaps, self).__init__(dim=dim, **kwargs)
    othername = 'lap'
    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'dim': 128})
        check_range(kwargs, {'dim': 'positive'})
        return kwargs

    def train_model(self, graph, **kwargs):
        adj_mat = torch.from_numpy(graph.adjmat(directed=True, weighted=False))
        lap_mat = getLap(adj_mat)
        w, vec = torch.symeig(lap_mat, eigenvectors=True)
        start = 0
        for i in range(graph.nodesize):
            if w[i] > 1e-10:
                start = i
                break
        vec = vec[:, start:start+self.dim]
        return vec
