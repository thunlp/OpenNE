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
    def __init__(self, rep_size=128):
        super(LaplacianEigenmaps, self).__init__(rep_size=rep_size)

    def get_train(self, graph, **kwargs):
        adj_mat = graph.adjmat(directed=True, weighted=False)
        lap_mat = getLap(adj_mat)
        w, vec = torch.symeig(lap_mat, eigenvectors=True)
        start = 0
        for i in range(graph.node_size):
            if w[i] > 1e-10:
                start = i
                break
        vec = vec[:, start:start+self.rep_size]
        return vec
