import numpy as np
from .gcn.utils import *
from .models import *
from .gcn.inits import *
import time
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class BaseSampler(Sampler):
    def __init__(self, name, adj, batch_size):
        #super(BaseSampler, self).__init__()
        self.adj = adj
        self.nnodes = self.adj.size()[0]
        self.nedges = self.adj._indices().size()[1]
        self.batch_size = batch_size
        self.sampler = sampler_dict[name](adj, self.nnodes, self.nedges)
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    
class NNRSampler(Sampler):
    def __init__(self, adj, nnodes, nedges):
        #super(NNRSampler, self).__init__()
        self.nnodes = nnodes
        self.nedges = nedges
        self.adj = adj
        self.adj_ind = self.adj._indices()
    
    def __iter__(self):
        xind = self.adj_ind[0]
        yind = self.adj_ind[1]
        negind = torch.randint(high=self.nnodes, size=(self.nedges,)).tolist()
        return iter(list(zip(xind, yind, negind)))

sampler_dict = {
    "node-nei-random": NNRSampler
}

'''
TO DO:
    ● node-random walk-random nodes (DeepWalk)
    ● node-neighborhood-except neighborhood (GAE)
    ● graph-node-permuted nodes (DGI)
    ● node-random walk-except neighborhood
'''