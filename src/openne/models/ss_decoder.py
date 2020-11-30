import numpy as np
from .gcn.utils import *
from .models import *
from .gcn.inits import *
from .ss_encoder import Linear
import time
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F




class Decoder(nn.Module):
    def __init__(self, name, dim, mlp_dim = None):
        super(Decoder, self).__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.layer = disc_dict[name](dim, mlp_dim)

    def forward(self, x, y):
        score = self.layer(x, y)
        return score

class InnerProd(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(InnerProd, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        score = torch.sum((x * y), dim=1)
        return score

class Bilinear(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(Bilinear, self).__init__()
        self.dim = dim
        self.bil = nn.Bilinear(dim, dim, 1)

    def forward(self, x, y):
        score = torch.squeeze(self.bil(x, y), dim=-1)
        return score

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(MLP, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.mlp_dim = mlp_dim
        for i in range(1, len(self.mlp_dim)-1):
            self.layers.append(Linear(self.mlp_dim[i-1], self.mlp_dim[i], act=F.relu))
        self.layers.append(Linear(self.mlp_dim[-2], self.mlp_dim[-1], act=lambda x: x))

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        for layer in self.layers:
            h = layer(h)
        return torch.squeeze(h, dim=-1)

disc_dict = {
    "inner": InnerProd,
    "bilinear": Bilinear,
    "mlp": MLP
}