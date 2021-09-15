import numpy as np
from .gcn.utils import *
from .models import *
from .gcn.inits import *
import time
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, name, dimensions, adj, features, dropout):
        super(Encoder, self).__init__()
        self.dimensions = dimensions
        self.adj = adj
        self.layers = nn.ModuleList()
        self.features = features
        self.nnodes = self.features.size()[0]
        self.sigm = nn.Sigmoid()
        self.name = name
        if name == 'none':
            self.embedding = nn.Embedding(self.nnodes, self.dimensions[-1])
        else:
            for i in range(1, len(self.dimensions)-1):
                self.layers.append(layer_dict[name](self.dimensions[i-1], self.dimensions[i], self.adj, dropout, act=F.relu))
            self.layers.append(layer_dict[name](self.dimensions[-2], self.dimensions[-1], self.adj, dropout, act=lambda x: x))

    def embed(self, x):
        if self.name == 'none':
            return self.embedding(x)
        else:
            return self.features[x]
    
    def forward(self, x):
        hx = self.embed(x)
        if self.name != 'none':
            for layer in self.layers:
                hx = layer(hx)
        return hx

"Layers"

class Linear(nn.Module):
    """Linear layer."""

    def __init__(self, input_dim, output_dim, adj=None, dropout=0., num_features_nonzero=0.,
                 sparse_inputs=False, act=torch.relu, bias=False):
        super(Linear, self).__init__()

        self.dropout = dropout  # note we modified the API
        self.act = act
        self.adj = adj
        self.sparse_inputs = sparse_inputs
        self.output_dim = output_dim
        self.input_dim = input_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero
        self.logging = False

        self.act = act
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        
        if bias:
            self.bias = zeros([output_dim])
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        x = inputs
        if self.training:
            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
            else:
                x = torch.dropout(x, self.dropout, True)
        elif self.sparse_inputs:
            x = tuple_to_sparse(x)
        
        pre_sup = torch.mm(x, self.weight)
        output = pre_sup
            
        # bias
        if self.bias is not None:
            output += self.bias
        return self.act(output)


layer_dict = {
    "linear": Linear
}

'''
TO DO:
    ● GCN
    ● GAT
    ● GIN
'''