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
    def __init__(self, dimensions, adj, dropout):
        super(Encoder, self).__init__()
        self.dimensions = dimensions
        self.adj = adj
        self.layers = nn.ModuleList()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(self.dimensions[-1])
        for i in range(1,len(self.dimensions)-1):
            self.layers.append(Linear(self.dimensions[i-1], self.dimensions[i], dropout, act=F.relu))
        self.layers.append(Linear(self.dimensions[-2], self.dimensions[-1], dropout, act=lambda x: x))

    def embed(self, x):
        hx = x
        for layer in self.layers:
            hx = layer(hx)
        return hx

    def forward(self, x, pos, neg):
        hx = x
        hpos = pos
        hneg = neg
        for layer in self.layers:
            hx = layer(hx)
            hpos = layer(hpos)
            hneg = layer(hneg)
        
        pos_score, neg_score = self.disc(hx, hpos, hneg)
        
        logits = torch.cat((pos_score, neg_score), 0)
        return torch.unsqueeze(logits, 0)

class Linear(nn.Module):
    """Linear layer."""

    def __init__(self, input_dim, output_dim, dropout=0., num_features_nonzero=0.,
                 sparse_inputs=False, act=torch.relu, bias=False):
        super(Linear, self).__init__()

        self.dropout = dropout  # note we modified the API
        self.act = act
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


    def forward(self, inputs=None):
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
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        '''
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        '''
        c_x = c
        #print(c.size())
        '''
        h_mi = torch.unsqueeze(h_mi, 1)
        h_mi = h_mi.expand_as(c_x)
        '''
        
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)

        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        #print(sc_1.size(), sc_2.size())
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        #logits = torch.cat((sc_1, sc_2), 1)

        return sc_1, sc_2