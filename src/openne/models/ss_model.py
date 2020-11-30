import numpy as np
from .gcn.utils import *
from .models import *
from .gcn.inits import *
from .ss_encoder import Encoder
from .ss_decoder import Decoder
from .ss_sampler import BaseSampler
import time
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, sampler_name, enc_dims, adj, features, batch_size, dropout=0, dec_dims=None):
        super(SSModel, self).__init__()
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.adj = adj

        self.layers = nn.ModuleList()
        self.sigm = nn.Sigmoid()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.sampler_name = sampler_name
        self.features = features
        self.encoder = Encoder(self.encoder_name, self.enc_dims, self.adj, self.features, dropout)
        self.decoder = Decoder(self.decoder_name, self.enc_dims[-1], self.dec_dims)
        self.sampler = BaseSampler(self.sampler_name, self.adj, batch_size)

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x, pos, neg):
        
        hx = self.embed(x)
        hpos = self.embed(pos)
        hneg = self.embed(neg)
        pos_score, neg_score = self.decoder(hx, hpos), self.decoder(hx, hneg)
        logits = torch.cat((pos_score, neg_score), 0)

        return torch.unsqueeze(logits, 0)
    
    def sample(self):
        return next(self.sampler)
    