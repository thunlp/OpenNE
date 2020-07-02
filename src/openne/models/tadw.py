from __future__ import print_function
import math
import numpy as np
import torch
from numpy import linalg as la
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
from .models import *


class TADW(ModelWithEmbeddings):

    def __init__(self, dim, lamb=0.2, **kwargs):
        super(TADW, self).__init__(dim=dim//2, lamb=lamb, **kwargs)

    @staticmethod
    def getT(graph):
        g = graph.G
        look_back = graph.look_back_list
        features = torch.from_numpy(graph.features())
        if features.shape[1] > 200:
            U, S, VT = lg.svds(features, k=200)
            Ud = torch.from_numpy(U)
            Sd = torch.from_numpy(S)
            features = Ud * Sd
        return features.t()

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'dim': 128,
                                 'lamb': 0.2,
                                 'epochs': 20})
        assert kwargs['dim'] % 2 == 0
        return kwargs

    @classmethod
    def check_graphtype(cls, graphtype, **kwargs):
        if not graphtype.attributed():
            raise TypeError("TADW only accepts attributed graphs.")

    def build(self, graph, **kwargs):
        self.adj = torch.from_numpy(graph.adjmat(weighted=False, directed=False, scaled=1)).type(torch.float32)
        # M = (A + A^2) / 2, A = adj (row-normalized adjmat)
        self.M = (self.adj + torch.mm(self.adj, self.adj)) / 2
        # T: text feature matrix (feature_size * node_num)
        self.T = self.getT(graph)
        self.node_size = graph.nodesize
        self.feature_size = self.T.shape[0]
        print(self.T.shape, self.node_size)
        self.W = torch.randn(self.dim, self.node_size)
        self.H = torch.randn(self.dim, self.feature_size)

    def get_train(self, graph, **kwargs):  # todo: rewrite with learning-models-based method

        # Update W
        B = torch.mm(self.H, self.T)
        drv = 2 * torch.mm(torch.mm(B, B.t()), self.W) - \
              2 * torch.mm(B, self.M.t()) + self.lamb * self.W
        Hess = 2 * torch.mm(B, B.t()) + self.lamb * torch.eye(self.dim)
        drv = torch.reshape(drv, [self.dim * self.node_size, 1])
        rt = -drv
        dt = rt
        vecW = torch.reshape(self.W, [self.dim * self.node_size, 1])
        while torch.norm(rt, 2) > 1e-4:
            dtS = torch.reshape(dt, (self.dim, self.node_size))
            Hdt = torch.reshape(torch.mm(Hess, dtS), [
                self.dim * self.node_size, 1])

            at = torch.mm(rt.t(), rt) / torch.mm(dt.t(), Hdt)
            vecW = vecW + at * dt
            rtmp = rt
            rt = rt - at * Hdt
            bt = torch.mm(rt.t(), rt) / torch.mm(rtmp.t(), rtmp)  # np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
            dt = rt + bt * dt
        self.W = torch.reshape(vecW, (self.dim, self.node_size))  # np

        # Update H
        drv = torch.mm((torch.mm(torch.mm(torch.mm(self.W, self.W.t()), self.H), self.T)
                        - torch.mm(self.W, self.M.t())), self.T.t()) + self.lamb * self.H
        drv = torch.reshape(drv, (self.dim * self.feature_size, 1))
        rt = -drv
        dt = rt
        vecH = torch.reshape(self.H, (self.dim * self.feature_size, 1))
        while torch.norm(rt, 2) > 1e-4:
            dtS = torch.reshape(dt, (self.dim, self.feature_size))
            Hdt = torch.reshape(torch.mm(torch.mm(torch.mm(self.W, self.W.t()), dtS), torch.mm(self.T, self.T.t()))
                                + self.lamb * dtS, (self.dim * self.feature_size, 1))
            at = torch.mm(rt.t(), rt) / torch.mm(dt.t(), Hdt)
            vecH = vecH + at * dt
            rtmp = rt
            rt = rt - at * Hdt
            bt = torch.mm(rt.t(), rt) / torch.mm(rtmp.t(), rtmp)
            dt = rt + bt * dt
        self.H = torch.reshape(vecH, (self.dim, self.feature_size))

    def make_output(self, graph, **kwargs):
        self.embeddings = torch.cat((
            torch.from_numpy(normalize(self.W.t())),
            torch.from_numpy(normalize(torch.mm(self.T.t(), self.H.t())))), dim=1)
