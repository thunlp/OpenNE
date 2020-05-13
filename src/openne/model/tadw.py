from __future__ import print_function
import math
import numpy as np
import torch
from numpy import linalg as la
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
from .gcn.utils import *


class TADW(object):

    def __init__(self, graph, dim, lamb=0.2, iters=20):
        self.g = graph
        self.lamb = lamb
        self.dim = int(dim/2)
        self.iters=iters
        self.train()

    def getAdj(self):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        # adj = np.zeros((node_size, node_size))
        adj = torch.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat
        return adj / adj.sum(dim=1)
        # return adj/np.sum(adj, axis=1)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim*2))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(float(x)) for x in vec])))
        fout.close()

    def getT(self):
        g = self.g.G
        look_back = self.g.look_back_list
        self.features = torch.from_numpy(
                            np.vstack([g.nodes[look_back[i]]['feature']
                                   for i in range(g.number_of_nodes())])
                            )
        self.preprocessFeature()
        return self.features.t()

    def preprocessFeature(self):
        if self.features.shape[1] > 200:
            # U, S, VT = torch.svd(self.features)
            # Ud = U[:, 0:200]
            # Sd = S[0:200]

            U, S, VT = lg.svds(self.features, k=200)
            Ud = torch.from_numpy(U)
            Sd = torch.from_numpy(S)

            self.features = Ud*Sd  #.reshape(200)

    def train(self): # todo: rewrite with learning-model-based method
        self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = (self.adj + torch.mm(self.adj, self.adj))/2
        # T is feature_size*node_num, text features
        self.T = self.getT()
        self.node_size = self.adj.shape[0]
        self.feature_size = self.features.shape[1]
        self.W = torch.randn(self.dim, self.node_size) # np.random.randn(self.dim, self.node_size)
        self.H = torch.randn(self.dim, self.feature_size)  # np.random.randn(self.dim, self.feature_size)
        # Update
        for i in range(self.iters):
            print('Iteration ', i)
            # Update W
            B = torch.mm(self.H, self.T)  # np.dot(self.H, self.T)
            drv = 2 * torch.mm(torch.mm(B, B.t()), self.W) - \
                  2 * torch.mm(B, self.M.t()) + self.lamb * self.W
                # 2 * np.dot(np.dot(B, B.T), self.W) - \
                # 2*np.dot(B, self.M.T) + self.lamb*self.W
            Hess = 2 * torch.mm(B, B.t()) + self.lamb * torch.eye(self.dim)  # 2*np.dot(B, B.T) + self.lamb*np.eye(self.dim)
            drv = torch.reshape(drv, [self.dim*self.node_size, 1]) # np.reshape(drv, [self.dim*self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = torch.reshape(self.W, [self.dim*self.node_size, 1])  # np.reshape(self.W, [self.dim*self.node_size, 1])
            while torch.norm(rt, 2) > 1e-4: #  np.linalg.norm(rt, 2) > 1e-4:
                dtS = torch.reshape(dt, (self.dim, self.node_size)) # np.reshape(dt, (self.dim, self.node_size))
                Hdt = torch.reshape(torch.mm(Hess, dtS), [
                                 self.dim*self.node_size, 1]) #  np.reshape(np.dot(Hess, dtS), [self.dim*self.node_size, 1])

                at = torch.mm(rt.t(), rt)/torch.mm(dt.t(), Hdt)  # np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecW = vecW + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = torch.mm(rt.t(), rt)/torch.mm(rtmp.t(), rtmp)  # np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = torch.reshape(vecW, (self.dim, self.node_size))  #  np

            # Update H
            drv = torch.mm((torch.mm(torch.mm(torch.mm(self.W, self.W.t()), self.H), self.T)
                          - torch.mm(self.W, self.M.t())), self.T.t()) + self.lamb*self.H
            #np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
            #              - np.dot(self.W, self.M.T)), self.T.T) + self.lamb*self.H
            drv = torch.reshape(drv, (self.dim*self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = torch.reshape(self.H, (self.dim*self.feature_size, 1))
            while torch.norm(rt, 2) > 1e-4:
                dtS = torch.reshape(dt, (self.dim, self.feature_size))
                Hdt = torch.reshape(torch.mm(torch.mm(torch.mm(self.W, self.W.t()), dtS), torch.mm(self.T, self.T.t()))
                                 + self.lamb*dtS, (self.dim*self.feature_size, 1))
                at = torch.mm(rt.t(), rt)/torch.mm(dt.t(), Hdt)
                vecH = vecH + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = torch.mm(rt.t(), rt)/torch.mm(rtmp.t(), rtmp)
                dt = rt + bt * dt
            self.H = torch.reshape(vecH, (self.dim, self.feature_size))
        self.Vecs = torch.cat((
            torch.from_numpy(normalize(self.W.t())),
            torch.from_numpy(normalize(torch.mm(self.T.t(), self.H.t())))), dim=1)
         # np.hstack(
            # (normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))
        # get embeddings
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.Vecs):
            self.vectors[look_back[i]] = embedding
