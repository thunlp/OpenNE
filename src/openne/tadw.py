from __future__ import print_function
import math
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize
from .gcn.utils import *


class TADW(object):

    def __init__(self, graph, dim, lamb=0.2):
        self.g = graph
        self.lamb = lamb
        self.dim = int(dim/2)
        self.train()

    def getAdj(self):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat
        return adj/np.sum(adj, axis=1)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim*2))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

    def getT(self):
        g = self.g.G
        look_back = self.g.look_back_list
        self.features = np.vstack([g.nodes[look_back[i]]['feature']
                                   for i in range(g.number_of_nodes())])
        self.preprocessFeature()
        return self.features.T

    def preprocessFeature(self):
        if self.features.shape[1] > 200:
            U, S, VT = la.svd(self.features)
            Ud = U[:, 0:200]
            Sd = S[0:200]
            self.features = np.array(Ud)*Sd.reshape(200)

    def train(self):
        self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = (self.adj + np.dot(self.adj, self.adj))/2
        # T is feature_size*node_num, text features
        self.T = self.getT()
        self.node_size = self.adj.shape[0]
        self.feature_size = self.features.shape[1]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)
        # Update
        for i in range(20):
            print('Iteration ', i)
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                2*np.dot(B, self.M.T) + self.lamb*self.W
            Hess = 2*np.dot(B, B.T) + self.lamb*np.eye(self.dim)
            drv = np.reshape(drv, [self.dim*self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim*self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [
                                 self.dim*self.node_size, 1])

                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecW = vecW + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb*self.H
            drv = np.reshape(drv, (self.dim*self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim*self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb*dtS, (self.dim*self.feature_size, 1))
                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecH = vecH + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))
        self.Vecs = np.hstack(
            (normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))
        # get embeddings
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.Vecs):
            self.vectors[look_back[i]] = embedding
