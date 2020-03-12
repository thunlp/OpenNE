import networkx as nx
# import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from . import graph as g
import tensorflow as tf
from sklearn.preprocessing import normalize

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class HOPE(object):
    def __init__(self, graph, d):
        '''
          d: representation vector dimension
        '''
        self._d = d
        self._graph = graph.G
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()

    def learn_embedding(self):

        graph = self.g.G
        A = torch.from_numpy(nx.to_numpy_matrix(graph))  # brute force...

        self._beta = 0.1 # 0.0728

        M_g = torch.eye(graph.number_of_nodes()) - self._beta * A
        M_l = self._beta * A

        # M_g = np.eye(graph.number_of_nodes())
        # M_l = np.dot(A, A)

        S = torch.mm(torch.inverse(M_g),M_l)  # np.dot(np.linalg.inv(M_g), M_l)
        # s: \sigma_k

        ######### let's test the performance of the following 2 algorithms
        # u, s, vt = lg.svds(S, k=self._d // 2) # this one directly use the d/2-dim core for svd
        u, s, vt = torch.svd(S) # this one performs a full svd before cutting dimension to d/2
        u = u[:, 0:self._d//2]
        s = s[0:self._d//2]
        vt = vt[0:self._d//2]

        sigma = torch.diagflat(torch.sqrt(s))
        X1 = torch.mm(u, sigma)
        X2 = torch.mm(vt.t(), sigma)
        # self._X = X2
        # self._X = np.concatenate((X1, X2), axis=1)
        self._X = torch.cat((X1, X2), dim=1)

    @property
    def vectors(self):
        vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self._X):
            vectors[look_back[i]] = embedding
        return vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self._d))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
