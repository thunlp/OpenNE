import networkx as nx
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from . import graph as g
import tensorflow as tf
from time import time
from sklearn.preprocessing import normalize

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class HOPE(object):
    def __init__(self, graph, d):
        '''
          d: representation vector dimension
        '''
        t1=time()
        self._d = d
        self._graph = graph.G
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()
        t2=time()
        print("TIME used: ",t2-t1)

    def learn_embedding(self):
        graph = self.g.G
        A = (nx.to_numpy_matrix(graph))  # brute force...
        # Katz: M_g^-1 * M_l = (I - beta * A)^-1 - I
        self.beta = 0.01 # 0.0728
        n = graph.number_of_nodes()
        # S = np.asarray((np.eye(n) - self.beta * np.mat(A)).I - np.eye(n))

        # M_g = torch.eye(graph.number_of_nodes(),dtype=torch.float64) - self._beta * A
        # M_l = self._beta * A

        # common neighbours
        M_g = torch.eye(graph.number_of_nodes(), dtype=torch.float64)
        #print("MG")
        A=torch.tensor(A)
        M_l = torch.mm(A, A)
        #print("ML")
        S = torch.mm(torch.inverse(M_g), M_l)  # np.dot(np.linalg.inv(M_g), M_l)
        # s: \sigma_k

        # print("d=",self._d//2)
        u, s, vt = lg.svds(S, k=self._d // 2)  # this one directly use the d/2-dim core for svd

        sigma = np.diagflat(np.sqrt(s))
        X1 = normalize(np.matmul(u, sigma))
        X2 = normalize(np.matmul(vt.T, sigma))
        self._X = torch.cat((torch.tensor(X1), torch.tensor(X2)), dim=1)


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
                                        ' '.join([str(float(x)) for x in vec])))
        fout.close()
