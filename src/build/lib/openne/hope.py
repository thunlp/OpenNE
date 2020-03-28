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

        t1=time()
        graph = self.g.G
        # A = (nx.adjacency_matrix(graph).todense())  # brute force...
        # Katz
        self.beta = 0.01 # 0.0728
        adj = nx.adjacency_matrix(graph).todense()
        t2=time()
        n = adj.shape[0]
        print(n,"n=n")
        katz_matrix = np.asarray((np.eye(n) - self.beta * np.mat(adj)).I - np.eye(n))
        # n=graph.number_of_nodes()
        # M_g = torch.eye(graph.number_of_nodes(),dtype=torch.float64) - self._beta * A
        # M_l = self._beta * A

        # common neighbours
        #M_g = torch.eye(graph.number_of_nodes(), dtype=torch.float64)
        #print("MG")
        #M_l = torch.mm(A, A)
        #print("ML")
        # S = torch.mm(torch.inverse(M_g), M_l)  # np.dot(np.linalg.inv(M_g), M_l)
        # s: \sigma_k
        #Katz
        #S=np.asarray((np.eye(n) - self._beta * np.mat(A)).I - np.eye(n))
        t3=time()
        print("d=",self._d//2)
        u, s, vt = lg.svds(katz_matrix, k=self._d // 2)  # this one directly use the d/2-dim core for svd
        # u = torch.from_numpy(u)
        # s = torch.from_numpy(s)
        # vt = torch.from_numpy(vt)
        t4=time()
        sigma = np.sqrt(s) # torch.diagflat(torch.sqrt(s))
        X1 = normalize(u * sigma) #torch.mm(u, sigma).nor
        X2 = normalize(vt.T * sigma) # torch.mm(vt.t(), sigma)
        t5=time()
        # self._X = X2
        # self._X = np.concatenate((X1, X2), axis=1)
        self._X = torch.cat((torch.tensor(X1), torch.tensor(X2)), dim=1)
        t6=time()
        print("TIME=",t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t6-t1)

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
