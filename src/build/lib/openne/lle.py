from time import time
import networkx as nx
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from . import graph as g
from sklearn.preprocessing import normalize

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class LLE(object):

    def __init__(self, graph, d):
        ''' Initialize the LocallyLinearEmbedding class

        Args:
          graph: nx.DiGraph
            input Graph
          d: int
            dimension of the embedding
        '''
        print("__INIT__")
        self._d = d
        self._method_name = "lle_svd"
        self.g = graph
        self._node_num = graph.node_size
        print("self.learn_embedding")
        self.learn_embedding()

    def learn_embedding(self):
        graph = self.g.G
        graph = graph.to_undirected()
        t1 = time()
        print("start")
        # A = nx.to_scipy_sparse_matrix(graph)
        A = nx.to_numpy_matrix(graph)  # todo: check when sparse matrix is better
        # print(np.sum(A.todense(), axis=0))
        normalize(A, norm='l1', axis=1, copy=False)
        I_n = np.eye(graph.number_of_nodes()) # sp.eye(graph.number_of_nodes())
        I_min_A = I_n - A
        print("I_min_A")
        # print(I_min_A)
        u, s, vt = lg.svds(I_min_A, k=self._d + 1) # , which='SM')
        print("svds")
        vt = torch.tensor(vt)
        t2 = time()
        print("END")
        self._X = vt.t()
        self._X = self._X[:, 1:]
        return self._X, (t2 - t1)
        # I_n = sp.eye(graph.number_of_nodes())

    @property
    def vectors(self):
        print("vectors")
        vectors = {}
        look_back = self.g.look_back_list
        print(len(look_back))
        for i, embedding in enumerate(self._X):
            vectors[look_back[i]] = embedding
        return vectors

    def save_embeddings(self, filename):
        print("save_embeddings")
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self._d))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(float(x)) for x in vec])))
        fout.close()
