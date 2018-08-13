#pylint: disable=E1101
from time import time
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from . import graph as g
from sklearn.preprocessing import normalize


class LLE(object):

  def __init__(self, graph: g.Graph, d: int):
    ''' Initialize the LocallyLinearEmbedding class

    Args:
      graph: nx.DiGraph
        input Graph
      d: int
        dimension of the embedding
    '''
    # hyper_params = {
    #     'method_name': 'lle_svd'
    # }
    # hyper_params.update(kwargs)
    # for key in hyper_params.keys():
    #     self.__setattr__('_%s' % key, hyper_params[key])
    # for dictionary in hyper_dict:
    #     for key in dictionary:
    #         self.__setattr__('_%s' % key, dictionary[key])
    self._d = d
    self._method_name = "lle_svd"
    self.g = graph
    self._node_num = graph.node_size
    self.learn_embedding()

  def learn_embedding(self):
    graph = self.g.G
    graph = graph.to_undirected()
    t1 = time()
    A = nx.to_scipy_sparse_matrix(graph)
    # print(np.sum(A.todense(), axis=0))
    normalize(A, norm='l1', axis=1, copy=False)
    I_n = sp.eye(graph.number_of_nodes())
    I_min_A = I_n - A
    u, s, vt = lg.svds(I_min_A, k=self._d + 1, which='SM')
    t2 = time()
    self._X = vt.T
    self._X = self._X[:, 1:]
    return self._X, (t2 - t1)
    # I_n = sp.eye(graph.number_of_nodes())

  @property
  def vectors(self):
    vectors = {}
    for i in range(self._node_num):
      vectors[str(i)] = self._X[i, :]
    return vectors

  # def get_embedding(self):
  #   return self._X

  # def get_edge_weight(self, i, j):
  #   return np.exp(
  #       -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
  #   )

  def save_embeddings(self, filename):
    fout = open(filename, 'w')
    node_num = len(self.vectors.keys())
    fout.write("{} {}\n".format(node_num, self._d))
    for node, vec in self.vectors.items():
        fout.write("{} {}\n".format(node,
                                    ' '.join([str(x) for x in vec])))
    fout.close()

  def getAdj(self):
    node_size = self._node_num
    look_up = self.g.look_up_dict
    adj = np.zeros((node_size, node_size))
    for edge in self.g.G.edges():
        adj[look_up[edge[0]]][look_up[edge[1]]] += 1.0
        adj[look_up[edge[1]]][look_up[edge[0]]] += 1.0
    # ScaleSimMat
    return adj/np.sum(adj, axis=1)