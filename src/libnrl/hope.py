# pylint: disable=e1101
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from . import graph as g
import tensorflow as tf
from sklearn.preprocessing import normalize


class HOPE(object):

  # def __init__(self, *hyper_dict, **kwargs):
  #     ''' Initialize the HOPE class

  #     Args:
  #         d: dimension of the embedding
  #         beta: higher order coefficient
  #     '''
  #     hyper_params = {  
  #         'method_name': 'hope_gsvd'
  #     }
  def __init__(self, graph: g.Graph, d: int, beta=0.1):
    self._d = d
    self._beta = beta
    self._graph = graph.G
    self.g = graph
    self._node_num = graph.node_size
    self.learn_embedding()

  def learn_embedding(self):
    # A = nx.to_scipy_sparse_matrix(graph)
    # I = sp.eye(graph.number_of_nodes())
    # M_g = I - self._beta*A
    # M_l = self._beta*A
    graph = self.g.G
    A = nx.to_numpy_matrix(graph)
    ## Katz
    # M_g = np.eye(graph.number_of_nodes()) - self._beta * A
    # M_l = self._beta * A

    M_g = np.eye(graph.number_of_nodes())
    M_l = np.dot(A, A)

    ## AA
    # d = np.mean(A + A.T, axis=0)
    # D = np.diagflat(d)
    # M_g = np.eye(graph.number_of_nodes())
    # M_l = np.dot(A, np.dot(D, A))

    S = np.dot(np.linalg.inv(M_g), M_l)
    # s: \sigma_k
    u, s, vt = lg.svds(S, k=self._d // 2)
    sigma = np.diagflat(np.sqrt(s))
    X1 = np.dot(u, sigma)
    X2 = np.dot(vt.T, sigma)
    # self._X = X2
    self._X = np.concatenate((X1, X2), axis=1)
    # p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
    # eig_err = np.linalg.norm(p_d_p_t - S)
    # print('SVD error (low rank): %f' % eig_err)

  @property
  def vectors(self):
    vectors = {}
    for i in range(self._node_num):
      vectors[str(i)] = self._X[i, :]
    return vectors

  def save_embeddings(self, filename):
    fout = open(filename, 'w')
    node_num = len(self.vectors.keys())
    fout.write("{} {}\n".format(node_num, self._d))
    for node, vec in self.vectors.items():
      fout.write("{} {}\n".format(node,
                                  ' '.join([str(x) for x in vec])))
    fout.close()

  # def get_edge_weight(self, i, j):
  #   return np.dot(self._X[i, :self._d // 2], self._X[j, self._d // 2:])

  def getAdj(self):
    # graph = self.g.G
    node_size = self.g.node_size
    look_up = self.g.look_up_dict
    adj = np.zeros((node_size, node_size))
    for edge in self.g.G.edges():
      adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
      adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
    # ScaleSimMat
    return adj/np.sum(adj, axis=1)
