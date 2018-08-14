import numpy as np
import networkx as nx


class LaplacianEigenmaps(object):
    def __init__(self, graph, rep_size=128):
        self.g = graph
        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = rep_size
        self.adj_mat = nx.to_numpy_array(self.g.G)
        self.vectors = {}
        self.embeddings = self._get_embeddings()
        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
        return adj

    def getLap(self):
        degree_mat = np.diagflat(np.sum(self.adj_mat, axis=1))
        deg_trans = np.diagflat(np.reciprocal(np.sqrt(np.sum(self.adj_mat, axis=1))))
        L = degree_mat-self.adj_mat

        # eye = np.eye(self.node_size)
        norm_lap_mat = np.matmul(np.matmul(deg_trans, L), deg_trans)
        return norm_lap_mat

    def _get_embeddings(self):
        # lap_mat = nx.normalized_laplacian_matrix(self.g.G)
        # lap_mat = lap_mat.toarray()

        print(np.sum(self.adj_mat, axis=1))
        print(np.sum(self.adj_mat, axis=0))

        lap_mat = np.diagflat(np.sum(self.adj_mat, axis=1)) - self.adj_mat

        # lap_mat = self.getLap()
        w, vec = np.linalg.eigh(lap_mat)
        w = np.diagflat(np.sqrt(w[self.node_size-self.rep_size:]))
        vec = vec[:, self.node_size-self.rep_size:]

        # ret = vec[:, self.node_size-self.rep_size:]
        # ret = vec[:, :self.rep_size]
        # ret = vec[:self.rep_size, :].T
        # ret = vec[self.node_size-self.rep_size:, :].T
        return np.matmul(vec, w)

