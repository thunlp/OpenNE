import numpy as np
import networkx as nx


__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


class LaplacianEigenmaps(object):
    def __init__(self, graph, rep_size=128):
        self.g = graph
        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = rep_size
        self.adj_mat = nx.to_numpy_array(self.g.G)
        self.vectors = {}
        self.embeddings = self.get_train()
        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = self.g.G[edge[0]][edge[1]]['weight']
        return adj

    def getLap(self):
        degree_mat = np.diagflat(np.sum(self.adj_mat, axis=1))
        deg_trans = np.diagflat(np.reciprocal(np.sqrt(np.sum(self.adj_mat, axis=1))))
        deg_trans = np.nan_to_num(deg_trans)
        L = degree_mat-self.adj_mat

        # eye = np.eye(self.node_size)
        norm_lap_mat = np.matmul(np.matmul(deg_trans, L), deg_trans)
        return norm_lap_mat

    def get_train(self):
        lap_mat = self.getLap()
        w, vec = np.linalg.eigh(lap_mat)

        start = 0
        for i in range(self.node_size):
            if w[i] > 1e-10:
                start = i
                break
        vec = vec[:, start:start+self.rep_size]

        return vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

