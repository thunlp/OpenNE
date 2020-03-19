from __future__ import print_function
import torch


__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


class GraphFactorization(object):
    def __init__(self, graph, rep_size=128, epoch=120, learning_rate=0.003, weight_decay=1.):
        self.g = graph

        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.max_iter = epoch
        self.lr = learning_rate
        self.lamb = weight_decay
        self.adj_mat = self.getAdj()
        self.vectors = {}

        self.embeddings = self.get_train()

        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = torch.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = self.g.G[edge[0]][edge[1]]['weight']
        return adj

    def get_train(self):


        adj_mat = self.adj_mat
        mat_mask = torch.as_tensor(adj_mat > 0, dtype=torch.float32)

        _embeddings = torch.tensor(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.node_size, self.rep_size)),
                               requires_grad=True)
        print(_embeddings)

        Adj = adj_mat
        AdjMask = mat_mask

        optimizer = torch.optim.Adam([_embeddings], lr=self.lr)

        print("total iter: %i" % self.max_iter)
        for step in range(self.max_iter):
            optimizer.zero_grad()
            cost = ((Adj - torch.mm(_embeddings, _embeddings.t()) * AdjMask) ** 2).sum() + \
                self.lamb * ((_embeddings ** 2).sum())
            cost.backward()
            optimizer.step()
            if step % 5 == 0:
                print("step %i: cost: %g" % (step, cost))
        return _embeddings.detach()

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(float(x)) for x in vec])))
        fout.close()
