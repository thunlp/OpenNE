import numpy as np
from .gcn.utils import *
from .gcn import gcnModel
from .gcn.layers import *
from .models import *
import time
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGAEModel(nn.Module):
    def __init__(self, dimensions, adj, dropout):
        super(VGAEModel, self).__init__()
        self.adj = adj
        self.gc1 = GraphConvolution(dimensions[0], dimensions[1], dropout, act=F.relu)
        self.gc2 = GraphConvolution(dimensions[1], dimensions[2], dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(dimensions[1], dimensions[2], dropout, act=lambda x: x)

    def encode(self, x):
        hidden1 = self.gc1(x, self.adj)
        return self.gc2(hidden1, self.adj), self.gc3(hidden1, self.adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VGAE(ModelWithEmbeddings):

    def __init__(self, output_dim=16, hiddens=None, max_degree=0, **kwargs):
        if hiddens is None:
            hiddens = [32]
        super(VGAE, self).__init__(output_dim=output_dim, hiddens=hiddens, max_degree=max_degree, **kwargs)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {"lr": 0.1,
                                 "epochs": 300,
                                 "dropout": 0.,
                                 "weight_decay": 1e-4,
                                 "early_stopping": 100,
                                 "clf_ratio": 0.8,
                                 "hiddens": [32],
                                 "max_degree": 0})
        check_range(kwargs, {"lr": (0, np.inf),
                             "epochs": (0, np.inf),
                             "dropout": (0, 1),
                             "weight_decay": (0, 1),
                             "early_stopping": (0, np.inf),
                             "clf_ratio": (0, 1),
                             "max_degree": (0, np.inf)})
        return kwargs

    @classmethod
    def check_graphtype(cls, graphtype, **kwargs):
        if not graphtype.attributed():
            raise TypeError("GAE only accepts attributed graphs!")

    def build(self, graph, *, lr=0.01, epochs=200,
              dropout=0.5, weight_decay=1e-4, early_stopping=100,
              clf_ratio=0.1, **kwargs):
        """
                        lr: Initial learning rate
                        epochs: Number of epochs to train
                        hidden1: Number of units in hidden layer 1
                        dropout: Dropout rate (1 - keep probability)
                        weight_decay: Weight for L2 loss on embedding matrix
                        early_stopping: Tolerance for early stopping (# of epochs)
                        max_degree: Maximum Chebyshev polynomial degree
        """
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.sparse = False
        self.preprocess_data(graph)
        # Create models
        input_dim = self.features.shape[1] if not self.sparse else self.features[2][1]
        feature_shape = self.features.shape if not self.sparse else self.features[0].shape[0]

        self.dimensions = [input_dim] + self.hiddens + [self.output_dim]
        self.model = VGAEModel(self.dimensions, self.support[0], self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_model(self, graph, **kwargs):
        # Train models
        if 'train_mask' not in self.__dict__:
            self.train_mask = torch.ones(graph.nodesize, self._device)
            print("Train_mask")
        output, train_loss, __ = self.evaluate(self.train_mask)
        self.debug_info = "train_loss = {:.5f}".format(train_loss)
        return output
        
    def build_label(self, graph):
        g = graph.G
        look_up = graph.look_up_dict
        labels = []
        label_dict = {}
        label_id = 0
        for node in g.nodes():
            labels.append((node, g.nodes[node]['label']))
            for l in g.nodes[node]['label']:
                if l not in label_dict:
                    label_dict[l] = label_id
                    label_id += 1
        self.labels = torch.zeros((len(labels), label_id), device=self._device)
        self.label_dict = label_dict
        for node, l in labels:
            node_id = look_up[node]
            for ll in l:
                l_id = label_dict[ll]
                self.labels[node_id][l_id] = 1
    
    def loss(self, output, adj_label, pos_weight, norm, mu, logvar, n_nodes):
        cost = 0.

        cost += norm * F.binary_cross_entropy_with_logits(torch.mm(output, output.t()), adj_label, pos_weight=pos_weight)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD


    # Define models evaluation function
    def evaluate(self, mask, train=True):
        #torch.autograd.set_detect_anomaly(True)
        t_test = time.time()
        self.optimizer.zero_grad()
        self.model.train(train)
        output, mu, var= self.model(self.features)

        loss = self.loss(output, self.adj_label, self.pos_weight, self.norm, mu, var, self.n_nodes)
        if train:
            loss.backward()
            #print([(name, param.grad) for name,param in self.model.named_parameters()])
            self.optimizer.step()
        return output, loss, (time.time() - t_test)

    def make_output(self, graph, **kwargs):
        output, mu, var= self.model(self.features)
        self.embeddings = mu.detach()

    def preprocess_data(self, graph):
        """
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
            y_train, y_val, y_test can merge to y
        """
        g = graph.G
        look_back = graph.look_back_list
        self.features = torch.from_numpy(graph.features()).type(torch.float32)
        self.features = preprocess_features(self.features, sparse=self.sparse)
        self.features = self.features.to(self._device)
        n = graph.nodesize
        self.n_nodes = n
        self.build_label(graph)
        adj_label = graph.adjmat(weighted=False, directed=False, sparse=True)

        self.adj_label = torch.tensor((adj_label + sp.eye(n).toarray()), dtype=torch.float32, device=self._device)
        adj = nx.adjacency_matrix(g)  # the type of graph
        self.pos_weight = torch.tensor([float(n * n - adj.sum()) / adj.sum()], dtype=torch.float32, device=self._device)
        self.norm = n * n / float((n * n - adj.sum()) * 2)

        if self.max_degree == 0:
            self.support = [preprocess_graph(adj)]
        else:
            self.support = chebyshev_polynomials(adj, self.max_degree)
        self.support = [i.to(self._device) for i in self.support]
        # for n, i in enumerate(self.support):
        #    self.register_buffer("support_{0}".format(n), i)
        # print(self.support)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'