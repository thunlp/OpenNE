import numpy as np
from .models import *
import torch
from torch.utils.data import DataLoader
from overloading import overload

__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def adjust_lr(optimizer, step, decay_strategy=lambda x: 0.03):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay_strategy(step)


# reference: https://github.com/mehulrastogi/Deep-Belief-Network-pytorch
# Pretrain mentioned in original paper; no significant improvement as compared with Xavier-init.
# Set pretrain=False to disable.


class RBM(torch.nn.Module):
    def __init__(self, weights=None, bias=None, visible_dim=128, hidden_dim=64, lr=1e-5, decay=False, batch_size=16):
        super(RBM, self).__init__()
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        if weights == None:
            self.weights = torch.nn.init.xavier_uniform_(torch.FloatTensor(visible_dim, hidden_dim))
            self.v_dim = visible_dim
            self.h_dim = hidden_dim
            self.v_bias = torch.zeros(visible_dim)
            self.h_bias = torch.zeros(hidden_dim)
        else:
            self.weights = weights.t()
            self.v_dim = len(self.weights)
            self.h_dim = len(bias)
            self.v_bias = torch.zeros(self.v_dim)  # the only tensor created
            self.h_bias = bias

    def sample(self, P):
        return torch.floor(torch.rand(P.size()) + P)  # torch.distributions.bernoulli.Bernoulli(P).sample()

    def forward(self, x):
        P = torch.sigmoid(self.h_bias + torch.mm(x, self.weights))
        return P

    def backward(self, x):
        P = torch.sigmoid(self.v_bias + torch.mm(x, self.weights.t()))
        return P

    def contrastive_divergence(self, x, update=True):
        h1 = self.sample(self.forward(x))
        v2 = self.sample(self.backward(h1))
        h2 = self.sample(self.forward(v2))

        dw = torch.mm(x.t(), h1) - torch.mm(v2.t(), h2)  # delta weight
        dvb = torch.sum(v2 - x, dim=0)  # delta v_bias
        dhb = torch.sum(h1 - h2, dim=0)  # delta h_bias
        lr = self.lr / self.batch_size
        if update:
            self.weights += lr * dw
            self.v_bias += lr * dvb
            self.h_bias += lr * dhb
        return torch.sum((x - v2) ** 2)  # , torch.mean(torch.sum((x-v2)**2, dim=0))

    def train_model(self, data, epochs=10, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        dataloader = DataLoader(data, batch_size=self.batch_size)
        lr0 = self.lr
        for epoch in range(epochs):
            allcost = 0.
            if self.decay:
                if epoch == 1:
                    self.lr /= 5
                elif epoch == 2:
                    self.lr /= 2
                elif epoch > 2:
                    self.lr = 0.0005
            for batch in dataloader:
                batch = batch.view(len(batch), self.v_dim)
                a = (self.contrastive_divergence(batch))
                allcost += a
            print("Epoch:{}, lr={},cost={}".format(epoch, self.lr, allcost))
        self.lr = lr0


class SDNENet(torch.nn.Module):
    def __init__(self, encoder_layer_list, alpha, nu1, nu2, pretrain_lr=1e-5, pretrain_epoch=1):
        super(SDNENet, self).__init__()
        self.alpha = alpha
        self.nu1 = nu1
        self.nu2 = nu2  # loss parameters
        self.pretrain_lr = pretrain_lr
        self.pretrain_epoch = pretrain_epoch
        layer_collector = []
        for i in range(1, len(encoder_layer_list)):
            layer_collector.append(torch.nn.Linear(encoder_layer_list[i - 1], encoder_layer_list[i]))
            layer_collector.append(torch.nn.LeakyReLU())
            # layer_collector.append(torch.nn.Sigmoid()) as written in paper
        self.encoder = torch.nn.Sequential(*layer_collector)

        layer_collector1 = []
        for i in range(len(encoder_layer_list) - 2, -1, -1):
            layer_collector1.append(torch.nn.Linear(encoder_layer_list[i + 1], encoder_layer_list[i]))
            layer_collector1.append(torch.nn.LeakyReLU())

        self.decoder = torch.nn.Sequential(*layer_collector1)
        self.layer_collector = layer_collector + layer_collector1

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def pretrain(self, data):  # deep-belief-network-based pretraining
        for layer in self.layer_collector:
            if type(layer) == torch.nn.Linear:
                rbm = RBM(visible_dim=len(layer.weight[0]), hidden_dim=len(layer.weight), batch_size=len(data),
                          lr=self.pretrain_lr * len(data),
                          decay=False)
                rbm.train_model(data, epochs=self.pretrain_epoch)
                layer.weight = torch.nn.Parameter(rbm.weights.t())
                layer.bias = torch.nn.Parameter(rbm.h_bias)
                data = rbm.sample(rbm.forward(data))

    def forward(self, a_b):
        embeddings = self.encoder(a_b)
        final = self.decoder(embeddings)
        return embeddings, final

    @staticmethod
    def _L_1st(a, e_n, e):
        return (a * (e_n - 2 * torch.mm(e, e.t()) + e_n.t())).sum()

    @staticmethod
    def _L_2nd(a_b, f, b):
        x1 = (((a_b - f) * b) ** 2).sum()
        return x1

    def loss(self, a_b, a, b, embeddings, final):
        embeddings_norm = (embeddings ** 2).sum(1, keepdims=True)
        l1 = self._L_1st(a, embeddings_norm, embeddings)
        l2 = self._L_2nd(a_b, final, b)
        L = l2 + self.alpha * l1
        for param in self.layer_collector:
            if type(param) == torch.nn.Linear:
                L += self.nu2 * (param.weight ** 2).sum() + self.nu1 * param.weight.abs().sum()
        return L, l1, l2


class SDNE(ModelWithEmbeddings):
    def __init__(self, encoder_layer_list, alpha=1e-6, beta=5., nu1=1e-8, nu2=1e-4, **kwargs):
        """
        encoder_layer_list: a list of numbers of the neuron at each encoder layer, the last number is the
        dimension of the output node representation
        Eg:
        if node size is 2000, encoder_layer_list=[1000, 128], then the whole neural network would be
        2000(input)->1000->128->1000->2000, SDNE extract the middle layer as the node representation
        """
        super(SDNE, self).__init__(encoder_layer_list=encoder_layer_list, alpha=alpha, beta=beta,
                                   nu1=nu1, nu2=nu2, **kwargs)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'encoder_layer_list': [128],
                                 'alpha': 1e-6,
                                 'beta': 5.,
                                 'nu1': 1e-8,
                                 'nu2': 1e-4,
                                 'batch_size': 200,
                                 'epochs': 100,
                                 'lr': 0.001,
                                 'decay': False,
                                 'pretrain': False})
        check_range(kwargs, {'batch_size': (0, np.inf),
                             'epochs': (0, np.inf),
                             'lr': (0, np.inf),
                             'decay': [True, False, 0, 1],
                             'pretrain': [True, False, 0, 1]})
        return kwargs

    def build(self, graph, *, batch_size=200, epochs=100, lr=0.001, decay=False, pretrain=False, **kwargs):
        self.node_size = graph.nodesize
        self.dim = self.encoder_layer_list[-1]
        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(self.encoder_layer_list)
        self.encoder_layer_num = len(self.encoder_layer_list) + 1
        self.bs = batch_size
        self.epochs = (epochs * self.node_size) // batch_size
        self.real_epochs = epochs
        self.decay = decay
        self.pretrain = pretrain
        if self.decay:
            self.lr = lambda x: lr / (1 + 0.9999 * x)
        else:
            self.lr = lambda x: lr
        self.adj_mat = torch.from_numpy(graph.adjmat(weighted=True, directed=True)).type(torch.float32)

        self.model = SDNENet(self.encoder_layer_list, self.alpha, self.nu1, self.nu2)
        if self.pretrain:
            self.model.pretrain(self.adj_mat)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr(0))

    def train_model(self, graph, *, step=0, **kwargs):
        index = torch.randint(high=self.node_size,
                              size=[self.bs])
        adj_batch_train = self.adj_mat[index, :]
        adj_mat_train = adj_batch_train[:, index]
        b_mat_train = torch.ones_like(adj_batch_train)
        b_mat_train[adj_batch_train != 0] = self.beta

        self.optimizer.zero_grad()
        embeddings, final = self.model(adj_batch_train)
        loss, l1, l2 = self.model.loss(adj_batch_train, adj_mat_train, b_mat_train, embeddings, final)
        loss.backward()
        self.optimizer.step()
        if self.decay:
            adjust_lr(self.optimizer, step, decay_strategy=self.lr)
        self.debug_info = "total loss: {}, l1 loss: {}, l2 loss: {}".format(loss, l1, l2)

    def make_output(self, graph, **kwargs):
        embeddings, _ = self.model(self.adj_mat)
        self.embeddings = embeddings.detach()
