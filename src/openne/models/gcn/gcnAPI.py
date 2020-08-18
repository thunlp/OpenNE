import numpy as np
from .utils import *
from . import gcnModel
from ..models import *
import time
import scipy.sparse as sp
import torch


class GCN(ModelWithEmbeddings):

    def __init__(self, hiddens=None, max_degree=0, **kwargs):
        if hiddens is None:
            hiddens = [16]
        super(GCN, self).__init__(hiddens=hiddens, max_degree=max_degree, **kwargs)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {"lr": 0.01,
                                 "epochs": 200,
                                 "dropout": 0.5,
                                 "weight_decay": 1e-4,
                                 "early_stopping": 100,
                                 "clf_ratio": 0.1,
                                 "hiddens": [16],
                                 "max_degree": 0,
                                 "sparse": False})
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
            raise TypeError("GCN only accepts attributed graphs!")

    def build(self, graph, *, lr=0.01, epochs=200,
              dropout=0.5, weight_decay=1e-4, early_stopping=100,
              clf_ratio=0.1, sparse=False, **kwargs):
        """
                        lr: Initial learning rate
                        epochs: Number of epochs to train
                        hidden1: Number of units in hidden layer 1
                        dropout: Dropout rate (1 - keep probability)
                        weight_decay: Weight for L2 loss on embedding matrix
                        early_stopping: Tolerance for early stopping (# of epochs)
                        max_degree: Maximum Chebyshev polynomial degree
        """
        self.clf_ratio = clf_ratio
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.sparse = sparse

        self.preprocess_data(graph)
        # Create models
        input_dim = self.features.shape[1]  # row
        if self.sparse:
            feature_shape = self.features.values().shape[0]  # nnz
        else:
            feature_shape = self.features.shape  # [col, row]
        output_dim = self.labels.shape[1]
        self.model = gcnModel.GCNModel(input_dim=input_dim, output_dim=output_dim, hidden_dims=self.hiddens,
                                       supports=self.support, dropout=self.dropout, sparse_inputs=self.sparse,
                                       num_features_nonzero=feature_shape, weight_decay=self.weight_decay,
                                       logging=False)
        self.cost_val = []

    def train_model(self, graph, **kwargs):
        # Train models
        output, train_loss, train_acc, __ = self.evaluate(kwargs['train_mask'])
        self.debug_info = "train_loss = {:.5f}, train_acc = {:.5f}".format(train_loss, train_acc)
        return output

    def make_output(self, graph, **kwargs):
        self.embeddings = self.model(self.features).detach()


    def early_stopping_judge(self, graph, *, step=0, **kwargs):
        return kwargs['validate'] and step > self.early_stopping and self.cost_val[-1] > torch.mean(
                    torch.stack(self.cost_val[-(self.early_stopping + 1):-1]))

    # Define models evaluation function
    def evaluate(self, mask, train=True):
        mask = mask.to(self._device)
        torch.autograd.set_detect_anomaly(True)
        t_test = time.time()
        self.model.zero_grad()
        self.model.train(train)

        output = self.model(self.features)
        loss = self.model.loss(self.labels, mask)
        accuracy = self.model.accuracy(self.labels, mask)
        if train:
            loss.backward()
            # print([(name, param.grad) for name,param in self.model.named_parameters()])
            self.model.optimizer.step()
        return output, loss, accuracy, (time.time() - t_test)

    # todo: check if is standard operation for supervised node prediction
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
        self.register_float_buffer("labels", torch.zeros((len(labels), label_id)))
        self.label_dict = label_dict
        for node, l in labels:
            node_id = look_up[node]
            for ll in l:
                l_id = label_dict[ll]
                self.labels[node_id, l_id] = 1

    def preprocess_data(self, graph):
        """
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
            y_train, y_val, y_test can merge to y
        """
        g = graph.G
        look_back = graph.look_back_list
        features = torch.from_numpy(graph.features()).type(torch.float32)
        features = preprocess_features(features, sparse=self.sparse)
        self.register_buffer("features", features)
        self.build_label(graph)
        adj = graph.adjmat(weighted=True, directed=True)
        if self.max_degree == 0:
            self.support = [preprocess_adj(adj)]
        else:
            self.support = chebyshev_polynomials(adj, self.max_degree)
        self.support = [i.to(self._device) for i in self.support]
        for n, i in enumerate(self.support):
            self.register_buffer("support_{0}".format(n), i)

