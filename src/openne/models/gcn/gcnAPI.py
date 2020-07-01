import numpy as np
from .utils import *
from . import models
from ..models import *
import time
import scipy.sparse as sp
import torch


class GCN(ModelWithEmbeddings):

    def __init__(self, hiddens=None, max_degree=0):
        if hiddens is None:
            hiddens = [16]
        super(GCN, self).__init__(hiddens=hiddens, max_degree=max_degree)

    @classmethod
    def check_train_parameters(cls, graphtype, **kwargs):
        if not graphtype.attributed():
            raise TypeError("GCN only accepts attributed graphs!")
        check_existance(kwargs, {"learning_rate": 0.01,
                                 "epochs": 200,
                                 "dropout": 0.5,
                                 "weight_decay": 1e-4,
                                 "early_stopping": 100,
                                 "clf_ratio": 0.1})
        check_range(kwargs, {"learning_rate": (0, np.inf),
                             "epochs": (0, np.inf),
                             "dropout": (0, 1),
                             "weight_decay": (0, 1),
                             "early_stopping": (0, np.inf),
                             "clf_ratio": (0, 1)})

    def build(self, graph, *, learning_rate=0.01, epochs=200,
              dropout=0.5, weight_decay=1e-4, early_stopping=100,
              clf_ratio=0.1, **kwargs):
        """
                        learning_rate: Initial learning rate
                        epochs: Number of epochs to train
                        hidden1: Number of units in hidden layer 1
                        dropout: Dropout rate (1 - keep probability)
                        weight_decay: Weight for L2 loss on embedding matrix
                        early_stopping: Tolerance for early stopping (# of epochs)
                        max_degree: Maximum Chebyshev polynomial degree
        """
        self.clf_ratio = clf_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.sparse = False

        self.preprocess_data(graph)
        # Create models
        input_dim = self.features.shape[1] if not self.sparse else self.features[2][1]
        feature_shape = self.features.shape if not self.sparse else self.features[0].shape[0]
        output_dim = self.labels.shape[1]
        self.model = models.GCNModel(input_dim=input_dim, output_dim=output_dim, hidden_dims=self.hiddens,
                                supports=self.support, dropout=self.dropout, sparse_inputs=self.sparse,
                                num_features_nonzero=feature_shape, weight_decay=self.weight_decay, logging=False)
        self.cost_val = []

    def get_train(self, graph, **kwargs):
        # Train models
        output, train_loss, train_acc, __ = self.evaluate(kwargs['train_mask'])
        self.debug_info = {"train_loss": "{:.5f}".format(train_loss),
                           "train_acc=": "{:.5f}".format(train_acc)}
        return output

    def early_stopping_judge(self, graph, *, step=0, **kwargs):
        return step > self.early_stopping and self.cost_val[-1] > torch.mean(
                    torch.stack(self.cost_val[-(self.early_stopping + 1):-1]))

    # Define models evaluation function
    def evaluate(self, mask, train=True):
        torch.autograd.set_detect_anomaly(True)
        t_test = time.time()
        self.model.zero_grad()
        self.model.train(train)
        output = self.model(self.features)
        loss = self.model.loss(self.labels, mask)
        accuracy = self.model.accuracy(self.labels, mask)
        if train:
            loss.backward()
            # print([(name, param.grad) for name,param in self.models.named_parameters()])
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
        self.labels = torch.zeros((len(labels), label_id))
        self.label_dict = label_dict
        for node, l in labels:
            node_id = look_up[node]
            for ll in l:
                l_id = label_dict[ll]
                self.labels[node_id][l_id] = 1


        # # Testing
        # test_res, test_cost, test_acc, test_duration = self.evaluate(self.test_mask, False)
        # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    def build_train_val_test(self, graph):
        """
            build train_mask test_mask val_mask
        """
        train_precent = self.clf_ratio
        training_size = int(train_precent * graph.G.number_of_nodes())
        state = torch.random.get_rng_state()
        torch.random.manual_seed(0)
        shuffle_indices = torch.randperm(graph.G.number_of_nodes())
        torch.random.set_rng_state(state)
        g = graph.G

        def sample_mask(begin, end):
            mask = torch.zeros(g.number_of_nodes())
            for i in range(begin, end):
                mask[shuffle_indices[i]] = 1
            return mask

        self.train_mask = sample_mask(0, training_size - 100)
        self.val_mask = sample_mask(training_size - 100, training_size)
        self.test_mask = sample_mask(training_size, g.number_of_nodes())

    def preprocess_data(self, graph):
        """
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
            y_train, y_val, y_test can merge to y
        """
        g = graph.G
        look_back = graph.look_back_list
        self.features = torch.stack([g.nodes[look_back[i]]['feature']
                                     for i in range(g.number_of_nodes())])
        self.features = preprocess_features(self.features, sparse=self.sparse)
        self.build_label(graph)
        self.build_train_val_test(graph)
        adj = nx.adjacency_matrix(g)  # the type of graph
        if self.max_degree == 0:
            self.support = [preprocess_adj(adj)]
        else:
            self.support = chebyshev_polynomials(adj, self.max_degree)
        # print(self.support)
