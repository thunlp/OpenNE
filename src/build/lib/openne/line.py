from __future__ import print_function
import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from .classify import Classifier, read_node_label
import torch
import torch.nn.functional as F


class _LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, negative_ratio=5, order=3, lr=0.001):
        self.cur_epoch = 0
        self.order = order
        self.g = graph
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.lr = lr
        self.table_size=1e8
        self.gen_sampling_table()
        cur_seed = random.getrandbits(32)
        self.build_graph()

    def build_graph(self):
        cur_seed = random.getrandbits(32)
        torch.manual_seed(cur_seed)
        self.embeddings = torch.nn.init.xavier_normal_(
            torch.FloatTensor(self.node_size, self.rep_size).requires_grad_(True))
        self.context_embeddings = self.embeddings.clone().detach().requires_grad_(True)
        self.second_loss=lambda s, h, t: -(F.logsigmoid(
            s*(self.embeddings[h]*self.context_embeddings[t]).sum(dim=1))).mean()
        self.first_loss=lambda s, h, t: -(F.logsigmoid(
            s*(self.embeddings[h]*self.embeddings[t]).sum(dim=1))).mean()
        if self.order == 1:
            self.loss = self.first_loss
        else:
            self.loss = self.second_loss
        self.optimizer = torch.optim.Adam([self.embeddings, self.context_embeddings], lr=self.lr)

    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            self.optimizer.zero_grad()
            cur_loss = self.loss(torch.tensor(sign),h,t)
            sum_loss += cur_loss
            cur_loss.backward()
            self.optimizer.step()
            batch_id += 1
        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1

    def batch_iter(self):
        look_up = self.g.look_up_dict

        table_size = self.table_size
        numNodes = self.node_size

        edges = [(look_up[x[0]], look_up[x[1]]) for x in self.g.G.edges()]

        data_size = self.g.G.number_of_edges()
        shuffle_indices = torch.randperm(data_size).tolist()
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            if mod == 0:
                sign = 1.
                h = []
                t = []
                for i in range(start_index, end_index):
                    if not random.random() < self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(h)):
                    t.append(
                        self.sampling_table[random.randint(0, table_size-1)])

            yield h, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index+self.batch_size, data_size)

    def gen_sampling_table(self):
        table_size = int(self.table_size)
        power = 0.75
        numNodes = self.node_size

        print("Pre-processing for non-uniform negative sampling!")
        node_degree = torch.zeros(numNodes)  # out degree
        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = float((node_degree**power).sum())  # float is faster than tensor when visited
        node_degree=node_degree.tolist() # list has fastest visit speed
        self.sampling_table = np.zeros(table_size, dtype=np.int32) # torch is much slower when referring to frequent visits
        p = 0
        i = 0
        for j in range(numNodes):
            p += math.pow(node_degree[j], power) / norm
            while i < table_size and i / table_size < p:
                self.sampling_table[i] = j
                i += 1
        # self.sampling_table=torch.from_numpy(self.sampling_table)
        data_size = self.g.G.number_of_edges()
        self.edge_alias = [0 for i in range(data_size)]
        self.edge_prob = [0 for i in range(data_size)]
        large_block = [0 for i in range(data_size)]
        small_block = [0 for i in range(data_size)]
        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"]
                         for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"] *
                     data_size/total_sum for edge in self.g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + \
                norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1
        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1

    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.detach()
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors


class LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, epoch=10, negative_ratio=5, order=3, label_file=None, clf_ratio=0.5, auto_save=True, lr=0.001):
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        if order == 3:
            self.model1 = _LINE(graph, rep_size//2, batch_size,
                                negative_ratio, order=1, lr=lr)
            self.model2 = _LINE(graph, rep_size//2, batch_size,
                                negative_ratio, order=2, lr=lr)
            for i in range(epoch):
                self.model1.train_one_epoch()
                self.model2.train_one_epoch()
                if label_file:
                    self.get_embeddings()
                    X, Y = read_node_label(label_file)
                    print("Training classifier using {:.2f}% nodes...".format(
                        clf_ratio*100))
                    clf = Classifier(vectors=self.vectors,
                                     clf=LogisticRegression())
                    result = clf.split_train_evaluate(X, Y, clf_ratio)

                    if result['macro'] > self.best_result:
                        self.best_result = result['macro']
                        if auto_save:
                            self.best_vector = self.vectors

        else:
            self.model = _LINE(graph, rep_size, batch_size,
                               negative_ratio, order=self.order, lr=lr)
            for i in range(epoch):
                self.model.train_one_epoch()
                if label_file:
                    self.get_embeddings()
                    X, Y = read_node_label(label_file)
                    print("Training classifier using {:.2f}% nodes...".format(
                        clf_ratio*100))
                    clf = Classifier(vectors=self.vectors,
                                     clf=LogisticRegression())
                    result = clf.split_train_evaluate(X, Y, clf_ratio)

                    if result['macro'] > self.best_result:
                        self.best_result = result['macro']
                        if auto_save:
                            self.best_vector = self.vectors

        self.get_embeddings()
        if auto_save and label_file:
            self.vectors = self.best_vector

    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = {}
        if self.order == 3:
            vectors1 = self.model1.get_embeddings()
            vectors2 = self.model2.get_embeddings()
            for node in vectors1.keys():
                self.vectors[node] = np.append(vectors1[node], vectors2[node])
        else:
            self.vectors = self.model.get_embeddings()

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(float(x)) for x in vec])))
        fout.close()
