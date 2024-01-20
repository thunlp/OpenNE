from __future__ import print_function
import numpy as np
import tensorflow as tf
import networkx as nx


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
        self.sess = tf.Session()
        self.adj_mat = self.getAdj()
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

    def get_train(self):

        adj_mat = self.adj_mat

        mat_mask = 1.*(adj_mat > 0)

        _embeddings = tf.Variable(tf.contrib.layers.xavier_initializer()([self.node_size, self.rep_size]),
                                  dtype=tf.float32, name='embeddings')

        Adj = tf.placeholder(tf.float32, [self.node_size, self.node_size], name='adj_mat')
        AdjMask = tf.placeholder(tf.float32, [self.node_size, self.node_size], name='adj_mask')

        cost = tf.reduce_sum(
                tf.square(Adj - tf.matmul(_embeddings, tf.transpose(_embeddings))*AdjMask)) + \
               self.lamb * tf.reduce_sum(tf.square(_embeddings))

        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        print("total iter: %i" % self.max_iter)
        for step in range(self.max_iter):
            self.sess.run(train_op, feed_dict={Adj: adj_mat, AdjMask: mat_mask})
            if step % 50 == 0:
                print("step %i: cost: %g" % (step, self.sess.run(cost, feed_dict={Adj: adj_mat, AdjMask: mat_mask})))
        return self.sess.run(_embeddings)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
