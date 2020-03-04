import tensorflow as tf
import numpy as np


__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer()([n_in, n_out]), dtype=tf.float32, name=scope + "w")

        # kernel = tf.Variable(tf.random_normal([n_in, n_out]))
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation


class SDNE(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4,
                 batch_size=200, epoch=100, learning_rate=None):
        """
        encoder_layer_list: a list of numbers of the neuron at each ecdoer layer, the last number is the
        dimension of the output node representation
        Eg:
        if node size is 2000, encoder_layer_list=[1000, 128], then the whole neural network would be
        2000(input)->1000->128->1000->2000, SDNE extract the middle layer as the node representation
        """
        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.dim = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(encoder_layer_list)
        self.encoder_layer_num = len(encoder_layer_list)+1

        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.bs = batch_size
        self.epoch = epoch
        self.max_iter = (epoch * self.node_size) // batch_size

        self.lr = learning_rate
        if self.lr is None:
            self.lr = tf.train.inverse_time_decay(0.03, self.max_iter, decay_steps=1, decay_rate=0.9999)

        self.sess = tf.Session()
        self.vectors = {}

        self.adj_mat = self.getAdj()
        self.embeddings = self.train()

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

    def train(self):
        adj_mat = self.adj_mat

        AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
        Adj = tf.placeholder(tf.float32, [None, None], name='adj_mat')
        B = tf.placeholder(tf.float32, [None, self.node_size], name='b_mat')

        fc = AdjBatch
        scope_name = 'encoder'
        layer_collector = []

        with tf.name_scope(scope_name):
            for i in range(1, self.encoder_layer_num):
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        scope_name = 'decoder'
        with tf.name_scope(scope_name):
            for i in range(self.encoder_layer_num-2, 0, -1):
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)
            fc = fc_op(fc,
                       name=scope_name+str(0),
                       n_out=self.encoder_layer_list[0],
                       layer_collector=layer_collector,)

        _embeddings_norm = tf.reduce_sum(tf.square(_embeddings), 1, keepdims=True)

        L_1st = tf.reduce_sum(
            Adj * (
                    _embeddings_norm - 2 * tf.matmul(
                        _embeddings, tf.transpose(_embeddings)
                    ) + tf.transpose(_embeddings_norm)
            )
        )

        L_2nd = tf.reduce_sum(tf.square((AdjBatch - fc) * B))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.nu1 * tf.reduce_sum(tf.abs(param[0])) + self.nu2 * tf.reduce_sum(tf.square(param[0]))

        optimizer = tf.train.AdamOptimizer(self.lr)

        train_op = optimizer.minimize(L)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        print("total iter: %i" % self.max_iter)
        for step in range(self.max_iter):
            index = np.random.randint(self.node_size, size=self.bs)
            adj_batch_train = adj_mat[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = np.ones_like(adj_batch_train)
            b_mat_train[adj_batch_train != 0] = self.beta

            self.sess.run(train_op, feed_dict={AdjBatch: adj_batch_train,
                                               Adj: adj_mat_train,
                                               B: b_mat_train})
            if step % 50 == 0:
                l, l1, l2 = self.sess.run((L, L_1st, L_2nd),
                                          feed_dict={AdjBatch: adj_batch_train,
                                                     Adj: adj_mat_train,
                                                     B: b_mat_train})
                print("step %i: total loss: %s, l1 loss: %s, l2 loss: %s" % (step, l, l1, l2))

        return self.sess.run(_embeddings, feed_dict={AdjBatch: adj_mat})

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()


class SDNE2(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-5,
                 batch_size=100, max_iter=2000, learning_rate=None):

        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size] + encoder_layer_list
        self.encoder_layer_num = len(encoder_layer_list)+1

        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.bs = batch_size
        self.max_iter = max_iter
        self.lr = learning_rate
        if self.lr is None:
            self.lr = tf.train.inverse_time_decay(0.1, self.max_iter, decay_steps=1, decay_rate=0.9999)

        self.sess = tf.Session()
        self.vectors = {}

        self.adj_mat = self.getAdj()
        self.deg_vec = np.sum(self.adj_mat, axis=1)
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

    def model(self, node, layer_collector, scope_name):
        fc = node
        with tf.name_scope(scope_name + 'encoder'):
            for i in range(1, self.encoder_layer_num):
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        with tf.name_scope(scope_name + 'decoder'):
            for i in range(self.encoder_layer_num-2, -1, -1):
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        return _embeddings, fc

    def generate_batch(self, shuffle=True):
        adj = self.adj_mat

        row_indices, col_indices = adj.nonzero()
        sample_index = np.arange(row_indices.shape[0])
        num_of_batches = row_indices.shape[0] // self.bs
        counter = 0
        if shuffle:
            np.random.shuffle(sample_index)

        while True:
            batch_index = sample_index[self.bs * counter:self.bs * (counter + 1)]

            nodes_a = adj[row_indices[batch_index], :]
            nodes_b = adj[col_indices[batch_index], :]
            weights = adj[row_indices[batch_index], col_indices[batch_index]]
            weights = np.reshape(weights, [-1, 1])

            beta_mask_a = np.ones_like(nodes_a)
            beta_mask_a[nodes_a != 0] = self.beta
            beta_mask_b = np.ones_like(nodes_b)
            beta_mask_b[nodes_b != 0] = self.beta

            if counter == num_of_batches:
                counter = 0
                np.random.shuffle(sample_index)
            else:
                counter += 1

            yield (nodes_a, nodes_b, beta_mask_a, beta_mask_b, weights)

    def get_train(self):

        NodeA = tf.placeholder(tf.float32, [None, self.node_size], name='node_a')
        BmaskA = tf.placeholder(tf.float32, [None, self.node_size], name='beta_mask_a')
        NodeB = tf.placeholder(tf.float32, [None, self.node_size], name='node_b')
        BmaskB = tf.placeholder(tf.float32, [None, self.node_size], name='beta_mask_b')
        Weights = tf.placeholder(tf.float32, [None, 1], name='adj_weights')

        layer_collector = []
        nodes = tf.concat([NodeA, NodeB], axis=0)
        bmasks = tf.concat([BmaskA, BmaskB], axis=0)
        emb, recons = self.model(nodes, layer_collector, 'reconstructor')
        embs = tf.split(emb, num_or_size_splits=2, axis=0)

        L_1st = tf.reduce_sum(Weights * (tf.reduce_sum(tf.square(embs[0] - embs[1]), axis=1)))

        L_2nd = tf.reduce_sum(tf.square((nodes - recons) * bmasks))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.nu1 * tf.reduce_sum(tf.abs(param[0])) + self.nu2 * tf.reduce_sum(tf.square(param[0]))

        # lr = tf.train.exponential_decay(1e-6, self.max_iter, decay_steps=1, decay_rate=0.9999)
        # optimizer = tf.train.MomentumOptimizer(lr, 0.99, use_nesterov=True)

        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(L)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        generator = self.generate_batch()

        for step in range(self.max_iter+1):
            nodes_a, nodes_b, beta_mask_a, beta_mask_b, weights = generator.__next__()

            feed_dict = {NodeA: nodes_a,
                         NodeB: nodes_b,
                         BmaskA: beta_mask_a,
                         BmaskB: beta_mask_b,
                         Weights: weights}

            self.sess.run(train_op, feed_dict=feed_dict)
            if step % 50 == 0:
                print("step %i: %s" % (step, self.sess.run([L, L_1st, L_2nd], feed_dict=feed_dict)))

        return self.sess.run(emb, feed_dict={NodeA: self.adj_mat[0:1, :], NodeB: self.adj_mat[1:, :]})

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()