import tensorflow as tf
import numpy as np
import torch
from overloading import overload

__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"


def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1].value

    # return
    with tf.name_scope(name) as scope:
        # kernel = torch.nn.init.xavier_uniform_(torch.FloatTensor(n_in, n_out).requires_grad_(True))
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer()([n_in, n_out]), dtype=tf.float32, name=scope + "w")
        # biases = torch.zeros((1, n_out), dtype=torch.float32, requires_grad=True)
        # kernel = tf.Variable(tf.random_normal([n_in, n_out]))
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')
        # fc = torch.mm(input_op, kernel)+biases
        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def adjust_learning_rate(optimizer, step, decay_strategy=lambda x: 0.03):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay_strategy(step)

# reference: https://github.com/mehulrastogi/Deep-Belief-Network-pytorch
import torch
class RBM(torch.nn.Module):
    def __init__(self, weights=None, bias=None, visible_dim = 128, hidden_dim = 64, lr=1e-5, batch_size=16):
        super(RBM, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        if weights==None:
            self.weights = torch.nn.init.xavier_uniform_(torch.FloatTensor(visible_dim, hidden_dim))
            self.v_dim = visible_dim
            self.h_dim = hidden_dim
            self.v_bias = torch.zeros(visible_dim)
            self.h_bias = torch.zeros(hidden_dim)
        else:
            self.weights = weights
            self.v_dim = len(weights)
            self.h_dim = len(bias)
            self.v_bias = torch.zeros(self.v_dim) # the only tensor created
            self.h_bias = bias

    def sample(self, P):
        return torch.floor(torch.rand(P.size())+P) # torch.distributions.bernoulli.Bernoulli(P).sample()

    def forward(self, x):
        P=torch.sigmoid(self.h_bias+torch.mm(x, self.weights))
        return P

    def backward(self, x):
        P=torch.sigmoid(self.v_bias+torch.mm(x, self.weights.t()))
        return P

    def contrastive_divergence(self, x, update=True):
        h1 = self.sample(self.forward(x))
        v2 = self.sample(self.backward(h1))
        h2 = self.sample(self.forward(v2))

        dw = torch.mm(x.t(),h1) - torch.mm(v2.t(),h2) # delta weight
        dvb = torch.sum(v2 - x, dim=0)  # delta v_bias
        dhb = torch.sum(h1 - h2, dim=0) # delta h_bias
        lr = self.lr / self.batch_size
        if update:
            self.weights += lr * dw
            self.v_bias += lr * dvb
            self.h_bias += lr * dhb
        return torch.sum((x-v2)**2) #, torch.mean(torch.sum((x-v2)**2, dim=0))

    def train(self, data, epochs=10, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)
        for epoch in range(epochs):
            allcost=0.
            for batch in dataloader:
                batch = batch.view(len(batch), self.v_dim)
                a=(self.contrastive_divergence(batch))
                allcost+=a
            print("Epoch:{}, cost={}".format(epoch, allcost))

class sdnenet(torch.nn.Module):
    def __init__(self, encoder_layer_list, alpha, nu1, nu2, pretrain_lr=1e-5, pretrain_epoch=50):
        super(sdnenet, self).__init__()
        self.alpha = alpha
        self.nu1 = nu1
        self.nu2 = nu2  # loss parameters
        self.pretrain_lr = pretrain_lr
        self.pretrain_epoch = pretrain_epoch
        layer_collector = []
        for i in range(1, len(encoder_layer_list)):
            layer_collector.append(torch.nn.Linear(encoder_layer_list[i - 1], encoder_layer_list[i]))
            #layer_collector.append(torch.nn.LeakyReLU())
            layer_collector.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layer_collector)

        layer_collector1 = []
        for i in range(len(encoder_layer_list) - 2, -1, -1):
            layer_collector1.append(torch.nn.Linear(encoder_layer_list[i + 1], encoder_layer_list[i]))
            layer_collector.append(torch.nn.Sigmoid())

        self.decoder = torch.nn.Sequential(*layer_collector1)
        self.layer_collector = layer_collector + layer_collector1

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def pretrain(self, data): # deep-belief-network-based pretraining
        for layer in self.layer_collector:
            rbm = RBM(layer.weight, layer.bias, lr=self.pretrain_lr)
            rbm.train(data, epochs=self.pretrain_epoch)
            _,data = rbm.forward(data)

    def forward(self, a_b):
        embeddings = self.encoder(a_b)
        final = self.decoder(embeddings)
        return embeddings, final

    def _L_1st(self, a, e_n, e):
        return (a * (e_n - 2 * torch.mm(e, e.t()) + e_n.t())).sum()

    def _L_2nd(self, a_b, f, b):
        x1 = (((a_b - f) * b) ** 2).sum()
        return x1

    # L_2nd =
    def loss(self, a_b, a, b, embeddings, final):
        embeddings_norm = (embeddings ** 2).sum(1, keepdims=True)
        l1 = self._L_1st(a, embeddings_norm, embeddings)
        l2 = self._L_2nd(a_b, final, b)
        L = l2 + self.alpha * l1
        for param in self.layer_collector:
            if type(param) == torch.nn.Linear:
                L += self.nu2 * (param.weight ** 2).sum()  #self.nu1 * param.weight.abs().sum() +: not in the paper

        return L, l1, l2


class SDNE(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4,
                 batch_size=200, epoch=100, learning_rate=None):
        """
        encoder_layer_list: a list of numbers of the neuron at each encoder layer, the last number is the
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
        self.encoder_layer_num = len(encoder_layer_list) + 1

        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.bs = batch_size
        self.epoch = epoch
        self.max_iter = (epoch * self.node_size) // batch_size

        self.lr = lambda x: learning_rate
        if learning_rate is None:
            self.lr = lambda x: 0.03 / (1 + 0.9999 * x)
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
        return torch.from_numpy(adj).type(torch.float32)

    def train_one_epoch(self, model, optimizer, adj_batch_train, adj_mat_train, b_mat_train, step):
        optimizer.zero_grad()
        embeddings, final = model(adj_batch_train)
        loss, l1, l2 = model.loss(adj_batch_train, adj_mat_train, b_mat_train, embeddings, final)
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, step, decay_strategy=self.lr)
        if step % 5 == 0:
            print("step %i: total loss: %s, l1 loss: %s, l2 loss: %s" % (step, loss, l1, l2))

    def train(self):
        adj_mat = self.adj_mat

        model = sdnenet(self.encoder_layer_list, self.alpha, self.nu1, self.nu2)
        labels=torch.tensor([i['label'] for i in self.g.G.nodes],dtype=torch.float32)
        model.pretrain(labels)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr(0))
        print("total iter: %i" % self.max_iter)
        for step in range(self.max_iter):
            index = torch.randint(high=self.node_size,
                                  size=[self.bs])  # np.random.randint(self.node_size, size=self.bs)
            adj_batch_train = adj_mat[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = torch.ones_like(adj_batch_train)
            b_mat_train[adj_batch_train != 0] = self.beta
            self.train_one_epoch(model, optimizer, adj_batch_train, adj_mat_train, b_mat_train, step)
        embeddings,final= model(adj_mat)
        return embeddings.detach()

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(float(x)) for x in vec])))
        fout.close()


class SDNE2(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-5,
                 batch_size=100, max_iter=2000, learning_rate=None):

        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size] + encoder_layer_list
        self.encoder_layer_num = len(encoder_layer_list) + 1

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
                           name=scope_name + str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        with tf.name_scope(scope_name + 'decoder'):
            for i in range(self.encoder_layer_num - 2, -1, -1):
                fc = fc_op(fc,
                           name=scope_name + str(i),
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

        for step in range(self.max_iter + 1):
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