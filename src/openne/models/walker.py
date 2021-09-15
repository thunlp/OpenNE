from __future__ import print_function
import random
# import numpy as np
import torch
import math
import multiprocessing
import networkx as nx
from time import time
import os


def wrapper(class_instance, epoch, walk_length):
    return class_instance.simulate_walks_one_epoch(epoch, walk_length)

class BasicWalker:
    def __init__(self, G, workers, silent=False):
        self.G = G.G   # nx.DiGraph(G.G)
        self.node_size = G.nodesize
        self.look_up_dict = G.look_up_dict
        self.silent = silent
        self.workers = None  # workers

    def rwalk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        # look_up_dict = self.look_up_dict
        # node_size = self.node_size

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        walk = [str(i) for i in walk]
        return walk

    def simulate_walks_one_epoch(self, epoch, walk_length):
        stime = time()
        self.debug("Run epoch {}".format(epoch))
        # print("Run epoch {} (PID {})".format(epoch, os.getpid()))
        G = self.G
        nodes = list(G.nodes())
        walks = []
        random.shuffle(nodes)
        for node in nodes:
            walks.append(self.rwalk(
                    walk_length=walk_length, start_node=node))
        etime = time()
        self.debug("Epoch {} ends in {} seconds.".format(epoch, etime - stime))
        # print("Epoch {} (PID {}) ends in {} seconds.".format(epoch, os.getpid(), etime - stime))
        return walks

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """

        walks = []

        self.debug('Walk iteration:')

        if self.workers:
            pool = multiprocessing.Pool(self.workers)

            walks_res = []
            for walk_iter in range(num_walks):
                walks_res.append(pool.apply_async(wrapper, args=(self, walk_iter, walk_length, )))

            pool.close()
            pool.join()

            for w in walks_res:
                walks.extend(w.get())

        else:
            for walk_iter in range(num_walks):
                walks.extend(self.simulate_walks_one_epoch(walk_iter, walk_length))

        # print(len(walks))
        return walks

    def debug(self, *args, **kwargs):
        if not self.silent:
            print(*args, **kwargs)


class Walker(BasicWalker):
    def __init__(self, G, p, q, workers, **kwargs):
        super(Walker, self).__init__(G, workers, **kwargs)
        self.p = p
        self.q = q

    def rwalk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    nxt = cur_nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                    walk.append(nxt)
            else:
                break
        walk = [str(i) for i in walk]
        return walk

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight']
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        look_up_dict = self.look_up_dict
        node_size = self.node_size
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    def debug(self, *args, **kwargs):
        if not self.silent:
            print(*args, **kwargs)


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = [0 for i in range(K)] # torch.zeros(K, dtype=torch.float32) # np.zeros(K, dtype=np.float32)
    J = [0.0 for i in range(K)] # np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)
    kk = int(math.floor(random.random()*K))

    if random.random() < q[kk]:
        return kk
    else:
        return J[kk]
