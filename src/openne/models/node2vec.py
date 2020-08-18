from __future__ import print_function
import time
from gensim.models import Word2Vec
from . import walker
import torch
from .models import *
class Node2vec(ModelWithEmbeddings):
    """
        Make sure graph.G is a networkx.DiGraph. If not, turn it into DiGraph using
        .. sourcecode:: pycon
            >>>import networkx as nx
            >>>graph.G = nx.DiGraph(graph.G)
    """
    def __init__(self, dim=128, dw=False, **kwargs):
        super(Node2vec, self).__init__(dim=dim, dw=dw, **kwargs)
        self.args = {}

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'dim': 128,
                                 'path_length': 80,
                                 'num_paths': 10,
                                 'p': 1.0,
                                 'q': 1.0,
                                 'window': 10,
                                 'workers': 8,
                                 'max_vocab_size': None,  #1 << 32,  # 4 GB
                                 })
        return kwargs

    def build(self, graph, *, path_length=80, num_paths=10, p=1.0, q=1.0, **kwargs):
        if self.dw:
            self.args['hs'] = 1
            p = 1.0
            q = 1.0
        self.args['workers'] = kwargs["workers"]

        if self.dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = walker.Walker(graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        self.args["sentences"] = sentences
        self.args["size"] = self.dim
        self.args['min_count'] = 0
        self.args['window'] = kwargs['window']
        self.args['sg'] = 1
        self.args['max_vocab_size'] = kwargs['max_vocab_size']

    def train_model(self, graph, **kwargs):
        print("training Word2Vec model...")
        word2vec = Word2Vec(**self.args)
        self.vectors = {}
        print("Obtaining vectors...")
        for word in graph.G.nodes():
            self.vectors[word] = torch.tensor(word2vec.wv[str(word)])
        del word2vec


class DeepWalk(Node2vec):
    def __init__(self, dim=128, **kwargs):
        super(DeepWalk, self).__init__(dim, True, **kwargs)