from __future__ import print_function
import time
from gensim.models import Word2Vec
from . import walker
import torch
from .models import *

class Node2vec(ModelWithEmbeddings):

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
                                 'workers': 1,
                                 'min_count': 0,
                                 'sg': 1})
        return kwargs

    def build(self, graph, *, path_length=80, num_paths=10, p=1.0, q=1.0, **kwargs):

        if self.dw:
            self.args['hs'] = 1
            p = 1.0
            q = 1.0

        self.args['p'] = p
        self.args['q'] = q
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
        for s in ["min_count", "sg"]:
            self.args[s] = kwargs[s]

    def get_train(self, graph, **kwargs):
        word2vec = Word2Vec(self.args)
        self.vectors = {}

        for word in graph.G.nodes():
            self.vectors[word] = torch.Tensor(word2vec.wv[word])
        del word2vec

class DeepWalk(Node2vec):
    def __init__(self, dim=128, **kwargs):
        super(DeepWalk, self).__init__(dim, True, **kwargs)