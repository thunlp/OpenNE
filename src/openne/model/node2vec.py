from __future__ import print_function
import time
from gensim.models import Word2Vec
from . import walker
import torch
from .models import *

class Node2vec(ModelWithEmbeddings):

    def __init__(self, dim=128, dw=False):
        super(Node2vec, self).__init__(dim=dim, dw=dw)

    @classmethod
    def check_train_parameters(cls, graphtype, **kwargs):
        check_existance(kwargs, {'path_length': 80,
                                 'num_paths': 10,
                                 'p': 1.0,
                                 'q': 1.0,
                                 'workers': 1,
                                 'min_count': 0,
                                 'sg': 1})

    def build(self, graph, *, path_length=80, num_paths=10, p=1.0, q=1.0, **kwargs):

        if self.dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        if self.dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = walker.Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["size"] = kwargs.get("size", self.dim)

    def get_train(self, graph, **kwargs):
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = torch.Tensor(word2vec.wv[word])
        del word2vec
