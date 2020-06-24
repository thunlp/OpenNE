import networkx as nx
import numpy as np
import torch
from ..utils import *
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import tensorflow as tf
from time import time
from sklearn.preprocessing import normalize
from .models import *

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class HOPE(ModelWithEmbeddings):
    def __init__(self, d):
        """
        :param d: representation dim
        """
        super(HOPE, self).__init__(_d=d)

    @classmethod
    def check_train_parameters(cls, graphtype, **kwargs):
        check_existance(kwargs, {'measurement': 'katz'})
        check_range(kwargs, {'measurement': ['katz', 'cn', 'rpr', 'aa']})
        if kwargs['measurement'] == 'katz':
            check_existance(kwargs, {'beta': 0.1})
        elif kwargs['measurement'] == 'rpr':
            check_existance(kwargs, {'alpha': 0.5})

    def get_train(self, graph, *, measurement='katz', **kwargs):
        n = graph.nodesize
        A = graph.adjmat(directed=True, weighted=False)  # brute force...
        if measurement == 'katz': # Katz: M_g^-1 * M_l = (I - beta * A)^-1 - I
            S = ((np.eye(n) - kwargs['beta'] * A).I - np.eye(n))
            # M_g = np.eye(n) - kwargs['beta'] * A
            # M_l = kwargs['beta'] * A
        elif measurement == 'cn': # Common Neighbors: S = A^2
            S = np.matmul(A, A)
            # M_g = I
            # M_l = A^2
        elif measurement == 'rpr': # Rooted PageRank: (1 - alpha)(I - alpha * P)^-1
            P = graph.adjmat(directed=True, weighted=False, scaled=1)  # scaled=0 in paper but possibly wrong?
            S = (1 - kwargs['alpha']) * (np.eye(n) - kwargs['alpha'] * P).I
        else: # Adamic-Adar: Mg^-1 * M_l
            D = np.eye(n)
            for i in range(n):
                k = sum(A[i][j] + A[j][i] for j in range(n))
                D[i][i] /= k
            S = np.matmul(np.matmul(A, D), A)

        # todo: check if the model REALLY DON'T NEED M_g and M_l!
        u, s, vt = lg.svds(S, k=self._d // 2)  # this one directly use the d/2-dim core for svd

        sigma = np.diagflat(np.sqrt(s))
        X1 = normalize(np.matmul(u, sigma))
        X2 = normalize(np.matmul(vt.T, sigma))
        return torch.cat((torch.tensor(X1), torch.tensor(X2)), dim=1)