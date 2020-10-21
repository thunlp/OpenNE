import scipy.sparse.linalg as sla
import scipy.linalg as la
from sklearn.preprocessing import normalize

from .models import *

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class HOPE(ModelWithEmbeddings):
    def __init__(self, dim, **kwargs):
        """
        :param dim: representation dim
        """
        super(HOPE, self).__init__(dim=dim, **kwargs)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'dim': 128})
        check_range(kwargs, {'dim': 'positive'})
        if 'measurement' not in kwargs:
            check_existance(kwargs, {'beta': 0.02})
            check_existance(kwargs, {'alpha': 0.5})
        check_existance(kwargs, {'measurement': 'katz'})
        check_range(kwargs, {'measurement': ['katz', 'cn', 'rpr', 'aa']})
        if kwargs['measurement'] == 'katz':
            check_existance(kwargs, {'beta': 0.02})
        if kwargs['measurement'] == 'rpr':
            check_existance(kwargs, {'alpha': 0.5})
        return kwargs

    def train_model(self, graph, *, measurement='katz', **kwargs):
        n = graph.nodesize
        A = graph.adjmat(directed=True, weighted=False)  # brute force...
        if measurement == 'katz':  # Katz: M_g^-1 * M_l = (I - beta * A)^-1 - I
            S = (la.inv(np.identity(n) - kwargs['beta'] * A) - np.identity(n))

            # M_g = np.eye(n) - kwargs['beta'] * A
            # M_l = kwargs['beta'] * A
        elif measurement == 'cn':  # Common Neighbors: S = A^2
            S = np.matmul(A, A)
            # M_g = I
            # M_l = A^2
        elif measurement == 'rpr':  # Rooted PageRank: (1 - alpha)(I - alpha * P)^-1
            P = graph.adjmat(directed=True, weighted=False, scaled=1)  # scaled=0 in paper but possibly wrong?
            S = (1 - kwargs['alpha']) * la.inv(np.eye(n) - kwargs['alpha'] * P)
        else: # Adamic-Adar: Mg^-1 * M_l
            D = np.eye(n)
            for i in range(n):
                k = sum(A[i][j] + A[j][i] for j in range(n))
                D[i][i] /= k
            S = np.matmul(np.matmul(A, D), A)

        u, s, vt = sla.svds(S, k=self.dim // 2)  # this one directly use the d/2-dim core for svd

        sigma = np.sqrt(s)
        X1 = normalize(u * sigma)
        X2 = normalize(vt.T * sigma)

        # another implementation
        # sigma = np.diagflat(np.sqrt(s))
        # X1 = normalize(np.matmul(u, sigma))
        # X2 = normalize(np.matmul(vt.T, sigma))

        return torch.cat((torch.tensor(X1), torch.tensor(X2)), dim=1)
