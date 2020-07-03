import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def scipy_to_torch(scipy_tensor, dtype=None):
    return torch.sparse_coo_tensor((scipy_tensor.row,scipy_tensor.col),scipy_tensor.data, scipy_tensor.shape, dtype=dtype)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if type(mx) == torch.Tensor:
            mx=mx.coalesce()
            coords=torch.stack((mx.indices()[0],mx.indices()[1])).t()
            values=mx.values()
            shape=mx.shape
        else:
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = torch.stack((torch.tensor(mx.row), torch.tensor(mx.col))).t() #vstack
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def tuple_to_sparse(tuple):
    w=tuple[0].t()
    return torch.sparse.FloatTensor(w.to(dtype=torch.long), torch.tensor(tuple[1]), torch.Size(tuple[2]))

def preprocess_features(features, sparse=False):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = torch.tensor(features.sum(1)) #np.array(features.sum(1))
    r_inv = (rowsum**-1).flatten() #np.power(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)#sp.diags(r_inv)
    features = r_mat_inv.mm(features)
    return sparse_to_tuple(features.to_sparse()) if sparse else features


def normalize_adj(adj): #  safe. don't change by now
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN models and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return scipy_to_torch(adj_normalized, torch.float32)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return [scipy_to_torch(st, torch.float32) for st in t_k]
