from abc import ABC
from .graph import *
import pickle as pkl
import scipy.sparse as sp
import torch

def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return mask.type(dtype=torch.bool)

class Planetoid(NetResources, ABC):
    def __init__(self):
        url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        name_dict = {name: 'ind.' + type(self).lname() + '.' + name for name in names}
        super(Planetoid, self).__init__(url, name_dict)
        # edge_attr, edge_index, test_mask, train_mask, val_mask, x, y

    @classmethod
    def lname(cls):
        return cls.__name__.lower()

    def read(self):
        obj = []
        for key, val in self.name_dict.items():
            path = osp.join(self.dir, val)
            if key == 'test.index':
                test_idx = np.loadtxt(path, dtype=int)
                continue
            with open(path, 'rb') as f:
                res = pkl.load(f, encoding='latin1')
                obj.append(res)
        x, tx, allx, y, ty, ally, graph = obj
        test_idx_range = np.sort(test_idx)

        if type(self).lname() == 'citeseer':
            # Fix citeseer dataloaders (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx), max(test_idx) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        self.G = nx.from_dict_of_lists(graph)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx, :] = features[test_idx_range, :]
        features = features.todense()
        self.set_node_features(features)

        labels = np.vstack((ally, ty))
        labels[test_idx, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        self.train_mask = sample_mask(idx_train, labels.shape[0])
        self.val_mask = sample_mask(idx_val, labels.shape[0])
        self.test_mask = sample_mask(idx_test, labels.shape[0])

        labels = [sp.coo_matrix(lbl).col for lbl in labels]
        self.set_node_label(labels)

        for i in self.G.nodes:
            for j in self.G.neighbors(i):
                self.G[i][j]['weight'] = 1.0

        self.G = nx.relabel_nodes(self.G, {i: str(i) for i in range(self.nodesize)})
        self.encode_node()

    @classmethod
    def attributed(cls):
        return True

class Cora(Planetoid):
    def __init__(self):
        super(Cora, self).__init__()

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

class CiteSeer(Planetoid):
    def __init__(self):
        super(CiteSeer, self).__init__()

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

class PubMed(Planetoid):
    def __init__(self):
        super(PubMed, self).__init__()

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False
