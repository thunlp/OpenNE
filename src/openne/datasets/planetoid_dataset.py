from abc import ABC
from torch_geometric.datasets import Planetoid
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
from .dataset import *

class TorchGeometricAttributed(Adapter, ABC):
    def __init__(self, AdapteeClass):
        super(TorchGeometricAttributed, self).__init__(AdapteeClass, self.root_dir,
                                                       type(self).__name__, T.TargetIndegree())
        # edge_attr, edge_index, test_mask, train_mask, val_mask, x, y

    def read(self):
        self.G = nx.from_edgelist(self.data[0]['edge_index'].t().numpy())
        n = self.data[0]['x'].numpy().shape[0]
        for i in range(n):
            if i not in self.G.nodes:
                self.G.add_node(i)
        self.encode_node()
        self.set_node_features(self.data[0]['x'].numpy())
        self.set_node_label(np.expand_dims(self.data[0]['y'].numpy(), 1))
        self.set_edge_attr(self.data[0]['edge_index'].t().numpy(), self.data[0]['edge_attr'].numpy())


    @classmethod
    def attributed(cls):
        return True

class Cora(TorchGeometricAttributed):
    def __init__(self):
        super(Cora, self).__init__(Planetoid)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

class CiteSeer(TorchGeometricAttributed):
    def __init__(self):
        super(CiteSeer, self).__init__(Planetoid)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

class PubMed(TorchGeometricAttributed):
    def __init__(self):
        super(PubMed, self).__init__(Planetoid)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

class RedditAdapter(datasets.Reddit):
    def __init__(self, path, dataset, transform):
        super(RedditAdapter, self).__init__(path, transform)

class Reddit(TorchGeometricAttributed):
    def __init__(self):
        super(Reddit, self).__init__(RedditAdapter)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False
