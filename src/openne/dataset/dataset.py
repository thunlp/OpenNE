"""
refer to Dataset of CogDL
"""
from abc import ABC

import torch.utils.data
import collections
import os.path as osp
import networkx as nx
import numpy as np
from torch_geometric.data import download_url
import os
import errno


def to_list(x):
    if x is None:
        return []
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    return all([osp.exists(f) for f in files])


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name, resource_url, root_dir, downloaded_names, processed_names):
        super(Dataset, self).__init__()
        self.name = name
        self.resource_url = resource_url
        self.dir = root_dir
        self.downloaded_dir = osp.join(root_dir, 'downloaded')
        self.processed_dir = osp.join(root_dir, 'processed')
        self.downloaded_names = downloaded_names
        self.processed_names = processed_names
        self.downloaded_paths = [osp.join(self.downloaded_dir, f) for f in to_list(self.downloaded_names)]
        self.processed_paths = [osp.join(self.processed_dir, f) for f in to_list(self.processed_names)]
        self.load_data()

    def load_data(self):
        if not files_exist(self.processed_paths):
            if not files_exist(self.downloaded_paths):
                print('Downloading dataset "{}" from "{}".\n'
                      'Files will be saved to "{}".'.format(
                    self.name, self.resource_url, self.downloaded_dir))
                makedirs(self.downloaded_dir)
                self.download()
                print('Downloaded.')
            print('Processing dataset. \n'
                  'Processed files will be saved to {}'.format(self.processed_dir))
            self.process()
            print('All done.\n')
        else:
            self.read()

    def download(self):
        for name in self.downloaded_names:
            download_url('{}/{}'.format(self.resource_url, name), self.downloaded_dir)

    def process(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError


class Graph(Dataset):
    def __init__(self, name, resource_url, root_dir, downloaded_name_dict, **kwargs):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        # self.node_size = 0
        self.downloaded_name_dict = downloaded_name_dict

        defaultkwargs = {}
        for kw in set(defaultkwargs).union(set(kwargs)):
            self.__setattr__(kw, kwargs.get(kw, defaultkwargs[kw]))

        super(Graph, self).__init__(name, resource_url, root_dir, [i for k, i in downloaded_name_dict.items()],
                                    "graph.sparse6")

    def process(self):
        downloaded_name_dict = self.downloaded_name_dict
        if 'edgefile' in downloaded_name_dict:
            self.read_edgelist(downloaded_name_dict['edgefile'])
        elif 'adjfile' in downloaded_name_dict:
            self.read_adjlist(downloaded_name_dict['adjfile'])
        if 'labelfile' in downloaded_name_dict:
            self.read_node_label(downloaded_name_dict['labelfile'])
        if 'features' in downloaded_name_dict:
            self.read_node_features(downloaded_name_dict['features'])
        if 'status' in downloaded_name_dict:
            self.read_node_status(downloaded_name_dict['status'])
        nx.write_sparse6(self.G, self.processed_paths[0])

    def read(self):
        self.G = nx.read_sparse6(self.processed_paths[0])
        self.encode_node()

    @classmethod
    def directed(cls):
        raise NotImplementedError

    @classmethod
    def weighted(cls):
        raise NotImplementedError

    @classmethod
    def attributed(cls):
        raise NotImplementedError

    def adjmat(self, directed, weighted, scaled=None, sparse=False):
        G = self.G
        if type(self).directed() and not directed:
            G = nx.to_undirected(G)
        if not sparse:
            A = nx.to_numpy_matrix(G, nodelist=range(G.number_of_nodes()))
        else:
            A = nx.to_scipy_sparse_matrix(G, nodelist=range(G.number_of_nodes()))
        if type(self).weighted() and not weighted:
            A = A.astype(np.bool).astype(np.float32)
        if scaled is not None:  # e.g. scaled = 1
            A = A / A.sum(scaled, keepdims=True)
        return A

    def labels(self):
        X = []
        Y = []
        for i in self.G.nodes:
            X.append(i)
            Y.append(self.G.nodes[i]['label'])
        return X, Y

    @property
    def nodesize(self):
        return self.G.number_of_nodes()

    @property
    def edgesize(self):
        return self.G.number_of_edges()

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        node_size = 0
        for node in self.G.nodes():
            look_up[node] = node_size
            look_back.append(node)
            node_size += 1
            self.G.nodes[node]['status'] = ''

    def set_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename):
        self.G = nx.DiGraph()
        if self.directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if self.weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    # use after G is not none
    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    # use after G is not none
    def set_node_label(self, labelvectors, split=False):
        for i, vec in enumerate(labelvectors):
            if split:
                self.G.nodes[vec[0]]['label'] = vec[1:]
            else:
                self.G.nodes[i]['label'] = vec

    # use after G is not none
    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array([float(x) for x in vec[1:]])
        fin.close()

    # use after G is not none
    def set_node_features(self, featurevectors, split=False):
        for i, vec in enumerate(featurevectors):
            if split:
                self.G.nodes[vec[0]]['feature'] = vec[1:]
            else:
                self.G.nodes[i]['feature'] = vec

    # use after encode_node()
    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()

    def set_edge_attr(self, edgelist, edgeattrvectors):
        for i in range(len(edgelist)):
            self.G[edgelist[i][0]][edgelist[i][1]]['feature'] = edgeattrvectors[i]


class SelfDefined(Graph, ABC):
    def __init__(self, name, root_dir, name_dict, **kwargs):
        super(SelfDefined, self).__init__(name, None, root_dir, name_dict, **kwargs)

    def download(self):
        pass


class NetResources(Graph, ABC):
    def __init__(self, name, resource_url, name_dict, **kwargs):
        self.name = name
        self.name_dict = name_dict
        super(NetResources, self).__init__(name, resource_url, self.root_dir, name_dict, **kwargs)

    @property
    def root_dir(self):
        return osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', self.name)  #########################
