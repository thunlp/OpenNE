"""
refer to Dataset of CogDL
"""
from abc import ABC

import torch.utils.data
import collections
import os.path as osp
import networkx as nx
import numpy as np
from ..utils import *
# from torch_geometric.data import download_url
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
    def __init__(self, name, resource_url, root_dir, downloaded_names, processed_names, *,
                 downloaded_dir='downloaded', processed_dir='processed'):
        """
        :param name:
        :param resource_url:
        :param root_dir:
        :param downloaded_names:
        :param processed_names:
        :param downloaded_dir: RELATIVE. root_dir/downloaded_dir.  if downloaded_dir==None: no downloaded dir
        :param processed_dir:
        """
        super(Dataset, self).__init__()
        self.name = name
        self.resource_url = resource_url
        self.dir = root_dir

        if downloaded_dir is not None:
            self.downloaded_dir = osp.join(root_dir, downloaded_dir)
            self.downloaded_names = to_list(downloaded_names)
            self.downloaded_paths = [self.fulldwn(f) for f in self.downloaded_names]
        else:
            self.downloaded_dir, self.downloaded_names, self.downloaded_paths = None, None, None
        if processed_dir is not None:
            self.processed_dir = osp.join(root_dir, processed_dir)
            self.processed_names = to_list(processed_names)
            self.processed_paths = [self.fullpro(f) for f in self.processed_names]
        else:
            self.processed_dir, self.processed_names, self.processed_paths = None, None, None
        self.load_data()

    # "edgelist.txt" -> "OPENNE/downloaded/edgelist.txt"
    def fulldwn(self, filename):
        return osp.join(self.downloaded_dir, filename)

    def fullpro(self, filename):
        return osp.join(self.processed_dir, filename)

    def load_data(self):
        """
            processed_paths and download_paths: 1. check if downloaded (download) 2. check if processed (process) 3. read if processed
            processed_paths is None: 1. check if downloaded (download) 2. process
            downloaded_paths is None: 1. check if processed (process) 2. read if processed
        """
        if self.processed_paths is None or not files_exist(self.processed_paths):
            if self.processed_paths is not None:
                makedirs(self.processed_dir)
            if self.downloaded_paths is not None and not files_exist(self.downloaded_paths):
                makedirs(self.downloaded_dir)
                print('Downloading dataloaders "{}" from "{}".\n'
                      'Files will be saved to "{}".'.format(self.name, self.resource_url, self.downloaded_dir))
                self.download()
                print('Downloaded.')
            print('Processing dataloaders. \n'
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
    def __init__(self, name, resource_url, root_dir, downloaded_name_dict, processed_name_dict, **kwargs):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        # self.node_size = 0
        self.downloaded_name_dict = downloaded_name_dict
        self.processed_name_dict = processed_name_dict

        defaultkwargs = {}
        for kw in set(defaultkwargs).union(set(kwargs)):
            self.__setattr__(kw, kwargs.get(kw, defaultkwargs[kw]))

        super(Graph, self).__init__(name, resource_url, root_dir, [i for k, i in downloaded_name_dict.items()],
                                    [i for k, i in processed_name_dict.items()], **kwargs)

    def read(self):
        name_dict = self.processed_name_dict
        if 'edgefile' in name_dict:
            self.read_edgelist(self.fulldwn(name_dict['edgefile']))
        elif 'adjfile' in name_dict:
            self.read_adjlist(self.fulldwn(name_dict['adjfile']))
        if 'labelfile' in name_dict:
            self.read_node_label(self.fulldwn(name_dict['labelfile']))
        if 'features' in name_dict:
            self.read_node_features(self.fulldwn(name_dict['features']))
        if 'status' in name_dict:
            self.read_node_status(self.fulldwn(name_dict['status']))

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
        A = nx.adjacency_matrix(G)
        if not sparse:
            A = np.array(nx.adjacency_matrix(G).todense())
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

        if self.directed():
            self.G = nx.DiGraph()

            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            self.G = nx.Graph()

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
        if self.weighted():
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


class LocalFileWithProcess(Graph, ABC):
    def __init__(self, name, root_dir, downloaded_name_dict, processed_name_dict, **kwargs):
        super(LocalFileWithProcess, self).__init__(name, None, root_dir, downloaded_name_dict, processed_name_dict,
                                                   downloaded_dir='.', **kwargs)

    def download(self):
        pass


class LocalFileWithoutProcess(Graph, ABC):
    def __init__(self, name, root_dir, processed_name_dict, **kwargs):
        super(LocalFileWithoutProcess, self).__init__(name, None, root_dir, {}, processed_name_dict,
                                                      downloaded_dir=None, processed_dir='.', **kwargs)

    def download(self):
        pass

    def process(self):
        pass


class NetResources(Graph, ABC):
    def __init__(self, name, resource_url, downloaded_name_dict, processed_name_dict, **kwargs):
        self.name = name
        super(NetResources, self).__init__(name, resource_url, self.root_dir,
                                           downloaded_name_dict, processed_name_dict, **kwargs)

    @property
    def root_dir(self):
        return osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', self.name)  #########################
