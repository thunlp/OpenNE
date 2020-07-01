"""
refer to matlab_matrix.py of CogDL
"""
from abc import ABC

from .dataset import *
import scipy.io
class MatlabMatrix(NetResources, ABC):
    def __init__(self, resource_url, filename):
        super(MatlabMatrix, self).__init__(resource_url, {'matfile': filename + '.mat'})

    def read(self):
        path = self.paths[0]
        smat = scipy.io.loadmat(path)
        adjmat, group = smat["network"], smat["group"]
        self.G = nx.from_scipy_sparse_matrix(adjmat)
        self.set_node_label(group)


class BlogCatalog(MatlabMatrix):
    def __init__(self):
        filename = 'blogcatalog'
        url = 'http://leitang.net/code/social-dimension/data'
        super(BlogCatalog, self).__init__(resource_url=url, filename=filename)

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return False

class Flickr(MatlabMatrix):
    def __init__(self):
        filename = 'flickr'
        url = 'http://leitang.net/code/social-dimension/data'
        super(Flickr, self).__init__(resource_url=url, filename=filename)

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return False

class Wikipedia(MatlabMatrix):
    def __init__(self):
        filename = 'POS'
        url = 'http://snap.stanford.edu/node2vec'
        super(Wikipedia, self).__init__(resource_url=url, filename=filename)

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return False

class PPI(MatlabMatrix):
    def __init__(self):
        filename = 'Homo_sapiens'
        url = 'http://snap.stanford.edu/node2vec'
        super(PPI, self).__init__(resource_url=url, filename=filename)

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return False

