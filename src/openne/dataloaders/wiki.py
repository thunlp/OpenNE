from .graph import *

class Wiki(LocalFile):
    def __init__(self, **kwargs):
        root = '../data/wiki'
        super(Wiki, self).__init__(root, {'edgefile': 'Wiki_edgelist.txt', 'labelfile': 'wiki_labels.txt'}, **kwargs)



    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def attributed(cls):
        return False
