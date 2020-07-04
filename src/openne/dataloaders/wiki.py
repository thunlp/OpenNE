from .graph import *

class Wiki(LocalFile):
    def __init__(self):
        root = '../data/wiki'
        super(Wiki, self).__init__(root, {'edgefile': 'Wiki_edgelist.txt', 'labelfile': 'wiki_labels.txt'})
        print(self.look_back_list[:10])


    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def attributed(cls):
        return False
