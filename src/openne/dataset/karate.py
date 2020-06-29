from .dataset import *

class Karate(NetResources):
    def __init__(self):
        url = 'https://raw.githubusercontent.com/thunlp/OpenNE/master/data/karate'
        super(Karate, self).__init__('karate', url, {'edgefile': 'karate.edgelist'})

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def directed(cls):
        return False

    @classmethod
    def attributed(cls):
        return False
