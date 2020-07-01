from .dataset import Dataset, Graph, LocalFile, Adapter, NetResources
from .karate import Karate
from .matlab_matrix import MatlabMatrix, PPI, Wikipedia, Flickr
from .wiki import Wiki
from .planetoid_dataset import PubMed, Reddit, Cora, CiteSeer

datasetlist = [Karate, PPI, Wikipedia, Flickr, Wiki, PubMed, Reddit, Cora, CiteSeer]
datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}

