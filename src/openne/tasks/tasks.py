import torch
from ..utils import *
class BaseTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def check(self, modelclass, datasetclass):
        raise NotImplementedError

    def train_kwargs(self) -> dict:
        check_existance(self.kwargs, {"validate": False, 'clf_ratio': 0.5})
        if not torch.cuda.is_available() or self.kwargs['cpu']:
            self.kwargs['data_parallel'] = False
            self.kwargs['_device'] = torch.device('cpu')
        else:
            self.kwargs['_device'] = torch.device('cuda', self.kwargs['devices'][0])
        return self.kwargs

    def _process(self, res):
        return res

    def split_dataset(self, graph, seed=None):
        if not self.kwargs['validate']:
            valsize = 0
        else:
            valsize = 100

        print(f"Creating test set using {self.kwargs['clf_ratio'] * 100}% nodes as training set...", end='')
        graph.get_split_data(train_percent=self.kwargs['clf_ratio'], validate_size=valsize, seed=seed)
        print('finished')

    def train(self, model, graph):
        print(f"Executing task {self.__class__.__name__}.")
        self.split_dataset(graph)
        res = model(graph, **self.train_kwargs())
        return self._process(res)

    def evaluate(self, model, res, dataset):
        raise NotImplementedError

