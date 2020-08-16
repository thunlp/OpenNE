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

    def train(self, model, graph):
        res = model(graph, **self.train_kwargs())
        return self._process(res)

    def evaluate(self, model, res, dataset):
        raise NotImplementedError

