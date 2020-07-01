import torch

class BaseTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def check(self, modelclass, datasetclass):
        raise NotImplementedError

    @property
    def model_kwargs(self):
        return {}

    @property
    def train_kwargs(self):
        return {}

    def _process(self, res):
        return res

    def train(self, model, graph):
        res = model(graph, **self.train_kwargs)
        return self._process(res)

    def evaluate(self, model, res, dataset):
        raise NotImplementedError

