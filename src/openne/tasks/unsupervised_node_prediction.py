from .tasks import BaseTask
from .classify import Classifier
from ..utils import *
from ..models import ModelWithEmbeddings
from sklearn.linear_model import LogisticRegression

class UnsupervisedNodePrediction(BaseTask):
    def __init__(self, **kwargs):
        super(UnsupervisedNodePrediction, self).__init__(**kwargs)

    def check(self, modelclass, datasetclass):
        assert(issubclass(modelclass, ModelWithEmbeddings))
        self.kwargs = modelclass.check(datasetclass, **self.train_kwargs)

    @property
    def train_kwargs(self):
        def f_v(model, graph, **kwargs):
            model.get_vectors(graph)
            res = self._classify(graph, model.vectors)
            if model.setvalue('best_result', res['macro']):
                if kwargs['auto_save']:
                    model.setvalue('best_vectors', model.vectors, lambda x, y: True)
        check_existance(self.kwargs, {'auto_save': True, 'validation_hooks': [f_v]})
        return self.kwargs

    def evaluate(self, model, res, graph):
        self._classify(graph, res, 0)

    def _classify(self, graph, vectors, seed=None):
        X, Y = graph.labels()
        print("Training classifier using {:.2f}% nodes...".format(
            self.kwargs['clf_ratio']*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        return clf.split_train_evaluate(X, Y, self.train_kwargs['clf_ratio'], seed=seed)