from .tasks import *
from ..classify import Classifier
from sklearn.linear_model import LogisticRegression

class UnsupervisedNodePrediction(BaseTask):
    def __init__(self, **kwargs):
        super(UnsupervisedNodePrediction, self).__init__(kwargs)

    def check(self, modelclass, datasetclass):
        pass

    @property
    def train_kwargs(self):
        return {}

    def evaluate(self, res, graph):
        vectors = res
        X, Y = graph.labels()
        print("Training classifier using {:.2f}% nodes...".format(
            self.kwargs['clf_ratio']*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, self.train_kwargs['clf_ratio'], seed=0)