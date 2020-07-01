from .classify import Classifier, TopKRanker
from .supervised_node_prediction import SupervisedNodePrediction, supervisedmodels
from .unsupervised_node_prediction import UnsupervisedNodePrediction
from .tasks import BaseTask

tasklist = [SupervisedNodePrediction, UnsupervisedNodePrediction]
taskdict = {Cls.__name__.lower(): Cls for Cls in tasklist}
