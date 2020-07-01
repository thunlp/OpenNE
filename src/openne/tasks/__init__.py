from .classify import Classifier, TopKRanker
from .supervised_node_prediction import SupervisedNodePrediction
from .unsupervised_node_prediction import UnsupervisedNodePrediction
from .tasks import BaseTask

tasklist = [SupervisedNodePrediction, UnsupervisedNodePrediction]
taskdict = {Cls.__name__: Cls for Cls in tasklist}
