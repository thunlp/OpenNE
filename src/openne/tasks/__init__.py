from .classify import Classifier, TopKRanker
from .supervised_node_classification import SupervisedNodeClassification, supervisedmodels
from .unsupervised_node_classification import UnsupervisedNodeClassification
from .tasks import BaseTask

tasklist = [SupervisedNodeClassification, UnsupervisedNodeClassification]
taskdict = {Cls.__name__.lower(): Cls for Cls in tasklist}
