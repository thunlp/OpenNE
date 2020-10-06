from .. import utils
from .tasks import *
from ..utils import *
from ..models import GCN

supervisedmodels = [GCN]
class SupervisedNodeClassification(BaseTask):
    def __init__(self, **kwargs):
        super(SupervisedNodeClassification, self).__init__(**kwargs)

    def check(self, modelclass, datasetclass):
#        assert(any(issubclass(modelclass, cls) for cls in supervisedmodels))
        self.kwargs = modelclass.check(datasetclass, **self.train_kwargs())

        # self.kwargs['train_mask'] = graph.train_mask

    def train_kwargs(self):
        #  by default validate == True
        #  iff --no-validate set: False
        check_existance(self.kwargs, {"_validate": False, "_no_validate": False})
        check_existance(self.kwargs, {"validate": not self.kwargs["_no_validate"], 'clf_ratio': 0.5})

        def validation_hook(model, graph, **kwargs):
            _, cost, acc, duration = model.evaluate(graph.val_mask)
            model.cost_val.append(cost)
            model.debug_info += '; val_loss = {:.5f}, val_acc = {:.5f}'.format(cost, acc)
        check_existance(self.kwargs, {'_validation_hooks': [validation_hook] if self.kwargs['validate'] else []})
        super(SupervisedNodeClassification, self).train_kwargs()
        return self.kwargs

    def evaluate(self, model, res, dataset):
        # # Testing
        test_res, test_cost, test_acc, test_duration = model.evaluate(dataset.test_mask, False)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))