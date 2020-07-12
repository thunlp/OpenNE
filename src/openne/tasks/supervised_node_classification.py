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
        
    def train(self, model, graph):
        self.split_dataset(graph)
        return super(SupervisedNodeClassification, self).train(model, graph)

    def split_dataset(self, graph):
        """
            build train_mask test_mask val_mask
        """
        train_percent = self.kwargs['clf_ratio']
        print("Creating test set using {}% nodes".format(train_percent * 100))
        training_size = int(train_percent * graph.G.number_of_nodes())
        state = torch.random.get_rng_state()
        torch.random.manual_seed(0)
        shuffle_indices = torch.randperm(graph.G.number_of_nodes())
        torch.random.set_rng_state(state)
        g = graph.G

        def sample_mask(begin, end):
            mask = torch.zeros(g.number_of_nodes())
            for i in range(begin, end):
                mask[shuffle_indices[i]] = 1
            return mask

        self.train_mask = sample_mask(0, training_size - 100)
        self.val_mask = sample_mask(training_size - 100, training_size)
        self.test_mask = sample_mask(training_size, g.number_of_nodes())
        self.kwargs['train_mask'] = self.train_mask

    def train_kwargs(self):
        check_existance(self.kwargs, {"validate": True, 'clf_ratio': 0.5})

        def validation_hook(model, graph, **kwargs):
            _, cost, acc, duration = model.evaluate(self.val_mask)
            model.cost_val.append(cost)
            model.debug_info += '; val_loss = {:.5f}, val_acc = {:.5f}'.format(cost, acc)
        check_existance(self.kwargs, {'_validation_hooks': [validation_hook] if self.kwargs['validate'] else []})
        return self.kwargs

    def evaluate(self, model, res, dataset):
        # # Testing
        test_res, test_cost, test_acc, test_duration = model.evaluate(self.test_mask, False)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))