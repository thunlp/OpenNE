from .tasks import *
from ..classify import Classifier
from ..utils import *
from ..model.models import ModelWithEmbeddings
from sklearn.linear_model import LogisticRegression

class SupervisedNodePrediction(BaseTask):
    def __init__(self, **kwargs):
        super(SupervisedNodePrediction, self).__init__(**kwargs)

    def check(self, modelclass, datasetclass):
        assert(issubclass(modelclass, ModelWithEmbeddings))
        self.kwargs = modelclass.check_train_parameters(datasetclass, **self.train_kwargs)

    def split_dataset(self, graph):
        """
            build train_mask test_mask val_mask
        """
        train_precent = self.kwargs['clf_ratio']
        training_size = int(train_precent * graph.G.number_of_nodes())
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

    @property
    def train_kwargs(self):
        def validation_hook(model, graph, **kwargs):
            _, cost, acc, duration = model.evaluate(self.val_mask)
            model.cost_val.append(cost)
            model.debug_info['val_loss'] = "{:.5f}".format(cost)
            model.debug_info['val_acc'] = "{:.5f}".format(acc)
        check_existance(self.kwargs, {'validation_hooks': [validation_hook]})
        return self.kwargs

    def evaluate(self, model, res, dataset):
        # # Testing
        test_res, test_cost, test_acc, test_duration = model.evaluate(self.test_mask, False)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))