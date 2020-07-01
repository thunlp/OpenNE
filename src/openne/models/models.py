import torch
import os
from time import time
from ..utils import *

class BaseModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

        for i in kwargs:
            self.__setattr__(i, kwargs[i])

    def forward(self, dataset):
        raise NotImplementedError

    def setvalue(self, attribute_name, value, compare_function=lambda x, y: True if x < y else False):
        """
        Update certain attribute with given new value.
        :param attribute_name: attribute name.
        :param value: new value.
        :param compare_function: Bool type. Accepts two values, the old and the new.
                Returns True if the new is to be updated.
        :return: bool, whether the new one is updated.
        """
        if attribute_name not in self.__dict__:
            self.__dict__[attribute_name] = value
            return True
        else:
            if compare_function(self.__dict__[attribute_name], value):
                self.__dict__[attribute_name] = value
                return True
            return False

class ModelWithEmbeddings(BaseModel):
    def __init__(self, *, output=None, save=True, **kwargs):
        self.vectors = {}
        self.embeddings = None
        self.debug_info = None
        if save:
            if output is None:
                outputpath = os.path.join('result', type(self).__name__)
                outputmodelfile = type(self).__name__ + '_model.txt'
                outputembeddingfile = type(self).__name__ + '_embeddings.txt'
            else:
                outputpath = os.path.dirname(output)
                if outputpath == '':
                    outputpath = '.'
                outputmodelfile = type(self).__name__ + 'models.txt'
                outputembeddingfile = os.path.basename(output)
            if not os.path.isdir(self.outputpath):
                os.mkdir(self.outputpath)
            try:
                with open(os.path.join(self.outputpath, self.outputembeddingfile), 'a'):
                    pass
            except Exception as e:
                raise FileNotFoundError('Failed to open target embedding file "{}": {}. '.format(
                    os.path.join(self.outputpath, self.outputembeddingfile), str(e)))
            try:
                with open(os.path.join(self.outputpath, self.outputembeddingfile), 'a'):
                    pass
            except Exception as e:
                raise FileNotFoundError('Failed to open target models file "{}": {}. '.format(
                    os.path.join(self.outputpath, self.outputembeddingfile), str(e)))

            super(ModelWithEmbeddings, self).__init__(outputpath=outputpath,
                                                      outputmodelfile=outputmodelfile,
                                                      outputembeddingfile=outputembeddingfile,
                                                      save=save,
                                                      **kwargs)
        else:
            super(ModelWithEmbeddings, self).__init__(save=save, **kwargs)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def save_embeddings(self, filename):
        with open(filename, 'w') as fout:
            node_num = len(self.vectors)
            fout.write("{} {}\n".format(node_num, self.rep_size))
            for node, vec in self.vectors.items():
                fout.write("{} {}\n".format(node, ' '.join([str(float(x)) for x in vec])))

    def load(self, path=None):
        if not os.path.isfile(path):
            raise FileNotFoundError("Model file not found.")
        self.load_state_dict(torch.load(path))

    @classmethod
    def check_train_parameters(cls, graphtype, **kwargs):
        return kwargs

    def get_vectors(self, graph):
        self.vectors = {}
        for i, embedding in enumerate(self.embeddings):
            self.vectors[graph.look_back_list[i]] = embedding
        return self.vectors

    def _check_train_parameters(self, graphtype, **kwargs):
        new_kwargs = kwargs.copy()
        ret_args = self.check_train_parameters(graphtype, **new_kwargs)
        if ret_args is not None:
            new_kwargs = ret_args
        if 'epochs' not in new_kwargs:
            epoch_debug_output = None
        else:
            epoch_debug_output = 5
        check_existance(new_kwargs, {'epochs': 1,
                                     'validation_hooks': [],
                                     'debug_output_interval': epoch_debug_output})
        return new_kwargs

    def build(self, graph, **kwargs):
        pass

    def get_train(self, graph, **kwargs):
        raise NotImplementedError

    def early_stopping_judge(self, graph, **kwargs):
        return False

    def make_output(self, graph, **kwargs):
        pass

    def forward(self, graph, **kwargs):
        kwargs = self._check_train_parameters(type(graph), **kwargs)
        self.vectors = {}
        self.embeddings = None
        t1 = time()
        print("Start training...")
        self.build(graph, **kwargs)
        epochs = kwargs['epochs']
        if kwargs['debug_output_interval']:
            print("total iter: %i" % epochs)

        for i in range(epochs):
            time0 = time()
            self.embeddings = self.get_train(graph, step=i, **kwargs)
            if epochs > 1:
                for f_v in kwargs['validation_hooks']:
                    f_v(self, graph, step=i, **kwargs)
            if kwargs['debug_output_interval'] and i % kwargs['debug_output_interval'] == 0:
                print("epoch {}: {}, time used = {} s".format(i, self.debug_info, time()-time0))
            if self.early_stopping_judge(graph, step=i, **kwargs):
                print("Early stopping condition satisfied. Abort training.")
                break
        self.make_output(graph, **kwargs)
        if not self.vectors:
            self.get_vectors(graph)
        if self.save:
            print("Saving embeddings...")
            self.save_embeddings(os.path.join(self.outputpath, self.outputembeddingfile))
            print("Saving models...")
            self.save_model(os.path.join(self.outputpath, self.outputmodelfile))
        t2 = time()
        print("Finished training. Time used = {}.".format(t2 - t1))
        return self.vectors
