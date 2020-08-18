from typing import Any

import torch
import os
from time import time
from ..utils import *
import inspect
import numpy

class ModelWithEmbeddings(torch.nn.Module):
    def __init__(self, *, output=None, save=True, **kwargs):
        super(ModelWithEmbeddings, self).__init__()
        self.vectors = {}
        self.embeddings = None
        self.debug_info = None
        if save:
            if output is None:
                self.outputpath = os.path.join(osp.dirname(osp.realpath(__file__)),
                                               '..', '..', '..', 'results', type(self).__name__)
                self.outputmodelfile = type(self).__name__ + '_model.txt'
                self.outputembeddingfile = type(self).__name__ + '_embeddings.txt'
            else:
                self.outputpath = os.path.dirname(output)
                if self.outputpath == '':
                    self.outputpath = '.'
                self.outputmodelfile = type(self).__name__ + 'models.txt'
                self.outputembeddingfile = os.path.basename(output)
            kwargs['outputpath'] = self.outputpath
            kwargs['outputmodelfile'] = self.outputmodelfile
            kwargs['outputembeddingfile'] = self.outputembeddingfile
            kwargs['save'] = save
            self.outputpath = osp.abspath(self.outputpath)
            print("output path = ", self.outputpath)
            if not os.path.isdir(self.outputpath):
                makedirs(self.outputpath)
            try:
                with open(os.path.join(self.outputpath, self.outputembeddingfile), 'a'):
                    pass
            except Exception as e:
                raise FileNotFoundError('Failed to open target embedding file "{}": {}. '.format(
                    os.path.join(self.outputpath, self.outputembeddingfile), str(e)))
            try:
                with open(os.path.join(self.outputpath, self.outputmodelfile), 'a'):
                    pass
            except Exception as e:
                raise FileNotFoundError('Failed to open target models file "{}": {}. '.format(
                    os.path.join(self.outputpath, self.outputmodelfile), str(e)))

        else:
            kwargs['save'] = save

        for i in kwargs:
            self.__setattr__(i, kwargs[i])

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def save_embeddings(self, filename):
        with open(filename, 'w') as fout:
            node_num = len(self.vectors)
            fout.write("{} {}\n".format(node_num, self.dim))
            for node, vec in self.vectors.items():
                fout.write("{} {}\n".format(node, ' '.join([str(float(x)) for x in vec])))

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.outputpath, self.outputmodelfile)
        if not os.path.isfile(path):
            raise FileNotFoundError("Model file not found.")
        self.load_state_dict(torch.load(path))
        # specials
        for name, buffers in self.named_buffers(prefix='__from_array_'):
            self.__setattr__(name.strip('__from_array_'), buffers.numpy())
        for name, buffers in self.named_buffers(prefix='__from_sparse_'):
            self.__setattr__(name.strip('__from_sparse_'), buffers.to_sparse())

    @classmethod
    def check_train_parameters(cls, **kwargs):
        return kwargs

    @classmethod
    def check_graphtype(cls, graphtype, **kwargs):
        pass

    @classmethod
    def check(cls, graphtype=None, **kwargs):
        new_kwargs = kwargs.copy()
        ret_args = cls.check_train_parameters(**new_kwargs)
        new_kwargs = ret_args
        if graphtype:
            cls.check_graphtype(graphtype, **new_kwargs)
        if 'epochs' not in new_kwargs:
            _multiple_epochs = False
        else:
            _multiple_epochs = True
        check_existance(new_kwargs, {'dim': 128,
                                     'epochs': 1,
                                     '_validation_hooks': [],
                                     'validation_interval': 5,
                                     'debug_output_interval': 5,
                                     '_multiple_epochs': _multiple_epochs,
                                     'output': None,
                                     'save': True,})
        if graphtype:
            if not torch.cuda.is_available() or new_kwargs['cpu']:
                new_kwargs['data_parallel'] = False
                new_kwargs['_device'] = torch.device('cpu')
            else:
                new_kwargs['_device'] = torch.device('cuda', new_kwargs['devices'][0])
        return new_kwargs

    def build(self, graph, **kwargs):
        pass

    def train_model(self, graph, **kwargs):
        """
            returns: embeddings or None
            In case of None, you NEED to rewrite make_output() and assign self.embeddings in it
        """
        raise NotImplementedError

    def early_stopping_judge(self, graph, **kwargs):
        return False

    def _get_vectors(self, graph):
        """
            Get self.vectors (which is a dict in format {node: embedding}) from self.embeddings.
            This should only be called in self.make_output().

            Rewrite when self.embeddings is not used and self.vectors is not acquired in self.train_model.
        """
        embs = self.embeddings
        if embs is None:
            return self.vectors
        self.vectors = {}
        for i, embedding in enumerate(embs):
            self.vectors[graph.look_back_list[i]] = embedding
        return self.vectors

    def _get_embeddings(self, graph, **kwargs):
        """
            Generates self.embeddings. This should only be called in self.make_output().

            The default process applies to most situation except:
              1. self.train_model() does not return a thing
              2. the model does not use self.embeddings and requires multiple training epochs
            In these cases rewrite _get_embeddings.
            For case 1, you need to update self.embeddings each time in _get_embeddings().
            See examples from LINE and SDNE.
            For case 2, do nothing in _get_embeddings() (write a simple `return`).
            For models with single epoch, you do not need to rewrite if you are not
            calling this function outside self.make_output() (e.g. Node2Vec).

            You can also rewrite for models with multiple epochs when you need to
            get embeddings without training. See examples from GCN and GAE.

            N.B. it is recommended that self.embeddings is acquired only after a training step,
            i.e. not appearing as rhs oprand in training calculations. If possible, try not
            involve self.embeddings in the body of self.train_model().

        """
        if self.embeddings is None and kwargs.get('_multiple_epochs', True):
            self.embeddings = self.train_model(graph, **kwargs)


    def make_output(self, graph, **kwargs):
        """
            Generates self.embeddings and self.vectors.
            called on two occasions:
              1. after training. Rewrite in case you need to do anything.
              2. EVERY TIME before you need self.embeddings or self.vectors,
                e.g. on validation of unsupervised_node_classification

        """
        self._get_embeddings(graph, **kwargs)
        if self.embeddings is not None:
            self.embeddings = self.embeddings.detach().to("cpu")
        self._get_vectors(graph)

    #  TODO: a set of rules of special buffers.

    def register_numpy(self, name, array):
        self.register_buffer("__from_array_" + name, torch.from_numpy(array))

    def register_buffer(self, name, tensor, *args):
        if torch.__version__ < '1.5.0' and tensor.is_sparse:  # will save a dense version
            self.__setattr__(name, tensor)
            super(ModelWithEmbeddings, self).register_buffer("__from_sparse_" + name, tensor.to_dense(), *args)
        else:
            super(ModelWithEmbeddings, self).register_buffer(name, tensor, *args)

    def register_float_buffer(self, name, *tensor_info):
        self.register_buffer(name, torch.tensor(*tensor_info, dtype=torch.float32))

    def adjmat_device(self, graph, weighted, directed):
        self.adj_mat = torch.from_numpy(graph.adjmat(weighted, directed)).type(torch.float32)
        self.register_buffer('adj_mat', self.adj_mat)
        return self.adj_mat

    def forward(self, graph, **kwargs):
        kwargs = self.check(type(graph), **kwargs)
        self.vectors = {}
        self.embeddings = None
        t1 = time()
        print("Start training...")
        self.build(graph, **kwargs)
        self.to(self._device)
        # print([(i, v.shape) for i, v in self.named_parameters(recurse=True)])
        # print([i for i in self.named_modules()])

        if kwargs['_multiple_epochs']:
            epochs = kwargs['epochs']
            print("total iter: %i" % epochs)
        else:
            epochs = 1
        time0 = time()
        for i in range(epochs):
            self.embeddings = self.train_model(graph, step=i, **kwargs)
            if kwargs['_multiple_epochs'] and (i + 1) % kwargs['validation_interval'] == 0:
                for f_v in kwargs['_validation_hooks']:
                    f_v(self, graph, step=i, **kwargs)
            if kwargs['_multiple_epochs'] and (i + 1) % kwargs['debug_output_interval'] == 0:
                if self.debug_info:
                    self.debug_info += '; '
                else:
                    self.debug_info = ''
                print("epoch {}: {}time used = {}s".format(i + 1, self.debug_info, time() - time0))
                time0 = time()
            elif not kwargs['_multiple_epochs']:
                if self.debug_info:
                    self.debug_info += '\n'
                else:
                    self.debug_info = ''
                print("{}Time used = {}s".format(self.debug_info, time() - time0))
                time0 = time()
            if self.early_stopping_judge(graph, step=i, **kwargs):
                print("Early stopping condition satisfied. Abort training.")
                break
        self.make_output(graph, **kwargs)

        t2 = time()
        print("Finished training. Time used = {}.".format(t2 - t1))
        if self.save:
            embeddingpath = osp.abspath(osp.join(self.outputpath, self.outputembeddingfile))
            print("Saving embeddings to {}...".format(embeddingpath))
            self.save_embeddings(embeddingpath)
            modelpath = osp.abspath(osp.join(self.outputpath, self.outputmodelfile))
            print("Saving model to {}...".format(modelpath))
            self.save_model(modelpath)
        return self.vectors

    @classmethod
    def args(cls):
        return cls.check()

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
