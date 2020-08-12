from .inits import *
from .utils import *
import torch
# import tensorflow as tf

#flags = tf.app.flags
#FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, drop_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = drop_prob
    random_tensor += torch.rand(noise_shape)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x.indices()[:, dropout_mask]
    preout = torch.sparse_coo_tensor(i, values=x.values()[dropout_mask], size=x.shape, dtype=torch.float32)
    return preout * (1./drop_prob)


class Layer(torch.nn.Module):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        super(Layer, self).__init__()
        allowed_kwargs = {'name', 'logging'} # logging: added later
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.sparse_inputs = False
        self.logging = False

    def forward(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        outputs = super(Layer, self).__call__(inputs, **kwargs)
        if self.logging:
            pass
        return outputs

    def _log_vars(self):
        for var in self.parameters():
            pass


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, support, dropout=0., num_features_nonzero=0.,
                 sparse_inputs=False, act=torch.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout  # note we modified the API
        self.act = act
        self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.output_dim = output_dim
        self.input_dim = input_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero

        for i in range(len(self.support)):
            setattr(self, 'weights_' + str(i),  glorot([input_dim, output_dim]))
            if bias:
                self.bias = zeros([output_dim])
            else:
                self.bias = None

        if self.logging:
            self._log_vars()

    def forward(self, inputs=None):
        x = inputs
        print("inputs: ", x.device)
        if not self.featureless and self.training:
            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
            else:
                x = torch.dropout(x, self.dropout,True)
        elif self.sparse_inputs:
            x = tuple_to_sparse(x)
        # convolve
        output = torch.zeros([self.support[0].size()[0], self.output_dim], device=x.device)
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = torch.mm(x, getattr(self, 'weights_' + str(i)))
                print("presup: ", pre_sup.device)
            else:
                pre_sup = getattr(self, 'weights_' + str(i))
            support = torch.mm(self.support[i], pre_sup)
            output += support

        # bias
        if self.bias is not None:
            output += self.bias
        return self.act(output)
