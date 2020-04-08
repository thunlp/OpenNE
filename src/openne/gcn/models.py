from .layers import *
from .metrics import *

#flags = tf.app.flags
#FLAGS = flags.FLAGS


class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        allowed_kwargs = {'name', 'logging'} #
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.layers = []
        self.sequential = None

        self.input = None
        self.output = None

        self._loss = 0
        self._accuracy = 0
        self.optimizer = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        self._build()

        # Build sequential layer model
        self.sequential = torch.nn.Sequential(*self.layers)
        self._optim()

    def forward(self, inputs):
        self.input = inputs
        self.output = self.sequential(inputs)
        return self.output

    def loss(self, *args):
        raise NotImplementedError

    def accuracy(self, *args):
        raise NotImplementedError

    def _optim(self):
        raise NotImplementedError

    def save(self):
        save_path = "tmp/%s.ckpt" % self.name
        torch.save(self.state_dict(), save_path)
        print("Model saved in file: %s" % save_path)

    def load(self):
        save_path = "tmp/%s.ckpt" % self.name
        state_dict = torch.load(save_path)
        self.load_state_dict(state_dict)
        print("Model restored from file: %s" % save_path)

# ignore this class
"""
class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
"""

class GCN(Model):
    def __init__(self, input_dim, output_dim, hidden_dims, supports, dropout=0., weight_decay=0.,num_features_nonzero=0, logging=False, sparse_inputs=False, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.dimensions = [input_dim]+hidden_dims+[output_dim]
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.supports = supports
        self.dropout = dropout
        self.logging = logging
        self.sparse_inputs = sparse_inputs
        self.num_features_nonzero=num_features_nonzero
        self.build()

    def loss(self, labels, labels_mask):
        self._loss = 0
        for i in self.parameters():
            self._loss += self.weight_decay * torch.norm(i, 2)
        self._loss += masked_softmax_cross_entropy(self.output, labels, labels_mask)
        return self._loss

    def accuracy(self, labels, labels_mask):
        self._accuracy = masked_accuracy(self.output, labels,
                                        labels_mask)
        return self._accuracy

    def _build(self):
        for i in range(1,len(self.dimensions)-1):
            self.layers.append(GraphConvolution(input_dim=self.dimensions[i-1],
                                                output_dim=self.dimensions[i],
                                                act=torch.relu,
                                                support=self.supports,
                                                dropout=self.dropout,
                                                sparse_inputs=self.sparse_inputs,
                                                num_features_nonzero=self.num_features_nonzero,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.dimensions[-2],
                                            output_dim=self.dimensions[-1],
                                            act=lambda x:x,
                                            support=self.supports,
                                            dropout=self.dropout,
                                            sparse_inputs=self.sparse_inputs,
                                            num_features_nonzero=self.num_features_nonzero,
                                            logging=self.logging))
        self.layers.append(torch.nn.Softmax())

    def _optim(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
