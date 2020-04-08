import numpy as np
import torch
import math


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(
        shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = math.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.rand(shape)*(2*init_range)-init_range
    return torch.nn.Parameter(initial)


def zeros(shape):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial)


def ones(shape):
    """All ones."""
    initial = torch.ones(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial)
