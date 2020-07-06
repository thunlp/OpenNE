from __future__ import print_function

import time
import ast

import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression

from . import tasks, datasets, models


def ListInput(s: str):
    l = list(ast.literal_eval(s))
    if type(l) is not list:
        raise TypeError

def xtype(val):
    if type(val) is str:
        return str.lower
    if type(val) is list:
        return ListInput
    return type(val)

def toargstr(s):
    if s[:2] != '--':
        s = '--' + s
    s = s.replace('_', '-')
    return s

def legal_arg_name(arg):
    if arg[0] == '_':
        return False
    return True

def addarg(arg, group, used_names, val):
    if arg not in used_names and legal_arg_name(arg):
        used_names.add(arg)
        if xtype(val) is bool:
            group.add_argument(toargstr(arg), action="store_true")
        else:
            group.add_argument(toargstr(arg), type=xtype(val))

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    # tasks, models, datasets
    parser.add_argument('--task', choices=tasks.taskdict.keys(), type=str.lower,
                        help='Assign a task.\n If unassigned, OpenNE will '
                             'automatically assign one according to the model.')
    parser.add_argument('--model', choices=models.modeldict.keys(), type=str.lower,
                        help='Assign a model.', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', choices=datasets.datasetdict.keys(), type=str.lower,
                       help='Assign a dataset as provided by OpenNE.\n'
                            'Use --local-dataset if you want to load dataset from file.')

    # self-defined dataset
    local_inputs = parser.add_argument_group('LOCAL DATASET INPUTS')
    group.add_argument('--local-dataset', action='store_true',
                       help='Load dataset from file. Check LOCAL DATASET INPUTS for more details.')
    local_inputs.add_argument('--root-dir', help='Root directory of input files. If empty, you should provide'
                                                 'absolute paths for graph files.',
                              default=None)
    local_input_format = local_inputs.add_mutually_exclusive_group()
    local_input_format.add_argument('--edgefile', help='Graph description in edgelist format.')
    local_input_format.add_argument('--adjfile', help='Graph description in adjlist format.')
    local_inputs.add_argument('--labelfile', help='Node labels.')
    local_inputs.add_argument('--features', help='Node features.')
    local_inputs.add_argument('--status', help="Dataset status.")
    local_inputs.add_argument('--name', help="Dataset name.", default='SelfDefined')
    local_inputs.add_argument('--weighted', action='store_true', help='View graph as weighted.')
    local_inputs.add_argument('--directed', action='store_true', help='View graph as directed.')

    used_names = set()
    # structure & training args
    group = parser.add_argument_group("GENERAL MODEL ARGUMENTS")
    addarg("clf_ratio", group, used_names, 0.1)
    group.add_argument('--validate', type=bool)
    model_args = models.ModelWithEmbeddings.args()
    for arg in model_args:
        addarg(arg, group, used_names, model_args[arg])

    for modelname in models.modeldict:
        model = models.modeldict[modelname]
        group = parser.add_argument_group(modelname.upper())
        model_args = model.args()
        for arg in model_args:
            addarg(arg, group, used_names, model_args[arg])

    args = parser.parse_args()

    return args


def parse(**kwargs):
    if 'dataset' in kwargs:
        Dataset = datasets.datasetdict[kwargs['dataset']]
    else:
        name_dict = {k: v for k,v in kwargs.items() if k in ['edgefile', 'adjfile', 'labelfile', 'features', 'status']}
        Dataset = datasets.create_self_defined_dataset(kwargs['root_dir'], name_dict, kwargs['name'],
                                                       kwargs['weighted'], kwargs['directed'], 'features' in kwargs)
    Model = models.modeldict[kwargs['model']]
    taskname = kwargs.get('task', None)
    if taskname is None:
        if Model in tasks.supervisedmodels:
            Task = tasks.SupervisedNodePrediction
        else:
            Task = tasks.UnsupervisedNodePrediction
    else:
        Task = tasks.taskdict[taskname]
    return Task, Dataset, Model


def main(args):

    # parsing
    args = {x: y for x, y in args.__dict__.items() if y is not None}

    print("actual args:",args)

    Task, Dataset, Model = parse(**args)  # parse required Task, Dataset, Model (classes)
    dellist = ['dataset', 'edgefile', 'adjfile', 'labelfile', 'features',
               'status', 'weighted', 'directed', 'root_dir', 'task', 'model']
    for item in dellist:
        if item in args:
            args.__delitem__(item)
    # preparation
    task = Task(**args)                 # prepare task
    task.check(Model, Dataset)          # check parameters
    args = task.kwargs
    model = Model(**args)               # prepare model
    dataset = Dataset()                 # prepare dataset

    res = task.train(model, dataset)    # train

    # evaluation
    task.evaluate(model, res, dataset)  # evaluate


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
