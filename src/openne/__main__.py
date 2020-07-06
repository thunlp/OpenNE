from __future__ import print_function

import time
import ast

import numpy as np
import random
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from . import tasks, dataloaders, models

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

def addarg(arg, group, used_names, val, default=False, hlp=None, choices=None):
    kwargs = {}
    if arg not in used_names and legal_arg_name(arg):
        used_names.add(arg)
        if xtype(val) is bool and default:
            if val:
                if 'dest' not in kwargs:
                    kwargs['dest'] = arg
                arg = 'no_' + arg
                kwargs['action'] = 'store_false'
                if hlp is not None:
                    kwargs['help'] = hlp + " (action {}, dest={})".format(kwargs['action'], kwargs['dest'])
                else:
                    kwargs['help'] = "(action {}, dest={})".format(kwargs['action'], kwargs['dest'])
            else:
                kwargs['action'] = 'store_true'
                if hlp is not None:
                    kwargs['help'] = hlp + " (action {})".format(kwargs['action'])
                else:
                    kwargs['help'] = '(action {})'.format(kwargs['action'])

        else:
            if val is not None:
                kwargs['type'] = xtype(val)
            if default:
                kwargs['default'] = val
                if hlp is None:
                    kwargs['help'] = ' '
                    kwargs['help'] += '(default: {})'.format(val)
                else:
                    kwargs['help'] = '(default: {})'.format(val)
            if hlp is not None:
                kwargs['help'] = hlp
        if choices and arg in choices:
            kwargs['choices'] = choices[arg]
        group.add_argument(toargstr(arg), **kwargs)
        return True
    elif legal_arg_name(arg):
        return False
    return True


def parse_args():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            conflict_handler='resolve')

    # tasks, models, dataloaders
    parser.add_argument('--task', choices=tasks.taskdict.keys(), type=str.lower,
                        help='Assign a task. If unassigned, OpenNE will '
                             'automatically assign one according to the model.')
    parser.add_argument('--model', choices=models.modeldict.keys(), type=str.lower,
                        help='Assign a model.', required=True)
    datasetgroup = parser.add_mutually_exclusive_group(required=True)
    datasetgroup.add_argument('--dataset', choices=dataloaders.datasetdict.keys(), type=str.lower,
                              help='Assign a dataset as provided by OpenNE. '
                                   'Use --local-dataset if you want to load dataset from file.')

    # self-defined dataset
    local_inputs = parser.add_argument_group('LOCAL DATASET INPUTS')
    datasetgroup.add_argument('--local-dataset', action='store_true',
                              help='Load dataset from file. Check LOCAL DATASET INPUTS for more details.')
    local_inputs.add_argument('--root-dir', help='Root directory of input files. If empty, you should provide '
                                                 'absolute paths for graph files.',
                              default=None)
    local_input_format = local_inputs.add_mutually_exclusive_group()
    local_input_format.add_argument('--edgefile', help='Graph description in edgelist format.')
    local_input_format.add_argument('--adjfile', help='Graph description in adjlist format.')
    local_inputs.add_argument('--labelfile', help='Node labels.')
    local_inputs.add_argument('--features', help='Node features.')
    local_inputs.add_argument('--status', help="Dataset status.")
    local_inputs.add_argument('--name', help="Dataset name.", default='SelfDefined')
    local_inputs.add_argument('--weighted', action='store_true', help='View graph as weighted. (action store_true)')
    local_inputs.add_argument('--directed', action='store_true', help='View graph as directed. (action store_true)')

    used_names = set()
    choices = {'measurement': ('katz', 'cn', 'rpr', 'aa')}
    # structure & training args
    generalgroup = parser.add_argument_group("GENERAL MODEL ARGUMENTS")
    no_default_args = ['epochs', 'output',]
    addarg("clf_ratio", generalgroup, used_names, 0.5, True)
    generalgroup.add_argument('--validate', type=bool)
    model_args = models.ModelWithEmbeddings.args()
    for arg in model_args:
        addarg(arg, generalgroup, used_names, model_args[arg], arg not in no_default_args, choices=choices)

    simpledict = models.modeldict.copy()
    simpledict.__delitem__('node2vec')
    simpledict.__delitem__('deepwalk')
    simpledict.__delitem__('gf')
    simpledict.__delitem__('lap')
    simpledict['node2vec & deepwalk'] = models.Node2vec

    # add duplicate args as general model args
    general_names = used_names.copy()
    tmp_used_names = {}
    addarg('sparse', generalgroup, used_names, False, True, '(in lle, gcn)')
    for modelname in simpledict:
        model = simpledict[modelname]
        model_args = model.args()
        for arg in model_args:
            if arg in tmp_used_names:
                tmp_used_names[arg][0].append(modelname)
            else:
                tmp_used_names[arg] = ([modelname], model_args[arg])
    for arg, (modelnames, argval) in tmp_used_names.items():
        if len(modelnames) > 1:
            addarg(arg, generalgroup, used_names, argval, False, '(in {})'.format(', '.join(modelnames)), choices)

    for modelname in simpledict:
        model = simpledict[modelname]
        modelgroup = parser.add_argument_group(modelname.upper())
        shared_params = []
        model_args = model.args()
        for arg in model_args:
            if not addarg(arg, modelgroup, used_names, model_args[arg], True, None, choices=choices) \
                    and arg not in general_names:
                argval = model_args[arg]
                paramdescript = ' ' + toargstr(arg)
                if len(paramdescript) < 22:
                    paramdescript += ' ' * (22 - len(paramdescript))
                else:
                    paramdescript += '\n' + ' ' * 22
                paramdescript += '(default: {})'.format(argval)
                shared_params.append(paramdescript)
        if shared_params:
            modelgroup.description = 'Shared params:\n{}'.format(' \n'.join(shared_params))

    args = parser.parse_args()

    return args


def parse(**kwargs):
    if 'dataset' in kwargs:
        Dataset = dataloaders.datasetdict[kwargs['dataset']]
    else:
        name_dict = {k: v for k,v in kwargs.items() if k in ['edgefile', 'adjfile', 'labelfile', 'features', 'status']}
        Dataset = dataloaders.create_self_defined_dataset(kwargs['root_dir'], name_dict, kwargs['name'],
                                                          kwargs['weighted'], kwargs['directed'], 'features' in kwargs)
    Model = models.modeldict[kwargs['model']]
    taskname = kwargs.get('task', None)
    if taskname is None:
        if Model in tasks.supervisedmodels:
            Task = tasks.SupervisedNodeClassification
        else:
            Task = tasks.UnsupervisedNodeClassification
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
