from numbers import Number
import numpy as np
import collections
import os
import os.path as osp
import errno

def check_existance(src_dict, req_items, defaults=None):
    """
    Check if each required item is in the given dict.
    There are two ways to provide default values for items not present in src_dict:
    - provide a list, defaults, with the same length as req_item, for the backup option
    - provide req_items as a dict (recommended)
    :param src_dict: the given dict. note: this may be changed if default values are provided.
    :param req_items: list of required items, or a dict of required items with default values.
        [itm_0, itm_1, ...] or {itm_0: default_v0, itm_1: default_v1, ...}
    :param defaults: default value if req_item not in src_dict.
        None or [default_v0, default_v1, ...]
    :raise: KeyError when default value of some absent item in src_dict is not set.
    """
    if isinstance(req_items, dict):
        for k, v in req_items.items():
            if k not in src_dict:
                src_dict[k] = v
    else:
        for i in range(len(req_items)):
            if req_items[i] not in src_dict:
                if defaults is not None and len(defaults) > i:
                    src_dict[req_items[i]] = defaults[i]
                else:
                    raise KeyError('Item "{}" not in src_dict'.format(i))

lop = lambda x, boundary, closed: x > boundary or (closed and x == boundary)
rop = lambda x, boundary, closed: x < boundary or (closed and x == boundary)
interval_value_constant = {'open': (0, 0), 'closed': (1, 1), 'open-closed': (0, 1), 'closed-open': (1, 0)}
def in_interval(x, a, b, description='closed'):
    l, r = interval_value_constant[description]
    return isinstance(x, Number) and lop(x, a, l) and rop(x, b, r)

def check_range(src_dict, req_ranges):
    """
    Check if each given item in src_dict meets the required range.
    :param src_dict: the given dict.
    :param req_ranges: the dict of required range.
        {itm_0: required_range0, itm_1: required_range1, ...}
        format for ranges:
        Not appearing for no restriction
        range, arange, list, dict and other iterables
        (a,b): closed interval
        (a,b, 'open' or 'closed' or 'open-closed' or 'closed-open'): interval
        'positive': (0, np.inf)
    :raise: ValueError.
    """

    for i in src_dict:
        if i in req_ranges:
            r = req_ranges[i]
            if r == 'positive':
                r = (0, np.inf)
            if isinstance(r, tuple) and len(r) in (2, 3):
                if not in_interval(src_dict[i], *r):
                    raise ValueError('arg value {} not in interval {}'.format(src_dict[i], r))
            elif src_dict[i] not in r:
                raise ValueError('arg value {} not in {}'.format(src_dict[i], r))

def to_list(x):
    if x is None:
        return []
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x

def split_to_int(l):
    return [int(i) for i in l.split()]

def files_exist(files):
    return all([osp.exists(f) for f in files])


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e