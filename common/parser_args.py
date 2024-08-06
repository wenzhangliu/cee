import argparse
import numpy as np
import torch
from .config import Config
import ast
from deepmerge import always_merger
import copy
from typing import Dict


def nice_print(d: Dict, indent=0):
    for key in d.keys():
        indent_str = ' ' * indent
        print(f'{indent_str}{key}:')
        if isinstance(d[key], Dict):
            nice_print(d[key], indent+4)
        else:
            print(f'{indent_str}  {d[key]}')


def get_config(name):
    parser = argparse.ArgumentParser()
    _, unparsed_args = parser.parse_known_args()
    unparsed_args, parsed_args = parse_args(parser, unparsed_args)

    print("Arguments to override:")
    nice_print(parsed_args)

    # Config
    config = Config(name)
    config.merge(parsed_args, override=True)

    if config.get('seed', False):
        config.seed = np.random.randint(2 ** 30 - 1)

    if torch.cuda.is_available():
        print(f'Using CUDA. Device: {config.device}')
    else:
        print('Using CPU, CUDA is not available on this machine.')

    return config


STRING_TO_VAL = {'nan': None, 'true': True, 'false': False}


def string_or_eval(val):
    if type(val) != str:
        return val

    if val.lower() in STRING_TO_VAL.keys():
        return STRING_TO_VAL[val.lower()]

    try:
        return ast.literal_eval(val)
    except:
        if '[' in val[0] or '(' in val[0]:
            return [string_or_eval(v) for v in val[1:-1].split(',')]
        else:
            return val


def create_nested_dict(d: dict, keys: list):
    if len(keys) > 1:
        d[keys[0]] = create_nested_dict(d={keys[1]: string_or_eval(d[keys[0]])},
                                        keys=keys[1:])
    return d


def parse_args(parser, unparsed_args):
    for arg in unparsed_args:
        if arg.startswith(("-", "--")):
            arg = arg.replace('=', ' ')
            arg = arg.replace(' true', '')
            arg = arg.replace('true', '')
            arg = arg.replace(' True', '')
            arg = arg.replace('True', '')
            parser.add_argument(arg, type=str)
    args = parser.parse_args()
    args_dict = args.__dict__
    return copy.deepcopy(args_dict), parse_nested_dict(args_dict)


def parse_nested_dict(args_dict):
    keys_to_remove = []
    added_dict = dict()
    for k, v in args_dict.items():
        if '.' in k:
            keys = k.split('.')
            dict_to_merge = create_nested_dict({keys[0]: v}, keys)
            added_dict = always_merger.merge(added_dict, dict_to_merge)
            keys_to_remove.append(k)
        else:
            args_dict[k] = string_or_eval(v)

    for k in keys_to_remove:
        del args_dict[k]

    args_dict.update(added_dict)

    return args_dict
