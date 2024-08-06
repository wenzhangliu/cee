from decimal import Decimal
import numpy as np


def format_string(s):
    s = np.reshape(s, -1).tolist()
    x = [('%.2E' % Decimal(x)) for x in s]
    return ', '.join(x)


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))