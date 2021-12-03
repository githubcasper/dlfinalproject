from hyperopt import hp, fmin, tpe, space_eval
import numpy as np
from collections import OrderedDict


def get_best_hyper(objective):
    space = OrderedDict([('num_hidden_layers', hp.randint('num_hidden_layers', 1, 5)),
                        ('size_hidden_layer', hp.randint('size_hidden_layer', 10, 30)),
                        ('dropout', hp.uniform('dropout', 0, 0.5)),
                        ('batch_size', hp.randint('batch_size', 50, 150))])

    best = fmin(objective, space, algo=tpe.suggest, max_evals=15)

    return best
