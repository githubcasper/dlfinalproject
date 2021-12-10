from hyperopt import hp, fmin, tpe
from collections import OrderedDict


def get_best_hyper(objective):
    space = OrderedDict([('num_hidden_layers', hp.randint('num_hidden_layers', 1, 5)),
                        ('size_hidden_layer', hp.randint('size_hidden_layer', 20, 40)),
                        ('size_embed', hp.randint('size_embed', 30, 60)),
                        ('dropout', hp.uniform('dropout', 0, 0.5)),
                        ('batch_size', hp.randint('batch_size', 300, 400))])

    best = fmin(objective, space, algo=tpe.suggest, max_evals=10)

    return best
