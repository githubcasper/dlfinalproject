from hyperopt import hp, fmin, tpe
#from collections import OrderedDict
#import hyperopt.pyll.stochastic

def get_best_hyper(objective):
    space = {'num_hidden_layers': hp.randint('num_hidden_layers', 1, 8),
             'size_hidden_layer': hp.randint('size_hidden_layer', 20, 60),
             'size_embed':        hp.randint('size_embed', 40, 80),
             'dropout':           hp.uniform('dropout', 0, 0.5),
             'batch_size':        hp.randint('batch_size', 300, 500),
             'LR':                hp.choice('LR', (0.01, 0.001))}

    best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

    return best

'''
-------------------------------------------------------------------
Det bør ikke gøre nogen forskel, men her er som det stod før:
-------------------------------------------------------------------
    
space = OrderedDict([('num_hidden_layers', hp.randint('num_hidden_layers', 1, 8)),
                    ('size_hidden_layer', hp.randint('size_hidden_layer', 20, 60)),
                    ('size_embed', hp.randint('size_embed', 40, 80)),
                    ('dropout', hp.uniform('dropout', 0, 0.5)),
                    ('batch_size', hp.randint('batch_size', 300, 500)),
                    ('LR', hp.choice('LR', (0.01, 0.001)))])

'''