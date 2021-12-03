from Source.Call_Hyperopt import get_best_hyper


def best_hyper(set_of_hyper):
    num_hidden_layers = set_of_hyper['num_hidden_layers']
    size_hidden_layer = set_of_hyper['size_hidden_layer']
    dropout = set_of_hyper['dropout']
    batch_size = set_of_hyper['batch_size']

    val_loss = num_hidden_layers/size_hidden_layer*dropout + batch_size

    return val_loss


best_parameter = get_best_hyper(best_hyper)
print(best_parameter)
