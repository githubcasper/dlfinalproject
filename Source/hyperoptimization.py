from Source.Call_Hyperopt import get_best_hyper
import torch
from torch import nn
from Source.Training import LSTMModel, VocabSizes
from Source.Dataloader import get_loaders
from torchtext.data.utils import get_tokenizer

def best_hyper(set_of_hyper):
    num_hidden_layers = set_of_hyper['num_hidden_layers']
    size_hidden_layer = set_of_hyper['size_hidden_layer']
    emsize = set_of_hyper['size_embed']
    dropout = set_of_hyper['dropout']
    batch_size = set_of_hyper['batch_size']

    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size,
                                                        test_split=0.1,
                                                        val_split=0.1,
                                                        shuffle_dataset=True,
                                                        random_seed=123)

    vocab_size = VocabSizes(train_loader, get_tokenizer('basic_english')).get_vocab_size()
    train_iter = iter(train_loader)
    num_class = len(set([label for (label, text) in train_iter]))

    model = LSTMModel(vocab_size, emsize, dropout, num_hidden_layers, size_hidden_layer, num_class)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(train_loader, model):
        model.train()

        for idx, (label, text) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

    def evaluate(val_loader):
        model.eval()

        with torch.no_grad():
            for idx, (label, text) in enumerate(val_loader):
                predicted_label = model(text)
                loss = criterion(predicted_label, label)
        return loss

    epochs = 10
    for epoch in range(epochs):
        train(iter(train_loader))

    val_loss = evaluate(val_loader)

    return val_loss


best_parameter = get_best_hyper(best_hyper)
print(best_parameter)
