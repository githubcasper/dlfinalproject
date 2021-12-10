from Call_Hyperopt import get_best_hyper
import torch
from torch import nn
from RNNModel import LSTMModel, VocabSizes
from Dataloader import get_loaders
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F


def best_hyper(set_of_hyper):
    num_hidden_layers = int(set_of_hyper['num_hidden_layers'])
    size_hidden_layer = int(set_of_hyper['size_hidden_layer'])
    emsize = int(set_of_hyper['size_embed'])
    dropout = float(set_of_hyper['dropout'])
    batch_size = int(set_of_hyper['batch_size'])

    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size,
                                                        test_split=0.1,
                                                        val_split=0.1,
                                                        shuffle_dataset=True,
                                                        random_seed=123)

    tokenizer = get_tokenizer('basic_english')
    vocab_sizes = VocabSizes(tokenizer)
    vocab_size, vocab_text = vocab_sizes.get_vocab_size_text()
    vocab_label = vocab_sizes.get_label_dict()
    vocab_int_to_label = vocab_sizes.get_int_to_label_dict()
    max_length = vocab_sizes.get_max_len()
    print(max_length)
    text_pipeline = lambda x: vocab_text(tokenizer(x))

    model = LSTMModel(vocab_size, emsize, dropout, num_hidden_layers, size_hidden_layer, max_length)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(train_loader, model):
        model.train()

        for idx, (label, text) in enumerate(train_loader):
            optimizer.zero_grad()
            for i in range(len(text)):
                input_list = text_pipeline(text[i])
                while len(input_list) < max_length:
                    input_list.append(text_pipeline('<pad>')[0])
                input_tensor = torch.tensor(input_list, dtype=torch.int64)
                predicted_label = model(input_tensor)
                one_hot_label = F.one_hot(torch.tensor(vocab_label[label[i]],
                                                       dtype=torch.int64), num_classes=40).unsqueeze(0)
                loss = criterion(predicted_label, one_hot_label.type(torch.FloatTensor))
                loss.backward()
            optimizer.step()

    def evaluate(val_loader):
        model.eval()

        with torch.no_grad():
            for idx, (label, text) in enumerate(val_loader):
                loss = 0
                for i in range(len(text)):
                    predicted_label = model(torch.tensor(text_pipeline(text[i]), dtype=torch.int64))
                    one_hot_label = F.one_hot(torch.tensor(vocab_label[label[i]], dtype=torch.int64), num_classes=40).unsqueeze(0)
                    loss += criterion(predicted_label, one_hot_label.type(torch.FloatTensor))
        return loss

    epochs = 10
    for epoch in range(epochs):
        train(iter(train_loader), model)

    val_loss = evaluate(val_loader)

    return val_loss


best_parameter = get_best_hyper(best_hyper)
print(best_parameter)
