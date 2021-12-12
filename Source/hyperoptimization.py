from Call_Hyperopt import get_best_hyper
import torch
from torch import nn
from RNNModel import LSTMModel, VocabSizes
from Dataloader import get_loaders
from torchtext.data.utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer('basic_english')
vocab_sizes = VocabSizes(tokenizer)
vocab_size, vocab_text = vocab_sizes.get_vocab_size_text()
amount_of_categories = len(vocab_sizes.get_label_dict())
max_length = vocab_sizes.get_max_len()

train_loader, val_loader, test_loader = get_loaders(batch_size=300,
                                                    test_split=0.025,
                                                    val_split=0.025,
                                                    shuffle_dataset=True,
                                                    random_seed=123)
#%%

def best_hyper(set_of_hyper):
    print(set_of_hyper)
    num_hidden_layers = int(set_of_hyper['num_hidden_layers'])
    size_hidden_layer = int(set_of_hyper['size_hidden_layer'])
    emsize = int(set_of_hyper['size_embed'])
    dropout = float(set_of_hyper['dropout'])
    lr = float(set_of_hyper['LR'])

    model = LSTMModel(vocab_size, emsize, dropout, num_hidden_layers, size_hidden_layer, max_length, amount_of_categories).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(train_loader, model):
        model.train()
        for batch_idx, (label, text) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print('Batch index: {:4d}/{:4d}'.format(batch_idx, len(train_loader)))

            predicted_label = model(text, text.size(0))
            predicted_label = predicted_label.squeeze() if predicted_label.size(0) > 1 else predicted_label.squeeze().unsqueeze(0)
            loss = criterion(predicted_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate(val_loader):
        model.eval()

        with torch.no_grad():
            for idx, (labels, texts) in enumerate(val_loader):
                predicted_labels = model(texts, texts.size(0))
                loss = criterion(predicted_labels.squeeze(), labels)
                
        return loss
    
    # training session
    epochs = 2
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        train(iter(train_loader), model)
    
    # validation session
    val_loss = evaluate(iter(val_loader))
    print("Validation loss:", val_loss.item())
    return val_loss.item()


best_parameter_ = get_best_hyper(best_hyper)
print(best_parameter_)

