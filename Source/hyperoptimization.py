from Call_Hyperopt import get_best_hyper
import torch
from torch import nn
from RNNModel import LSTMModel, VocabSizes
from Dataloader import get_loaders
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
vocab_sizes = VocabSizes(tokenizer)
vocab_size, vocab_text = vocab_sizes.get_vocab_size_text()
vocab_label = vocab_sizes.get_label_dict()
vocab_int_to_label = vocab_sizes.get_int_to_label_dict()
max_length = vocab_sizes.get_max_len()
text_pipeline = lambda x: vocab_text(tokenizer(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
##########
##########
##########
##########
##########
##########                        BESKED TIL CASPER:
##########  Jeg har midlertidigt smidt breaks ind i nedenstående kode for at 
##########  tage et shortcut og teste kaldet til hyperopt. Jeg får fejlen 
##########  "TypeError: iteration over a 0-d tensor", når den er færdig med 
##########  første omgang af train og val.
########## 
##########
##########
##########
##########
#%%




def best_hyper(set_of_hyper):
    print(set_of_hyper)
    num_hidden_layers = int(set_of_hyper['num_hidden_layers'])
    size_hidden_layer = int(set_of_hyper['size_hidden_layer'])
    emsize = int(set_of_hyper['size_embed'])
    dropout = float(set_of_hyper['dropout'])
    batch_size = int(set_of_hyper['batch_size'])
    lr = float(set_of_hyper['LR'])

    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size,
                                                        test_split=0.1,
                                                        val_split=0.1,
                                                        shuffle_dataset=True,
                                                        random_seed=123)

    model = LSTMModel(vocab_size, emsize, dropout, num_hidden_layers, size_hidden_layer, max_length).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(train_loader, model):
        model.train()

        for batch_idx, (label, text) in enumerate(train_loader):
            if batch_idx % 5 == 0:
                print(f'Batch index: {batch_idx}')
            if batch_idx == 1:
                break
            optimizer.zero_grad()
            loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float32, device=device))
            for i in range(len(text)):
                input_list = text_pipeline(text[i])
                while len(input_list) < max_length:
                    input_list.append(text_pipeline('<pad>')[0])
                input_tensor = torch.tensor(input_list, dtype=torch.int64, device=device)
                predicted_label = model(input_tensor)

                target_label = torch.tensor(vocab_label[label[i]], dtype=torch.int64, device=device).unsqueeze(0)

                loss += criterion(predicted_label, target_label)

            loss.backward()
            optimizer.step()

    def evaluate(val_loader):
        model.eval()

        with torch.no_grad():
            loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float32, device=device))
            for idx, (label, text) in enumerate(val_loader):
                if idx == 250:
                    break
                input_list = text_pipeline(text[0])
                while len(input_list) < max_length:
                    input_list.append(text_pipeline('<pad>')[0])
                input_tensor = torch.tensor(input_list, dtype=torch.int64, device=device)
                predicted_label = model(input_tensor)

                target_label = torch.tensor(vocab_label[label[0]], dtype=torch.int64, device=device).unsqueeze(0)

                loss += criterion(predicted_label, target_label)
        return loss

    epochs = 1
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        train(iter(train_loader), model)

    val_loss = evaluate(val_loader)

    return val_loss


best_parameter = get_best_hyper(best_hyper)
print(best_parameter)
