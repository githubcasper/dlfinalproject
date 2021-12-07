# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:07:56 2021

@author: Andreas Tind
"""

from Source.Dataloader import get_loaders
#import neptune.new as neptune
#import keyring+
#import os
#from pathlib import Path as PL

import time
import torch
#from torchtext.datasets import AG_NEWS
#train_iter = AG_NEWS(split='train')
from torch import nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

#%% Neptune

'''
secret_api = keyring.get_password('Neptune', "andreastind")

run = neptune.init(project="andreastind/DeepLearningFinalProject",
                   api_token=secret_api)

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66
run.stop()
'''

#%% Training

#json_path = "..\\Data\\News_Category_Dataset_v2.json"

train_loader, val_loader, test_loader = get_loaders(batch_size=2, 
                                                    test_split=0.1, 
                                                    val_split=0.1, 
                                                    shuffle_dataset=True, 
                                                    random_seed=123)

#train_iter = iter(train_loader)
#next(train_iter)


#%% Most recent chunk

class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, dropout, num_hidden_layers, size_hidden_layer, num_classes=40):
        super(LSTMModel, self).__init__()
        self.size_embed = embed_dim
        self.size_hidden_layer = size_hidden_layer
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout

        self.embedding = nn.Embedding(vocab_size, self.size_embed, sparse=True)

        self.rnn = nn.LSTM(input_size=self.size_embed,
                           hidden_size=self.size_hidden_layer,
                           num_layers=self.num_hidden_layers,
                           bidirectional=True)

        self.out = nn.Linear(self.size_hidden_layer, num_classes)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.rnn.weight.data.uniform_(-initrange, initrange)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()

    def forward(self, text, hidden):
        embedded = self.embedding(torch.tensor(text))
        embedded = self.dropout(embedded)
        output = self.rnn(embedded, hidden)
        output = self.out(output)
        return nn.Softmax(output)

    def initHidden(self):
        return torch.zeros(1, 1, self.size_embed, device=device)


train_loader, val_loader, test_loader = get_loaders(batch_size=1, 
                                                    test_split=0.1, 
                                                    val_split=0.1, 
                                                    shuffle_dataset=True, 
                                                    random_seed=123)


# TEXT VOCAB GENERATION
class VocabSizes():
    def __init__(self, data_loader, tokenizer):
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    def yield_tokens_text(self, data_iter):
        for label, text in data_iter:
            yield self.tokenizer(text[0])

    def get_vocab_size(self):
        train_iter = iter(self.data_loader)
        vocab_text = build_vocab_from_iterator(self.yield_tokens_text(train_iter), specials=["<unk>"])
        vocab_text.set_default_index(vocab_text["<unk>"])
        return len(vocab_text)


def yield_tokens_text(data_iter):
    for label, text in data_iter:
        yield tokenizer(text[0])

train_iter = iter(train_loader)
vocab_text = build_vocab_from_iterator(yield_tokens_text(train_iter), specials=["<unk>"])
vocab_text.set_default_index(vocab_text["<unk>"])
text_pipeline = lambda x: vocab_text(tokenizer(x))


# LABEL VOCAB GENERATION
def yield_tokens_label(data_iter):
    for label, text in data_iter:
        yield tokenizer(label[0])

train_iter = iter(train_loader)
vocab_label = build_vocab_from_iterator(yield_tokens_label(train_iter), specials=["<unk>"])
vocab_label.set_default_index(vocab_label["<unk>"])
label_pipeline = lambda x: vocab_label(tokenizer(x))


vocab_label_size = len(vocab_label)
train_iter = iter(train_loader)
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab_text)
emsize = 64
model = LSTMModel(vocab_size, emsize, num_class).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=1,
                                                       patience=15)


def train(train_loader, model):
    model.train()

    for idx, (label, text) in enumerate(train_loader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        #        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()


def evaluate(val_loader, model):
    model.eval()

    with torch.no_grad():
        for idx, (label, text) in enumerate(val_loader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
    return loss


epochs = 10
for epoch in range(epochs):
    train(train_loader)

val_loss = evaluate(val_loader)

################ COPIED FROM THE INTERNET ############################
#%%

from torchtext.datasets import AG_NEWS
train_iter = AG_NEWS(split='train')
train_iter
#%%


tokenizer = get_tokenizer('basic_english')
train_iter = iter(train_loader)
train_iter

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text[0])

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = text_pipeline

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device)#, offsets.to(device)

train_iter = iter(train_loader)
#dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)




def train_(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate_(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


#%% 

train_iter = iter(train_loader)
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


#%% error?

# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
train_iter, test_iter = iter(train_loader), iter(test_loader)
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

#train_loader = DataLoader(split_train_, batch_size=BATCH_SIZE,
#                              shuffle=True, collate_fn=collate_batch)
#val_loader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
#                              shuffle=True, collate_fn=collate_batch)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
#                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_loader)
    accu_val = evaluate(val_loader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
    
    
print('Checking the results of test dataset.')
accu_test = evaluate(test_loader)
print('test accuracy {:8.3f}'.format(accu_test))

#%%
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])



