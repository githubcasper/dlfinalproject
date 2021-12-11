# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:07:56 2021

@author: Andreas Tind
"""

from Dataloader import get_loaders
import neptune.new as neptune
import keyring
from RNNModel import LSTMModel, VocabSizes
import time
import torch
from torch import nn
#import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Model setup

num_hidden_layers = 4
size_hidden_layer = 17
emsize = 25
dropout = 0.2
batch_size = 500
learning_rate = 0.01

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
text_pipeline = lambda x: vocab_text(tokenizer(x))

model = LSTMModel(vocab_size, emsize, dropout, num_hidden_layers, size_hidden_layer, max_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#%% Neptune

secret_api = keyring.get_password('Neptune', "andreastind")

run = neptune.init(project="andreastind/DeepLearningFinalProject",
                   api_token=secret_api)

params = {"num_hidden_layers": num_hidden_layers,
          "size_hidden_layer": size_hidden_layer,
          "embidding_size": emsize,
          "dropout": dropout,
          "batch_size": batch_size,
          "learning_rate": learning_rate, 
          "Optimizer": "Adam"}
run["parameters"] = params

#for epoch in range(10):
#    run["train/loss"].log(0.9 ** epoch)

#run["eval/f1_score"] = 0.66
#run.stop()


#%% Training

def train_step(batch_labels, batch_texts):
    '''
    takes one training step using one batch
    '''
    loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float32, device=device))
    total_correct = 0 
    total_count = 0

    for i in range(len(batch_texts)): # iterate through the batch
        input_list = text_pipeline(batch_texts[i])
        while len(input_list) < max_length:
            input_list.append(text_pipeline('<pad>')[0])
        input_tensor = torch.tensor(input_list, dtype=torch.int64, device=device)
        predicted_label = model(input_tensor)
        
        target_label = torch.tensor(vocab_label[batch_labels[i]],
                                    dtype=torch.int64,
                                    device=device).unsqueeze(0)
    
        loss += criterion(predicted_label, target_label)

        total_correct += (predicted_label.argmax(1) == target_label).sum().item()
        total_count += target_label.size(0)

    return loss, total_correct, total_count



def train(dataloader):
    model.train()
#    total_acc, total_count = 0, 0
    log_interval = 2
    total_loss = []
    start_time = time.time()
    n_data = len(dataloader)
    run["train/n_data"] = n_data

    for batch_num, (batch_labels, batch_texts) in enumerate(dataloader):
        optimizer.zero_grad()
        loss, total_correct, total_count = train_step(batch_labels, batch_texts)
        loss.backward()

        print("avg_loss:", loss.item()/total_count)
        total_loss.append(loss.item()/total_count)

        run["train/avg_batch_loss"].log(loss.item()/total_count) # log average loss to neptune
        run["train/avg_batch_accuracy"].log(round(100*total_correct/total_count, 3)) # log average accuracy to neptune

        if batch_num % log_interval == 0 and batch_num > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches '
                  '| train_accuracy {:8.3f} | Time elapsed {:.2f}s'.format(batch_num,
                                                                           n_data,
                                                                           total_correct/total_count,
                                                                           elapsed))
        optimizer.step()
        
    return total_loss


train(train_loader)

#%%

run.stop()

#%% Evaluate model

def evaluate(dataloader):
    model.eval()
    total_correct = 0 
    total_count = 0
    n_data = len(dataloader)
    start_time = time.time()

    
    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
#            loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float32, device=device))
                
            input_list = text_pipeline(text[0])
            while len(input_list) < max_length:
                input_list.append(text_pipeline('<pad>')[0])
            input_tensor = torch.tensor(input_list, dtype=torch.int64, device=device)

            predicted_label = model(input_tensor)
            target_label = torch.tensor(vocab_label[label[0]],
                                        dtype=torch.int64,
                                        device=device).unsqueeze(0)
            
#            loss += criterion(predicted_label, target_label)
            total_correct += (predicted_label.argmax(1) == target_label).sum().item()
            total_count += target_label.size(0)
            
            if idx % 500 == 0 and idx > 0:
                elapsed = round(time.time() - start_time, 2)
                print("| {:5d}/{:5d} evaluated "
                      "| accuracy: {:3.2f}% | time elapsed: {:.1f}s".format(#loss,
                                                  idx,
                                                  n_data,
                                                  100*total_correct/total_count,
                                                  elapsed))
            
    return total_correct/total_count

eval_acc = evaluate(test_loader)

#%% Custom input eval

model.eval()

def custom_input_eval(input_string):
    pipe = text_pipeline(input_string)
    while len(pipe) < 35:
        pipe.append(text_pipeline('<pad>')[0])
    pipe = torch.tensor(pipe, dtype=torch.int64, device=device)
    label_int_pred = model(pipe).argmax(1).item()
    return vocab_int_to_label[label_int_pred]


tmp_ = "Former NFL Star Demaryius Thomas Found Dead At 33"
tmp_ = "sport sport sport sport sport sport"
tmp_ = "Jimmy Kimmel Mocks Fox News For Spinning Christmas Tree Fire Into A ‘Hate Crime’"
tmp_ = "Daunte Wright’s Girlfriend Recalls His Death In Emotional Testimony At Kim Potter Trial"
tmp_ = "Appeals Court Denies Trump’s Request To Keep Jan. 6 Records Hidden"
tmp_ = "53 Migrants Dead, 54 Injured In Truck Crash In South Mexico"
tmp_ = "Stephen Colbert Turns Fox News' Latest Whine Into A Taunting New Chant"
tmp_ = "Is It Rude To Send A Cocktail Back?"
tmp_ = "How To Advocate For Yourself In Your Year-End Review"
tmp_ = "Are Your House Slippers Destroying Your Feet? Here’s What Podiatrists Say."
tmp_ = "Body Found, Boyfriend Arrested Amid Search For Florida Woman Abducted From Work"
tmp_ = "Chile's President Signs Same-Sex Marriage Bill Into Law After Historic Vote"
tmp_ = "Simone Biles Is Time Magazine's 2021 Athlete Of The Year"
tmp_ = "I'm Black But Look White. Here Are The Horrible Things White People Feel Safe Telling Me."
tmp_ = "Body Found, Boyfriend Arrested Amid Search For Florida Woman Abducted From Work"
tmp_ = "As Biden Talks of a Boom, Inflation and the Virus Weigh on Americans"
tmp_ = "The National Bank disagrees with government: there is still a need for interference in the housing market"
tmp_ = "Sex and The City has resurrected and the critics are not imppressed"

custom_input_eval(tmp_)




#%% Garbage code storage below


train_loader, val_loader, test_loader = get_loaders(batch_size=1, 
                                                    test_split=0.1, 
                                                    val_split=0.1, 
                                                    shuffle_dataset=True, 
                                                    random_seed=123)


# TEXT VOCAB GENERATION
tokenizer = get_tokenizer('basic_english')
class VocabSizes():
    def __init__(self, data_loader, tokenizer):
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    @staticmethod
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
#model = LSTMModel(vocab_size, emsize, num_class).to(device)

#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                       mode='min',
#                                                       factor=1,
#                                                       patience=15)


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


#epochs = 10
#for epoch in range(epochs):
#    train(train_loader)

#val_loss = evaluate(val_loader)

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



