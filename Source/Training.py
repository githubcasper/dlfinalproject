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
neptune_on        = False
num_hidden_layers = 1
size_hidden_layer = 43
emsize            = 67
dropout           = 0.4501
dropout_lstm      = 0.5
batch_size        = 300
learning_rate     = 0.01
class_weights     = 1

train_loader, val_loader, test_loader, class_weights_ = get_loaders(batch_size_train=batch_size,
                                                                    batch_size_val=300,
                                                                    batch_size_test=300,
                                                                    test_split=0.1,
                                                                    val_split=0.1,
                                                                    shuffle_dataset=True,
                                                                    random_seed=123)

tokenizer              = get_tokenizer('basic_english')
vocab_sizes            = VocabSizes(tokenizer)
vocab_size, vocab_text = vocab_sizes.get_vocab_size_text()
vocab_label            = vocab_sizes.get_label_dict()
vocab_int_to_label     = vocab_sizes.get_int_to_label_dict()
max_length             = vocab_sizes.get_max_len()
text_pipeline          = lambda x: vocab_text(tokenizer(x))
amount_of_categories   = len(vocab_sizes.get_label_dict())


model = LSTMModel(vocab_size, emsize, dropout, dropout_lstm, num_hidden_layers, 
                  size_hidden_layer, max_length, amount_of_categories).to(device)


if class_weights:
    class_weights = sorted([[vocab_label[i[0]], i[1]] for i in class_weights_.items()], key=lambda x: x[0])
    class_weights = torch.Tensor([i[1] for i in class_weights]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion2 = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(reduction='sum')


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
optimizer_name = type(optimizer).__name__


#%% Neptune
if neptune_on:
    secret_api = keyring.get_password('Neptune', "andreastind")
    
    run = neptune.init(project="andreastind/DeepLearningFinalProject",
                       api_token=secret_api)
    
    params = {"Number of hidden layers": num_hidden_layers,
              "Size of hidden layer":    size_hidden_layer,
              "Embedding size":          emsize,
              "Dropout":                 dropout,
              "Batch size":              batch_size,
              "Learning rate":           learning_rate, 
              "Optimizer":               optimizer_name}
    
    run["parameters"] = params


#%% Training

def train(dataloader, model):
    model.train()
    log_interval = 100
    total_loss = []
    start_time = time.time()
    n_data = len(dataloader)
    if neptune_on:
        run["train/n_data"] = n_data

    for batch_idx, (batch_labels, batch_texts) in enumerate(dataloader):
        predicted_labels = model(batch_texts, batch_texts.size(0))
        predicted_labels = predicted_labels.squeeze(1)

        loss = criterion(predicted_labels, batch_labels)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if neptune_on:
            run["train/batch_idx"].log(batch_idx)
            run["train/avg_batch_loss"].log(loss.item()) # log average loss to neptune

        if ((batch_idx+1) % log_interval == 0 and batch_idx > 0) or (batch_idx+1 == n_data):
            elapsed = time.time() - start_time
            print('| {:4d} /{:4d} batches '
                  '| train_loss {:5.3f} '
                  '| Time elapsed {:5.1f}s |'.format(batch_idx+1,
                                                     n_data,
                                                     loss,
                                                     elapsed))
    return total_loss


#tot_loss = train(iter(train_loader), model)


#%% Evaluate model

def evaluate(dataloader, model):
    model.eval()
    correct_count_in_batch_list = [] 
    count_in_batch_list = []
    n_data = len(dataloader)
    correct = 0
    count = 0
    loss = 0
    start_time = time.time()

    with torch.no_grad():
        for idx, (labels, texts) in enumerate(dataloader):
            predicted_labels = model(texts, texts.size(0))
            batch_class_preds = predicted_labels.argmax(2).squeeze()

            count += len(labels)

            loss += criterion2(predicted_labels.squeeze(), labels)
            cum_loss = loss / count
            
            correct += torch.sum(torch.eq(batch_class_preds, labels)).item()
            accuracy = 100 * correct / count
            
            correct_count_in_batch_list.append(correct)
            count_in_batch_list.append(len(labels))
            
            if ((idx+1) % 25 == 0 and idx > 0) or (idx+1 == n_data):
                elapsed = round(time.time() - start_time, 2)
                print("| {:3d}/{:3d} batches "
                      "| validation_loss: {:.4f} "
                      "| validation_accuracy: {:3.2f}% "
                      "| time elapsed: {:5.1f}s |".format(idx+1,
                                                          n_data,
                                                          cum_loss,
                                                          accuracy,
                                                          elapsed))
            
    return accuracy, cum_loss

#eval_acc = evaluate(iter(val_loader), model)

#%% Epoch loop

epochs = 3
avg_epoch_loss = []
epoch_val_acc_list = []
epoch_val_loss_list = []
best_loss = 100


for epoch in range(epochs):
    print("Initiating training...")
    epoch_loss = train(train_loader, model)
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    avg_epoch_loss.append(epoch_loss)
    
    if neptune_on:
        run["train/avg_epoch_loss"].log(epoch_loss)    

    print("Initiating evaluation...")
    epoch_val_acc, epoch_val_loss = evaluate(iter(val_loader), model)
    epoch_val_acc_list.append(epoch_val_acc)
    epoch_val_loss_list.append(epoch_val_loss)
    scheduler.step(epoch_val_loss)

    if neptune_on:
        run["validation/epoch_val_accuracy"].log(epoch_val_acc)
        run["validation/epoch_val_loss"].log(epoch_val_loss)

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        torch.save(model.state_dict(), 'model_weights_best_val_loss.pth')
    
 

    print("Finished epoch: {}/{} "
          "| Val_loss {:.4f} "
          "| Val_accuracy: {:3.2f}% |".format(epoch+1, epochs, epoch_val_loss, epoch_val_acc))

torch.save(model.state_dict(), 'model_weights_12-14-13-25.pth')

if neptune_on:
    run.stop()

#%%
model = LSTMModel(vocab_size, emsize, dropout, dropout_lstm, num_hidden_layers, 
                  size_hidden_layer, max_length, amount_of_categories).to(device) 
model.load_state_dict(torch.load('model_weights_12-12-00-49.pth'))
model.eval()

#run.stop()

#%%
import matplotlib.pyplot as plt

plt.plot(avg_epoch_loss)
plt.ylabel("Average epoch loss")
plt.xlabel("Epoch number")
plt.show()


#%% Custom input eval


def custom_input_eval(input_string, model):
    model.eval()
    placeholder = text_pipeline("placeholder")
    input_string = text_pipeline(input_string)
    
    while len(placeholder) < 94:
        placeholder.append(text_pipeline('<pad>')[0])
    while len(input_string) < 94:
        input_string.append(text_pipeline('<pad>')[0])
    
    tens = torch.tensor([input_string, placeholder], dtype=torch.int64, device=device)
    preds = model(tens, tens.size(0)).squeeze()
    pred = preds.squeeze(1)[0].argmax(0).item()
    return vocab_int_to_label[pred]


tmp_ = "Former NFL Star Demaryius Thomas Found Dead At 33"
tmp_ = "sport sport sport sport sport sport"
tmp_ = "Jimmy Kimmel Mocks Fox News For Spinning Christmas Tree Fire Into A ‘Hate Crime’"
tmp_ = "Daunte Wright’s Girlfriend Recalls His Death In Emotional Testimony At Kim Potter Trial"
tmp_ = "Appeals Court Denies Trump’s Request To Keep Jan. 6 Records Hidden"
tmp_ = "53 Migrants Dead, 54 Injured In Truck Crash In South Mexico"
tmp1 = "Stephen Colbert Turns Fox News' Latest Whine Into A Taunting New Chant"
tmp_ = "Is It Rude To Send A Cocktail Back?"
tmp_ = "How To Advocate For Yourself In Your Year-End Review"
tmp_ = "Are Your House Slippers Destroying Your Feet? Here’s What Podiatrists Say."
tmp_ = "Body Found, Boyfriend Arrested Amid Search For Florida Woman Abducted From Work"
tmp_ = "Chile's President Signs Same-Sex Marriage Bill Into Law After Historic Vote"
tmp_ = "Simone Biles Is Time Magazine's 2021 Athlete Of The Year"
tmp_ = "I'm Black But Look White. Here Are The Horrible Things White People Feel Safe Telling Me."
#tmp_ = "Body Found, Boyfriend Arrested Amid Search For Florida Woman Abducted From Work"
tmp_ = "As Biden Talks of a Boom, Inflation and the Virus Weigh on Americans"
#tmp_ = "The National Bank disagrees with government: there is still a need for interference in the housing market"
#tmp_ = "Sex and The City has resurrected and the critics are not imppressed"

print(custom_input_eval(tmp_, model))



