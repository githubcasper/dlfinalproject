import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from VocabDataloader import loader_for_vocab
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, dropout, dropout_lstm, num_hidden_layers, size_hidden_layer, max_length, classes):
        super(LSTMModel, self).__init__()
        self.size_embed = embed_dim
        self.size_hidden_layer = size_hidden_layer
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout
        self.dropout_lstm_p = dropout_lstm

        self.embedding = nn.Embedding(vocab_size, self.size_embed, sparse=False)

        self.rnn = nn.LSTM(input_size=self.size_embed,
                           hidden_size=self.size_hidden_layer,
                           num_layers=self.num_hidden_layers,
                           bidirectional=True,
                           dropout=self.dropout_lstm_p,
                           batch_first=True)

        self.out = nn.Linear(self.size_hidden_layer * max_length * 2, classes)  # 40 different classes
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.dropout_lstm = nn.Dropout(p=self.dropout_lstm_p)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, text, batch_size):
        hidden = (self.initHidden(batch_size), self.initHidden(batch_size))
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout_lstm(output)
        output = self.out(output.reshape(batch_size, 1, -1))
        return F.log_softmax(output, dim=0)

    def initHidden(self, batch_size):
        return torch.zeros(2 * self.num_hidden_layers, batch_size, self.size_hidden_layer, device=device)


class VocabSizes():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 0
        self.label_dict = {}
        self.loader = loader_for_vocab()

    @staticmethod
    def yield_tokens_text(self, data_iter):
        count = 0
        for label, text in data_iter:
            if label[0] not in self.label_dict.keys():
                self.label_dict[label[0]] = count
                count += 1
            the_tokenized = self.tokenizer(text[0])
            if len(the_tokenized) > self.max_len:
                self.max_len = len(the_tokenized)
            yield the_tokenized

    def get_vocab_size_text(self):
        train_iter = iter(self.loader)
        vocab_text = build_vocab_from_iterator(self.yield_tokens_text(self, train_iter), specials=["<unk>"])
        vocab_text.set_default_index(vocab_text["<unk>"])
        return len(vocab_text), vocab_text

    def get_max_len(self):
        return self.max_len

    def get_label_dict(self):
        return self.label_dict

    def get_int_to_label_dict(self):
        return {v: k for k, v in self.label_dict.items()}
