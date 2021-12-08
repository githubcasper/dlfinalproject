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
from Source.Dataloader import get_loaders
import torch.nn.functional as F


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, dropout, num_hidden_layers, size_hidden_layer, output_dim=None):
        super(LSTMModel, self).__init__()
        self.size_embed = embed_dim
        self.size_hidden_layer = size_hidden_layer
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout

        self.embedding = nn.Embedding(vocab_size, self.size_embed, sparse=True)

        self.rnn = nn.LSTM(input_size=self.size_embed,
                           hidden_size=self.size_hidden_layer,
                           num_layers=self.num_hidden_layers,
                           bidirectional=True,
                           batch_first=True)

        self.out = nn.Linear(self.size_hidden_layer * self.size_embed * 2, 40)  # 40 different classes
        self.dropout = nn.Dropout(p=self.dropout_p)
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

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        output, _ = self.rnn(embedded.unsqueeze(0))
        output = self.out(output.view(len(text), 1, -1))
        return F.log_softmax(output, dim=1)

    def initHidden(self):
        return torch.zeros(1, 1, self.size_embed, device=device)


class VocabSizes():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_loader, self.val_loader, self.test_loader = get_loaders(batch_size=1,
                                                                           test_split=0.0,
                                                                           val_split=0.0,
                                                                           shuffle_dataset=True,
                                                                           random_seed=123)

    @staticmethod
    def yield_tokens_text(self, data_iter):
        for label, text in data_iter:
            yield self.tokenizer(text[0])

    def get_vocab_size_text(self):
        train_iter = iter(self.train_loader)
        vocab_text = build_vocab_from_iterator(self.yield_tokens_text(self, train_iter), specials=["<unk>"])
        vocab_text.set_default_index(vocab_text["<unk>"])
        return len(vocab_text), vocab_text

    @staticmethod
    def yield_tokens_label(self, data_iter):
        for label, text in data_iter:
            yield self.tokenizer(label[0])

    def get_vocab_size_label(self):
        train_iter = iter(self.train_loader)
        vocab_label = build_vocab_from_iterator(self.yield_tokens_label(self, train_iter), specials=["<unk>"])
        vocab_label.set_default_index(vocab_label["<unk>"])
        return len(vocab_label), vocab_label
