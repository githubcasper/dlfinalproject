import pandas as pd
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler
import torch
from RNNModel import VocabSizes
from torchtext.data.utils import get_tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NewsDatasetTraining(Dataset):
    def __init__(self, json_path, input_length, vocab, label_dict):
        self.label_dict = label_dict
        self.vocab = vocab
        self.input_length = input_length
        self.df = pd.read_json(json_path, lines=True)[['category', 'headline']]  # Load data, but only keep columns of interest
        self.df = self.df.dropna(axis=0)  # Remove rows with no category or headline
        self.df = self.df.loc[self.df['headline'].str.len() > 0]  # Remove rows where headline is empty string
        self.df = self.df.loc[self.df['headline'].str.len() <= 120]  # Remove 274 rows where length of headline is above 120
        self.df['category'] = self.df['category'].replace({"ARTS & CULTURE": "CULTURE & ARTS",
                                                           "HEALTHY LIVING": "WELLNESS",
                                                           "QUEER VOICES": "VOICES",
                                                           "BUSINESS": "BUSINESS & FINANCES",
                                                           "PARENTS": "PARENTING",
                                                           "BLACK VOICES": "VOICES",
                                                           "THE WORLDPOST": "WORLD NEWS",
                                                           "STYLE": "STYLE & BEAUTY",
                                                           "GREEN": "ENVIRONMENT",
                                                           "TASTE": "FOOD & DRINK",
                                                           "WORLDPOST": "WORLD NEWS",
                                                           "SCIENCE": "SCIENCE & TECH",
                                                           "TECH": "SCIENCE & TECH",
                                                           "MONEY": "BUSINESS & FINANCES",
                                                           "ARTS": "CULTURE & ARTS",
                                                           "COLLEGE": "EDUCATION",
                                                           "LATINO VOICES": "VOICES",
                                                           "FIFTY": "MISCELLANEOUS",
                                                           "GOOD NEWS": "MISCELLANEOUS"})  # Group some categories
        self.df['headline'] = self.df['headline'].str.lower()  # All headlines in lower case

    def __len__(self):
        return len(self.df)

    @staticmethod
    def str_to_ints(string, input_length, vocab):
        input_list = vocab(string)
        while len(input_list) < input_length:
            input_list.append(vocab('<pad>')[0])
        return input_list

    @staticmethod
    def label_to_int(string, vocab):
        return vocab[string]

    def __getitem__(self, idx):
        data_row = self.df.iloc[idx, :]
        data_point_category = data_row[0]
        data_point_headline = data_row[1]
        return data_point_category, data_point_headline


def get_loaders(batch_size: int, test_split: float, val_split: float, shuffle_dataset: bool, random_seed: int):

    tokenizer = get_tokenizer('basic_english')
    vocab_sizes = VocabSizes(tokenizer)
    vocab_size, vocab_text = vocab_sizes.get_vocab_size_text()
    vocab_label = vocab_sizes.get_label_dict()
    max_length = vocab_sizes.get_max_len()
    text_pipeline = lambda x: vocab_text(tokenizer(x))

    data_path = '../Data/News_Category_Dataset_v2.json'
    df = pd.read_json(data_path, lines=True)
    df = df.dropna(axis=0)  # Remove rows with no category or headline
    df = df.loc[df['headline'].str.len() > 0]  # Remove rows where headline is empty string
    df = df.loc[df['headline'].str.len() <= 120]  # Remove 274 rows where length of headline is above 120

    amount_of_data = len(df)
    dataset = NewsDatasetTraining(data_path, max_length, text_pipeline, vocab_label)

    # Create indices to randomly split data into training and test sets:
    indices = list(range(amount_of_data))
    split = int(np.floor(test_split * amount_of_data))
    split_val = int(np.floor((val_split + test_split) * amount_of_data))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split_val:], indices[split:split_val], indices[:split]

    # Create samplers and DataLoaders
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(vocab_label[_label])
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            while len(processed_text) < max_length:
                processed_text = torch.cat((processed_text, torch.tensor(text_pipeline('<pad>')[0], dtype=torch.int64).unsqueeze(0)))
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.stack(text_list)
        return label_list.to(device), text_list.to(device)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               collate_fn=collate_batch)

    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=100000,  # Large number chosen to get entire validation as one batch
                                             sampler=val_sampler,
                                             collate_fn=collate_batch)

    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=100000,  # Large number chosen to get entire validation as one batch
                                              sampler=test_sampler,
                                              collate_fn=collate_batch)

    return train_loader, val_loader, test_loader

