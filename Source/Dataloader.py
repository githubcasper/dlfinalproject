import pandas as pd
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler
import torch


class NewsDataset(Dataset):
    def __init__(self, json_path):
        self.df = pd.read_json(json_path, lines=True)[['category', 'headline']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_row = self.df.iloc[idx, :]
        data_point_category = data_row[0]
        data_point_headline = data_row[1]
        return data_point_category, data_point_headline


def get_loaders(batch_size: int, test_split: float, val_split: float, shuffle_dataset: bool, random_seed: int):
    batch_size = batch_size
    test_split = test_split
    val_split = val_split
    shuffle_dataset = shuffle_dataset
    random_seed = random_seed

    df = pd.read_json('../Data/News_Category_Dataset_v2.json', lines=True)

    amount_of_data = len(df)
    # Create indices to randomly split data into training and test sets:

    dataset = NewsDataset('../Data/News_Category_Dataset_v2.json')

    indices = list(range(amount_of_data))
    split = int(np.floor(test_split * amount_of_data))
    split_val = int(np.floor((val_split + test_split) * amount_of_data))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split_val:], indices[split:split_val], indices[:split]

    # Create samplers and DataLoaders
    test_sampler = SubsetRandomSampler(test_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)


    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(dataset,
                                             sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(dataset,
                                              sampler=test_sampler)

    return train_loader, val_loader, test_loader
