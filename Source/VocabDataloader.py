from torch.utils.data import Dataset
import pandas as pd
import torch


class NewsDataset(Dataset):
    def __init__(self, json_path):
        self.df = pd.read_json(json_path, lines=True)[['category', 'headline']]  # Load data, but only keep columns of interest
        self.df = self.df.dropna(axis=0)  # Remove rows with no category or headline
        self.df = self.df.loc[self.df['headline'].str.len() > 0]  # Remove rows where headline is empty string
        self.df = self.df.loc[self.df['headline'].str.len() <= 120]  # Remove 274 rows where length of headline is above 120
        self.df['category'] = self.df['category'].replace({'ARTS & CULTURE': 'CULTURE & ARTS'})  # Same label
        self.df['headline'] = self.df['headline'].str.lower()  # All headlines in lower case

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_row = self.df.iloc[idx, :]
        data_point_category = data_row[0]
        data_point_headline = data_row[1]
        return data_point_category, data_point_headline


def loader_for_vocab():
    data_path = '../Data/News_Category_Dataset_v2.json'

    dataset = NewsDataset(data_path)

    loader_ = torch.utils.data.DataLoader(dataset)

    return loader_
