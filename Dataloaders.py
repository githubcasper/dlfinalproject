import pandas as pd
from torch.utils.data import Dataset


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
