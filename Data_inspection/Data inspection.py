import numpy as np
import pandas as pd

df = pd.read_json('../News_Category_Dataset_v2.json', lines=True)
pd.set_option('display.max_columns', 100)
df = df.drop(df[df['headline'] == ""].index) #removing rows with empty headline

print(df.head())

amount_of_data = len(df)
amount_of_labels = len(set(df['category']))
list_of_lengths = [len(i) for i in df['headline']]
mean_len_headline = round(np.mean(list_of_lengths), 2)
mean_words_headline = round(np.mean([i.count(" ") + 1 for i in df['headline']]), 2)
max_len_headline = max(list_of_lengths)
min_len_headline = min(list_of_lengths)

print(f"""\nIn this project we are interested in predicting the category of 
online news through their headlines. In our dataset there are {amount_of_data} 
different news articles with in total {amount_of_labels} different news 
categories. The length of the average headline is {mean_len_headline} 
characters, and {mean_words_headline} in terms of average number of words. The 
maximum headline length in terms of characters is {max_len_headline}, while the
minimum is {min_len_headline}.""")