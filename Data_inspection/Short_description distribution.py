import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_json('../Data/News_Category_Dataset_v2.json', lines=True)[['category', 'headline', 'short_description']]
pd.set_option('display.max_columns', 100)
df = df.loc[df['short_description'].str.len() > 0]
df = df.dropna(axis=0)  # Remove rows with no category or headline
df = df.loc[df['short_description'].str.len() <= 300]  # Remove rows where length of short_description is above 300
df['category'] = df['category'].replace({"ARTS & CULTURE": "CULTURE & ARTS",
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
df['short_description'] = df['short_description'].str.lower()  # All headlines in lower case

list_of_lengths = [len(i) for i in df['short_description']]
length_occurence = Counter(list_of_lengths).most_common()

plt.rcParams["figure.figsize"] = (18.5, 10.5)
plt.hist(list_of_lengths, bins=300)
plt.xlabel('Length of headline')
plt.ylabel('Occurrences')
plt.show()
