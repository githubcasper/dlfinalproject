import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_json('../Data/News_Category_Dataset_v2.json', lines=True)

label_occurence = Counter(df['category']).most_common()
plt.rcParams["figure.figsize"] = (18.5, 10.5)
plt.barh([i[0] for i in label_occurence], [i[1] for i in label_occurence])
plt.show()
