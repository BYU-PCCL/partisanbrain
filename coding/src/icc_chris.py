import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg

# read in data/nytcodes3humansandgpt3.csv
df = pd.read_csv('data/nytcodes3humansandgpt3.csv', index_col=0)
# reset index
df = df.reset_index()

gpt, coder0, coder1, coder2 = df['gpt3guess'], df['coder0'], df['coder1'], df['coder2']

# combine into dataframe
df = pd.DataFrame({'gpt': gpt, 'coder0': coder0, 'coder1': coder1, 'coder2': coder2})
# dropna
df = df.dropna()
question_numbers = df.index.to_list()

coders = ['coder0', 'coder1', 'coder2', 'gpt']
# coders = ['gpt', 'true']
# target = df.true

ratings = []
for coder in coders:
    column = df[coder]
    df_add = pd.DataFrame({'question_number': question_numbers, 'rating': column})
    df_add['coder'] = coder
    ratings.append(df_add)
ratings = pd.concat(ratings)

# convert each column to an int through unique
for column in ratings.columns:
    unique = np.unique(ratings[column])
    map_dict = {unique[i]: i for i in range(len(unique))}
    ratings[column] = ratings[column].map(map_dict)

# icc = pg.intraclass_corr(data=ratings, targets='question_number', raters='coder', ratings='rating')
icc = pg.intraclass_corr(data=ratings, targets='question_number', raters='coder', ratings='rating')
icc