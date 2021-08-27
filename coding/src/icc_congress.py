import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg

responses = pd.read_csv('data/results-congress.csv')

df1 = responses.filter(regex=('.*\[1\]')).dropna()
df2 = responses.filter(regex=('.*\[2\]')).dropna()
df3 = responses.filter(regex=('.*\[3\]')).dropna()

# from coding/experiments/nyt/07-20-2021/EI/generatelimesurveyprompts.p
# df = pd.read_pickle('experiments/nyt/07-20-2021/EI/EIwhole.pickle')
df = pd.read_pickle('experiments/congress/08-04-2021/EI/gpt3.pkl')
# lsdf = df[(df.n_exemplars == 3) & (df.exemplar_set_ix == 0) & (df.n_per_category == 4)].set_index('level_0')
lsdf = df[(df.n_exemplars == 3) & (df.exemplar_set_ix == 0) & (df.n_per_category == 4)]# .set_index('level_0')
# reset index
lsdf = lsdf.reset_index()

def rename_columns(df):
    '''Extract the question number and rename all columns to be an int'''
    columns = list(df.columns)
    # take everything after Q
    columns = [x.split('Q')[1] for x in columns]
    # take everything before [
    columns = [x.split('[')[0] for x in columns]
    # TODO - remove? off by one shift
    # convert to int
    # columns = [int(x) for x in columns]
    columns = [int(x) -1 for x in columns]
    # rename    
    df.columns = columns
    return df

# rename columns for df1, df2, and df3
df1 = rename_columns(df1)
df2 = rename_columns(df2)
df3 = rename_columns(df3)

true, gpt, coder0, coder1 = lsdf.category, lsdf.guess, df1.iloc[0], df1.iloc[1]
# combine into dataframe
df = pd.DataFrame({'true': true, 'gpt': gpt, 'coder0': coder0, 'coder1': coder1})
# dropna
df = df.dropna()
question_numbers = df.index.to_list()

# calculate scores for each coder
coders = ['coder0', 'coder1', 'gpt']
for coder in coders:
    df[coder + '_score'] = 1*(df[coder] == df.true)

# make 'index' column
df['index'] = df.index
lsdf['index'] = lsdf.index
# merge on index
df = pd.merge(df, lsdf, on='index')

# # dropna
# df = df.dropna()
# question_numbers = df.index.to_list()
# 
# coders = ['coder0', 'coder1', 'coder2', 'coder3', 'gpt']
# # coders = ['gpt', 'true']
# target = df.true
# 
# ratings = []
# for coder in coders:
#     column = df[coder]
#     df_add = pd.DataFrame({'question_number': question_numbers, 'rating': column, 'target': target})
#     df_add['coder'] = coder
#     ratings.append(df_add)
# ratings = pd.concat(ratings)
# 
# # convert each column to an int through unique
# for column in ratings.columns:
#     unique = np.unique(ratings[column])
#     map_dict = {unique[i]: i for i in range(len(unique))}
#     ratings[column] = ratings[column].map(map_dict)
# 
# icc = pg.intraclass_corr(data=ratings, targets='question_number', raters='coder', ratings='rating')
# icc

# get accuracies and agg by mean per coder
accuracies = df.groupby('true').agg('mean')
# sort accuracies
accuracies = accuracies.sort_values(by='gpt_score', ascending=False)
# plot
for column in accuracies.columns:
    plt.plot(accuracies[column], label=column)
plt.ylim(0,1)
plt.title('Accuracies')
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.subplots_adjust(bottom=0.5)
plt.legend()
plt.show()
# plt.show()
if save_path is not None:
    plt.savefig(save_path + '_accuracies.pdf')
    # clear plt
    plt.clf()