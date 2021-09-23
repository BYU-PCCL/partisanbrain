import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg
from congress_categories import categories
from shift_logits import *
from plotting_utils import *

# responses = pd.read_csv('data/results-congress.csv')
responses = pd.read_csv('data/results-survey556688.csv')

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

######################
# take values from dict
categories = list(categories.values())
# shift logits
# target_mean = np.ones(len(categories)) / len(categories)
# probs = lsdf[categories].values
# shifted_probs, shift = fit(probs, target_mean)
# 
# # replace lsdf[categories] with shifted_probs
# lsdf[categories] = shifted_probs
# # replace guess with argmax col of lsdf[categories]
# lsdf['guess'] = lsdf[categories].idxmax(axis=1)
######################


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

true, gpt, coder0, coder1, coder2 = lsdf.category, lsdf.guess, df1.iloc[0], df1.iloc[1], df1.iloc[2]
# combine into dataframe
df = pd.DataFrame({'true': true, 'gpt': gpt, 'coder0': coder0, 'coder1': coder1, 'coder2': coder2})
# dropna
df = df.dropna()
question_numbers = df.index.to_list()

# save df as csv
df.to_csv('congress_data.csv', index=False)

breakpoint()


# calculate scores for each coder
coders = ['gpt', 'coder0', 'coder1', 'coder2']
for coder in coders:
    df[coder + '_score'] = 1*(df[coder] == df.true)
    # print coder and average score
    print(coder + ': ' + str(np.round(df[coder + '_score'].mean(), 3)))

# make 'index' column
df['index'] = df.index
lsdf['index'] = lsdf.index
# merge on index
df = pd.merge(df, lsdf, on='index')

# calculate joint agreement for each coder
# make empty dataframe
joint_agreement = pd.DataFrame(columns=coders, index=coders)
for coder1 in coders:
    for coder2 in coders:
        agreement = 1*(df[coder1] == df[coder2]).mean()
        joint_agreement.loc[coder1, coder2] = agreement
        joint_agreement.loc[coder2, coder1] = agreement
print(joint_agreement)


# # get accuracies and agg by mean per coder
# accuracies = df.groupby('true').agg('mean')
# # sort accuracies
# accuracies = accuracies.sort_values(by='gpt_score', ascending=False)
# # plot
# for coder in coders:
#     plt.plot(accuracies[coder + '_score'], label=coder)
# plt.ylim(0,1)
# plt.title('Accuracies')
# plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.subplots_adjust(bottom=0.5)
# plt.legend()
# plt.show()

plt.figure(figsize=(15, 10))
# get accuracies and agg by mean per coder
accuracies = df.groupby('true').agg('mean')
# sort accuracies
accuracies = accuracies.sort_values(by='gpt_score', ascending=False)
label_map = {'gpt': 'GPT-3',
    'coder0': 'Human 1',
    'coder1': 'Human 2',
    'coder2': 'Human 3',
    'coder3': 'Human 4',
}
labels = [label_map[x] for x in coders]
# plot
for coder, label in zip(coders, labels):
    # plot with markers
    # plt.plot(accuracies[coder + '_score'], label=label, marker='o', alpha=.9)
    plt.plot(accuracies[coder + '_score'], label=label, alpha=.8)
plt.ylim(0,1)
xlim = plt.xlim()

# plot faded gray horizontal lines at [.2, .4, .6, .8]
plt.hlines(.2, xlim[0], xlim[1], color='gray', alpha=.2)
plt.hlines(.4, xlim[0], xlim[1], color='gray', alpha=.2)
plt.hlines(.6, xlim[0], xlim[1], color='gray', alpha=.2)
plt.hlines(.8, xlim[0], xlim[1], color='gray', alpha=.2)
# reset xlim to xlim
plt.xlim(xlim)
plt.ylabel('Accuracy')
plt.title('Congress Accuracies')
plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right')
plt.subplots_adjust(bottom=0.5)
plt.legend()
# save as congress_accuracies.pdf
plt.savefig('congress_accuracies.pdf')
plt.show()

corr = joint_agreement.values.astype(float)
plot_correlations_congress(corr, labels)
plt.title('Congress Joint Agreement')
# save fig as congress_jointagreement.pdf
plt.savefig('congress_jointagreement.pdf')
plt.show()