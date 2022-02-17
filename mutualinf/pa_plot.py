import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

# read in data/pa/plot_data.pkl
df = pd.read_pickle('data/pa/plot_data.pkl')

# if plots doesn't exist,
# make plots directory
if not os.path.exists('plots'):
    os.mkdir('plots')

# %%
# get all indices
indices = df.index.values
for index in indices:
    # make a new df with the model as the index, and accuracy, correct_weight, mutual_inf, and param_count as columns
    models = list(df.columns)
    d = {m: df.loc[index, m].loc['first_person_backstory'].to_dict() for m in models}
    # to anes df
    anes_df = pd.DataFrame(d).T

    # plot accuracy vs param_count
    # plt.figure(figsize=(10, 6))
    # color by model type
    model_types = anes_df['model_type'].unique()
    # for each model type, plot and color
    for model_type in model_types:
        df_model_type = anes_df[anes_df.model_type == model_type]
        #
        # plt.scatter(df_model_type['param_count'], df_model_type['accuracy'], label=model_type)
        # instead of scatter, do line plot with a dot for each point
        plt.plot(df_model_type['param_count'], df_model_type['accuracy'], label=model_type, marker='o')
    # log scale for param
    plt.xscale('log')
    # make y lim from .4 to 1
    plt.ylim(.4, .95)
    plt.xlabel('param count (log)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(index)
    plt.savefig(f'plots/pa/{index}_accuracy_vs_param_count.pdf')
    plt.close()

    # for each model type, plot and color
    for model_type in model_types:
        df_model_type = anes_df[anes_df.model_type == model_type]
        plt.scatter(df_model_type['mutual_inf'], df_model_type['accuracy'], label=model_type)
    plt.xlabel('mutual_inf')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(index)
    plt.savefig(f'plots/pa/{index}_accuracy_vs_mutual_inf.pdf')
    plt.close()

