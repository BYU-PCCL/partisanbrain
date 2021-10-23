import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

def agg_prob_dicts(dicts):
    '''
    Given a list of probability dictionaries, aggregate them.
    '''
    n = len(dicts)
    agg_dict = {}
    for d in dicts:
        for k, v in d.items():
            if k not in agg_dict:
                agg_dict[k] = v / n
            else:
                agg_dict[k] += v / n
    return agg_dict

def calculate_accuracy(df):
    '''
    Calculates the accuracy of the model. Adds a column called 'accuracy' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    # if row['ground_truth'] starts with argmax(row['probs']) stripped and lowercase, then it's correct
    def accuracy_lambda(row):
        # guess is argmax of row['probs'] dict
        guess = max(row['probs'], key=row['probs'].get)
        # lower and strip
        guess = guess.lower().strip()
        if row['ground_truth'].lower().strip().startswith(guess):
            return 1
        else:
            return 0
    df['accuracy'] = df.apply(accuracy_lambda, axis=1)

    return df

def ensemble(df):
    '''
    Aggregate 'probs' column with agg_prob_dicts, then fill in accuracy column.
    '''
    # groupby 'raw_idx'
    df_grouped = df.groupby('raw_idx')
    # aggregate 'probs' column with agg_prob_dicts, and 'ground_truth' column with first
    df_agg = df_grouped.agg({'probs': agg_prob_dicts, 'ground_truth': lambda x: x.iloc[0]})
    df_agg = calculate_accuracy(df_agg)
    return df_agg

def get_sorted_templates(df):
    group = df.groupby(by='template_name')

    # agg accuracy and conditional entropy by mean, and prompt by first
    output_df = group.agg({
        'accuracy': 'mean',
        'mutual_inf': 'mean',
        'coverage': 'mean',
        'prompt': 'first',
    })
    
    # sort by conditional entropy
    output_df = output_df.sort_values(by='mutual_inf', ascending=True)
    return output_df

def get_accuracies(df):
    '''
    Returns a list of accuracies.
        First: average accuracy of all prompts
        Second: accuracy of ensemble of all prompts
        Third: accuracy of top 5 mutual information prompts
    '''
    # first
    avg_acc = df['accuracy'].mean()
    # second
    ensemble_acc = ensemble(df)['accuracy'].mean()
    # third
    templates = get_sorted_templates(df)
    top_k_templates = templates.iloc[-5:].index.to_list()
    # filter df to only include top 5 templates
    df_top_k = df[df['template_name'].isin(top_k_templates)]
    top_k_acc = ensemble(df_top_k)['accuracy'].mean()
    return [avg_acc, ensemble_acc, top_k_acc]



if __name__ == '__main__':
    # first argument is the path to the data
    data_path = sys.argv[1]
    df = pd.read_pickle(data_path)
    accs = get_accuracies(df)
    print(f'Average accuracy: {accs[0]}')
    print(f'Ensemble accuracy: {accs[1]}')
    print(f'Top 5 ensemble accuracy: {accs[2]}')