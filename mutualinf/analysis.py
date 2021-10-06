import pandas as pd
import numpy as np

def read_df(path):
    '''
    Loads the dataframe at path.
    Path (str): 'csv' or 'pkl'
    Expected columns are:
        - 'template' (str): name of templatizing strategy
        - 'categories' (list): list of potential categories. For example, ['Trump', 'Clinton']
        - 'ground truth' (string): ground truth response. One of the categories in df['categories']
        - category1, category2, ... (float): each category will be a column in the df, corresponds to probability mass for each respones. Should sum to one across all categories.
    '''
    if path.endswith('csv'):
        df = pd.read_csv(path)
    elif path.endswith('pkl'):
        df = pd.read_pickle(path)
    else:
        raise ValueError('Unknown file type')
    # convert categories to list
    df['categories'] = eval(df['categories'].astype(str))
    return df

def calculate_accuracy(df):
    '''
    Calculates the accuracy of the model. Adds a column called 'accuracy' to df.
    df (pandas.DataFrame): dataframe with columns 'template', 'categories', and 'ground truth'

    Returns modified df.
    '''
    raise NotImplementedError

def calculate_mutual_info(df):
    '''
    Calculates the mutual information of the model. Adds a column called 'mutual info' to df.
    df (pandas.DataFrame): dataframe with columns 'template', 'categories', and 'ground truth'

    Returns modified df.
    '''
    raise NotImplementedError

# TODO - do correlation analysis between templates and ground truth. Add in functions for this?

if __name__ == '__main__':
    # read in 'example.csv'
    df = read_df('example.csv')
    # calculate accuracy
    df = calculate_accuracy(df)
    # calculate mutual info
    df = calculate_mutual_info(df)
    # TODO - correlation analysis between templates and ground truth.