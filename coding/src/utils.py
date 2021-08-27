import numpy as np
import pandas as pd

def pairwise_agreement(df):
    '''
    Calculate the pairwise agreement of a dataframe.
    Arguments:
        df (pd.DataFrame): rows are instances, columns are coders.
    Returns:
        agreement (pd.DataFrame): agreement matrix
    '''
    # empty matrix
    agreement = np.zeros((df.shape[1], df.shape[1]))
    # iterate over pairs of coders
    for i, coder1 in enumerate(df.columns):
        for j, coder2 in enumerate(df.columns):
            # calculate agreement
            agreement[i, j] = (df[coder1] == df[coder2]).mean()
    return pd.DataFrame(agreement, columns=df.columns, index=df.columns)


def joint_agreement(df):
    '''
    Calculate the percentage agreement between each column of the model.
    Arguments:
        df (pd.DataFrame): rows are instances, columns are coders.
    Returns:
        perc (float): average pairwise agreement
    '''
    # iterate over pairs of coders
    percs = []
    for i, coder1 in enumerate(df.columns):
        for coder2 in df.columns[i+1:]:
            # calculate agreement
            percs.append((df[coder1] == df[coder2]).mean())
    # average
    return np.mean(percs)

def ensemble(df):
    '''
    Given a dataframe of codings, ensemble the codings into a single
    coding.
    Arguments:
        df (pd.DataFrame): rows are instances, columns are coders.
    Returns:
        mode (pd.Series): the most common codings
    '''
    mode = df.mode(axis=1)[0]
    return mode
