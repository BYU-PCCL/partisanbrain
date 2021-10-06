import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def read_df(path):
    '''
    Loads the dataframe at path.
    Path (str): 'csv' or 'pkl'
    Expected columns are:
        - 'template_name' (str): name of templatizing strategy
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
    df['categories'] = df.categories.astype(str).apply(eval)
    return df

def calculate_accuracy(df):
    '''
    Calculates the accuracy of the model. Adds a column called 'accuracy' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', 'categories', and 'ground truth'

    Returns modified df.
    '''
    df = df.copy()

    # Get the possible target values
    y = df.categories.iloc[0]

    # Calculate the accuracy for each row
    df['accuracy'] = (df[y].idxmax(axis=1) == df.ground_truth).astype(int)

    return df

def calculate_mutual_info(df):
    '''
    Calculates the mutual information, up to a constant. Adds a column called 'mutual info' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', 'categories', and 'ground truth'

    Returns modified df.
    '''
    df = df.copy()

    # Get the possible target values
    y = df.categories.iloc[0]

    # Our function for calculating conditional entropy
    h = lambda row: - sum([ row[y_i] * np.log(row[y_i]) for y_i in y])

    # Calculate conditional entropy for each row
    df['mutual_info'] = [h(row) for _, row in df.iterrows()]

    return df

def calculate_correct_weight(df):
    '''
    Calculates the correct_weight. Adds a column called 'correct_weight' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', 'categories', and 'ground truth'

    Returns modified df.
    '''
    df = df.copy()

    # Our function for calculating weight on ground truth
    get_correct_weight = lambda row: row[f'{row.ground_truth}']

    # Calculate conditional entropy for each row
    df['correct_weight'] = [get_correct_weight(row) for _, row in df.iterrows()]

    return df

# TODO - do correlation analysis between templates and ground truth. Add in functions for this?
def compare_per_template(df):
    group = df.groupby(by='template_name')

    output_df = group[['accuracy', 'mutual_info']].agg(np.mean)

    corr = output_df.corr().iloc[0,1]

    plt.scatter(
        x=output_df.mutual_info,
        y=output_df.accuracy,
        alpha=0.7,
        s=50,
        edgecolors='none',
    )
    plt.title(f'Grouped by Template, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Info')
    plt.ylabel('Accuracy')

    return corr

def compare_per_response(df):
    corr = df[['accuracy', 'mutual_info']].corr().iloc[0,1]

    plt.scatter(
        x=df.mutual_info,
        y=df.accuracy,
        alpha=0.7,
        s=20,
        edgecolors='none',
    )
    plt.title(f'Mutual Info vs. Response Accuracy, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Info')
    plt.ylabel('Accuracy')

    return corr

def compare_per_response_weight(df):
    corr = df[['correct_weight', 'mutual_info']].corr().iloc[0,1]

    plt.scatter(
        x=df.mutual_info,
        y=df.correct_weight,
        alpha=0.3,
        s=20,
        edgecolors='none',
    )
    plt.title(f'Mutual Info vs. Weight of Correct Response, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Info')
    plt.ylabel('Accuracy')

    return corr

def compare_per_prompt(df):
    group = df.groupby(by='prompt')

    output_df = group[['accuracy', 'mutual_info']].agg(np.mean)

    corr = output_df.corr().iloc[0,1]

    plt.scatter(
        x=output_df.mutual_info,
        y=output_df.accuracy,
        alpha=0.7,
        s=50,
        edgecolors='none',
    )
    plt.title(f'Grouped by Prompt, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Info')
    plt.ylabel('Accuracy')

    return corr

def plot_comparisons(df, show=True, save=False, filename=None):
    """
    Calculates four different comparisons and shows or saves the results based
    on user input.
    """
    corrs = {}
    plt.subplot(221)
    corrs['per_template'] = compare_per_template(df)

    plt.subplot(222)
    corrs['per_response'] = compare_per_response(df)

    plt.subplot(223)
    corrs['per_response_weight'] = compare_per_response_weight(df)

    plt.subplot(224)
    corrs['per_prompt'] = compare_per_prompt(df)

    plt.tight_layout()

    if save:
        if filename is None:
            raise ValueError('filename needs to be specified if save is True')
        plt.savefig(filename)
    if show:
        plt.show()

    return corrs


if __name__ == '__main__':
    # read in 'example.csv'
    df = read_df('example.csv')
    # calculate accuracy
    df = calculate_accuracy(df)
    # calculate mutual info
    df = calculate_mutual_info(df)
    # calculate correct weight
    df = calculate_correct_weight(df)

    plot_comparisons(df)
