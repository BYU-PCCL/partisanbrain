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
        - 'ground_truth' (string): ground truth response. One of the categories in df['categories']
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
    df (pandas.DataFrame): dataframe with columns 'template_name', 'categories', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    # Get the possible target values
    y = df.categories.iloc[0]

    # Calculate the accuracy for each row
    df['accuracy'] = (df[y].idxmax(axis=1) == df.ground_truth).astype(int)

    return df

def calculate_conditional_entropy(df):
    '''
    Calculates the conditional entropy, up to a constant. Adds a column called 'conditional_entropy' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', 'categories', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    # Get the possible target values
    y = df.categories.iloc[0]

    # Our function for calculating conditional entropy
    get_cond_entropy = lambda row: -sum([row[y_i] * np.log(row[y_i]) for y_i in y])

    # Calculate conditional entropy for each row
    df['conditional_entropy'] = [get_cond_entropy(row) for _, row in df.iterrows()]

    return df

def get_marginal_distribution(df, groupby='template_name'):
    '''
    Calculates the marginal distribution over categories.
    '''
    columns_to_keep = df['categories'].iloc[0]
    marginal_df = df.groupby(by=groupby)[columns_to_keep].agg('mean')
    return marginal_df

def KL_divergence(p, q):
    '''
    Calculates the KL divergence between two probability distributions.
    '''
    return (p * np.log(p / q)).sum()

def calculate_mutual_information(df, groupby='template_name'):
    '''
    Calculate the mutual information between the template and the output distribution.
    '''
    # TODO - verify correctness? Estimated with KL divergence?
    marginal_df = get_marginal_distribution(df, groupby)
    # function to apply per row
    def mutual_inf(row):
        categories = row.categories
        marginal_dist_index = row[groupby]
        marginal_dist = marginal_df.loc[marginal_dist_index].values
        dist = row[categories].astype(float).values
        # calculate KL divergence
        divergence = KL_divergence(dist, marginal_dist)
        return divergence
    
    # apply function to each row
    df['mutual_inf'] = df.apply(mutual_inf, axis=1)
    return df


def calculate_correct_weight(df):
    '''
    Calculates the correct_weight. Adds a column called 'correct_weight' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', 'categories', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    # Our function for calculating weight on ground truth
    get_correct_weight = lambda row: row[row.ground_truth]

    # Calculate conditional entropy for each row
    df['correct_weight'] = [get_correct_weight(row) for _, row in df.iterrows()]

    return df

# TODO - do correlation analysis between templates and ground truth. Add in functions for this?
def compare_per_template(df):
    group = df.groupby(by='template_name')

    output_df = group[['accuracy', 'mutual_inf']].agg(np.mean)

    corr = output_df.corr().iloc[0,1]

    plt.scatter(
        x=output_df.mutual_inf,
        y=output_df.accuracy,
        alpha=0.7,
        s=50,
        edgecolors='none',
    )
    plt.title(f'Grouped by Template, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Information: $H(Y|X)$')
    plt.ylabel('Accuracy')

    return corr


def compare_per_response(df):
    corr = df[['accuracy', 'mutual_inf']].corr().iloc[0,1]

    plt.scatter(
        x=df.mutual_inf,
        y=df.accuracy,
        alpha=0.7,
        s=20,
        edgecolors='none',
    )
    plt.title(f'Mutual Information vs. Response Accuracy, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Information: $H(Y|X)$')
    plt.ylabel('Accuracy')

    return corr

def compare_per_response_weight(df):
    corr = df[['correct_weight', 'mutual_inf']].corr().iloc[0,1]

    plt.scatter(
        x=df.mutual_inf,
        y=df.correct_weight,
        alpha=0.3,
        s=20,
        edgecolors='none',
    )
    plt.title(f'Mutual Information vs. Weight of Correct Response, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Information: $H(Y|X)$')
    plt.ylabel('Weight on Correct')

    return corr

def compare_per_prompt(df):
    group = df.groupby(by='respondant')

    output_df = group[['accuracy', 'mutual_inf']].agg(np.mean)

    corr = output_df.corr().iloc[0,1]

    plt.scatter(
        x=output_df.mutual_inf,
        y=output_df.accuracy,
        alpha=0.7,
        s=50,
        edgecolors='none',
    )
    plt.title(f'Grouped by Respondant, Corr Coeff: {corr:.3f}')
    plt.xlabel('Mutual Information: $H(Y|X)$')
    plt.ylabel('Accuracy')

    return corr

def plot_comparisons(df, show=True, save=False, filename=None):
    """
    Calculates four different comparisons and shows or saves the results based
    on user input.
    """
    corrs = {}
    # make figure big
    plt.figure(figsize=(14,8))
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

def get_sorted_templates(df):
    group = df.groupby(by='template_name')

    # agg accuracy and conditional entropy by mean, and prompt by first
    output_df = group.agg({
        'accuracy': np.mean,
        'mutual_inf': np.mean,
        'prompt': 'first',
    })
    
    # sort by conditional entropy
    output_df = output_df.sort_values(by='mutual_inf', ascending=True)
    return output_df


if __name__ == '__main__':
    # read in 'example.csv'
    # df = read_df('example.csv')
    # read in 'data/exp_results.csv'
    df = read_df('data/exp_results.csv')
    df = calculate_mutual_information(df)
    
    # TODO - fix, this is very patchy
    # get number of unique templates
    n_templates = len(df.template_name.unique())
    n_respondants = len(df) // n_templates
    # make a new column called "respondant" that is the same number repeated n_templates times
    respondant = np.arange(n_respondants).reshape(-1,1)
    # repeat n_templates times
    respondant = np.repeat(respondant, n_templates, axis=0).reshape(-1)
    df['respondant'] = respondant
    # calculate accuracy
    df = calculate_accuracy(df)
    # calculate conditional entropy
    df = calculate_conditional_entropy(df)
    # calculate correct weight
    df = calculate_correct_weight(df)

    templates = get_sorted_templates(df)
    print(templates)
    breakpoint()
    # calculate mutual information

    plot_comparisons(df, save=True, filename='plots/experiment1.pdf')
