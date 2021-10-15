import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_df(path):
    '''
    Loads the dataframe at path.
    Path (str): 'csv' or 'pkl'
    Expected columns are:
        - 'template_name' (str): name of templatizing strategy
        - 'ground_truth' (string): ground truth response
        - 'probs' (dict: str->float): dictionary of probabilities for each category/token.
    '''
    if path.endswith('csv'):
        df = pd.read_csv(path)
        # convert probs to dictionary
        df['probs'] = df.probs.apply(eval)
    elif path.endswith('pkl'):
        df = pd.read_pickle(path)
    else:
        raise ValueError('Unknown file type')
    return df

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
        if row['ground_truth'].startswith(guess):
            return 1
        else:
            return 0
    df['accuracy'] = df.apply(accuracy_lambda, axis=1)

    return df

def prob_dict_to_arr(d):
    '''
    Converts a probability dictionary into an array of probabilities.
    Args:
        d (dict: str->float): dictionary of probabilities for each category/token.
    Returns:
        arr (np.array): array of probabilities.
    '''
    arr = np.array([d[k] for k in d])
    return arr

def entropy(arr):
    '''
    Given an array of probabilities, calculate the entropy.
    '''
    return -sum(arr * np.log(arr))

entropy_lambda = lambda row: entropy(prob_dict_to_arr(row['probs']))

def calculate_conditional_entropy(df):
    '''
    Calculates the conditional entropy, up to a constant. Adds a column called 'conditional_entropy' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    # Calculate entropy for each row
    df['conditional_entropy'] = df.apply(entropy_lambda, axis=1)

    return df

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

def get_marginal_distribution(df, groupby='template_name'):
    '''
    Calculates the marginal distribution over categories.
    '''
    marginal_df = df.groupby(by=groupby)['probs'].agg(agg_prob_dicts)
    # series to df
    marginal_df = pd.DataFrame(marginal_df)
    return marginal_df

def calculate_mutual_information(df, groupby='template_name'):
    '''
    Calculate the mutual information between the template and the output distribution.
    '''
    # H(Y) - H(Y|X) method
    # first, calculate conditional entropy
    df = calculate_conditional_entropy(df)
    # get marginal distributions
    marginal_df = get_marginal_distribution(df, groupby)
    # get entropy
    marginal_df['entropy'] = marginal_df.apply(entropy_lambda, axis=1)
    # function to apply per row
    def mutual_inf(row):
        index = row[groupby]
        mutual_info = marginal_df.loc[index]['entropy'] - row['conditional_entropy']
        return mutual_info
    
    # apply function to each row
    df['mutual_inf'] = df.apply(mutual_inf, axis=1)

    return df


def calculate_correct_weight(df):
    '''
    Calculates the correct_weight. Adds a column called 'correct_weight' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    # Our function for calculating weight on ground truth
    get_correct_weight = lambda row: row['probs'].get(row['ground_truth'], 0)

    # Calculate conditional entropy for each row
    df['correct_weight'] = df.apply(get_correct_weight, axis=1)

    return df

# TODO - do correlation analysis between templates and ground truth. Add in functions for this?
def compare_per_template(df):
    group = df.groupby(by='template_name')

    output_df = group[['accuracy', 'mutual_inf']].agg(np.mean)

    corr = output_df.corr().iloc[0,1]

    x, y = output_df.mutual_inf.values, output_df.accuracy.values

    plt.scatter(
        x=x,
        y=y,
        alpha=0.7,
        s=50,
        edgecolors='none',
    )

    # fit linear regression
    lr = LinearRegression()
    a, b = lr.fit(x.reshape(-1,1), y).coef_[0], lr.intercept_
    # plot line
    x_linspace = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_linspace, a*x_linspace + b, 'C1', alpha=0.7)

    plt.title(f'Grouped by Template, Corr Coeff: {corr:.3f}')
    plt.xlabel(r'Mutual Information: $I(Y, f_{\theta}(X))$')
    plt.ylabel('Accuracy')

    return corr

def compare_per_response(df, y_jitter=.05):
    corr = df[['accuracy', 'mutual_inf']].corr().iloc[0,1]

    x, y = df.mutual_inf.values, df.accuracy.values

    plt.scatter(
        x=x,
        y=y + np.random.normal(0, y_jitter, len(y)),
        alpha=0.2,
        s=20,
        edgecolors='none',
    )

    # fit logistic regression
    lr = LogisticRegression()
    a, b = lr.fit(x.reshape(-1,1), y).coef_[0], lr.intercept_
    # plot sigmoid regression
    x_linspace = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_linspace, sigmoid(a*x_linspace + b), 'C1', alpha=0.7)

    plt.title(f'Entropy Difference vs. Response Accuracy, Corr Coeff: {corr:.3f}')
    plt.xlabel(r'Entropy Difference: $H(Y) - H(Y|f_{\theta}(x_i))$')
    plt.ylabel('Accuracy')

    return corr


def compare_per_response_weight(df):
    corr = df[['correct_weight', 'mutual_inf']].corr().iloc[0,1]

    x, y = df.mutual_inf.values, df.correct_weight.values

    plt.scatter(
        x=x,
        y=y,
        alpha=0.2,
        s=20,
        edgecolors='none',
    )

    # fit linear regression
    lr = LinearRegression()
    a, b = lr.fit(x.reshape(-1,1), y).coef_[0], lr.intercept_
    # plot line
    x_linspace = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_linspace, a*x_linspace + b, 'C1', alpha=0.7)

    plt.title(f'Weight of Correct Response, Corr Coeff: {corr:.3f}')
    plt.xlabel(r'Entropy Difference: $H(Y) - H(Y|f_{\theta}(x_i))$')
    plt.ylabel('Weight on Correct')


    return corr

def compare_per_idx(df):
    group = df.groupby(by='raw_idx')

    output_df = group[['accuracy', 'mutual_inf']].agg(np.mean)

    corr = output_df.corr().iloc[0,1]

    x, y = output_df.mutual_inf.values, output_df.accuracy.values

    plt.scatter(
        x=x,
        y=y,
        alpha=0.7,
        s=50,
        edgecolors='none',
    )

    # fit linear regression
    lr = LinearRegression()
    a, b = lr.fit(x.reshape(-1,1), y).coef_[0], lr.intercept_
    # plot line
    x_linspace = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_linspace, a*x_linspace + b, 'C1', alpha=0.7)


    plt.title(f'Grouped by Instance, Corr Coeff: {corr:.3f}')
    plt.xlabel(r'Mean Entropy Difference: $\mathbb{E}_{\theta}[H(Y) - H(Y|f_{\theta}(x_i))]$')
    plt.ylabel('Accuracy')

    return corr

def plot_comparisons(df, show=True, save=False, filename=None):
    """
    Calculates four different comparisons and shows or saves the results based
    on user input.
    """
    corrs = {}
    # # make figure big
    # plt.figure(figsize=(14,6))

    # plt.subplot(121)
    # corrs['per_template'] = compare_per_template(df)

    # plt.subplot(122)
    # corrs['per_response_weight'] = compare_per_response_weight(df)

    plt.figure(figsize=(14,8))
    plt.subplot(221)
    corrs['per_template'] = compare_per_template(df)

    plt.subplot(222)
    corrs['per_response'] = compare_per_response(df)

    plt.subplot(223)
    corrs['per_response_weight'] = compare_per_response_weight(df)

    plt.subplot(224)
    corrs['per_id'] = compare_per_idx(df)

    # make suptitle with dataset
    plt.suptitle(f'{df.dataset.unique()[0].upper()} - {df.model.unique()[0].upper()}')

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
        'accuracy': 'mean',
        'mutual_inf': 'mean',
        'coverage': 'mean',
        'prompt': 'first',
    })
    
    # sort by conditional entropy
    output_df = output_df.sort_values(by='mutual_inf', ascending=True)
    return output_df


if __name__ == '__main__':
    # first process
    from postprocessor import Postprocessor
    # df = pd.read_pickle('data/imdb/exp_results_gpt2.pkl')
    df = pd.read_pickle('data/imdb/exp_results.pkl')
    # df = pd.read_pickle('experiments/exp_results.pkl')
    df['token_sets'] = [{'positive': ['positive', 'good', 'happy', 'great', 'excellent'], 'negative': ['negative', 'bad', 'poor', 'sad', 'depressing']}] * len(df)
    df = df.dropna()
    postprocessor = Postprocessor(df)
    df = postprocessor.df

    # calculate mutual information
    df = calculate_mutual_information(df)
    # calculate accuracy
    df = calculate_accuracy(df)
    # calculate correct weight
    df = calculate_correct_weight(df)

    templates = get_sorted_templates(df)
    print(templates)
    # calculate mutual information

    # get model and dataset
    model = df.model.unique()[0]
    dataset = df.dataset.unique()[0]
    # save to plots/dataset_model
    plot_comparisons(df, save=True, filename=f'plots/{dataset}_{model}.png')
    pass