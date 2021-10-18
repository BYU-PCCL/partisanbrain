import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

    # unique_template_names = list(set(df.template_name.values))
    # template_name_map = dict(zip(unique_template_names,
    #                          range(len(unique_template_names))))

    plt.scatter(
        x=x,
        y=y,
        alpha=0.2,
        s=20,
        edgecolors='none',
        # c=[template_name_map[v] for v in df.template_name.values]
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
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, help='file with results to use')
    # get dataset arg
    args = parser.parse_args()
    results_file = args.results

    df = pd.read_pickle(args.results)

    templates = get_sorted_templates(df)
    print(templates)
    # calculate mutual information

    # get model and dataset
    model = df.model.unique()[0]
    dataset = df.dataset.unique()[0]
    # make 'plots' if missing
    if not os.path.exists('plots'):
        os.mkdir('plots')
    # save to plots/dataset_model
    model = model.replace('/', '-')
    plot_comparisons(df, save=True, filename=f'plots/{dataset}_{model}.png')
    pass
