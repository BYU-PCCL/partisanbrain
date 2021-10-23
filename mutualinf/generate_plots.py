import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import concordance index
from lifelines.utils import concordance_index

def get_data(file_name='data/plot_data.pkl'):
    '''
    Get plot data.
    '''
    return pd.read_pickle(file_name)

def get_models(df):
    '''
    Get models from data. (columns)
    '''
    return df.columns.values

def get_datasets(df):
    '''
    Get datasets from data. (index)
    '''
    return df.index.values

def scatter_plot(df, ax):
    '''
    Make a scatter plot of 'mutual_inf' vs 'accuracy'.
    '''
    x, y = df['mutual_inf'], df['accuracy']
    ax.scatter(x, y)
    # do linear regression and plot
    a, b = np.polyfit(x, y, 1)
    xlim = ax.get_xlim()
    x_linspace = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x, a*x+b, 'orange')
    ax.set_xlim(xlim)


def make_big_scatter(df, save_path='plots/big_scatter.pdf'):
    '''
    Make scatter plot of all data.
    '''
    models, datasets = get_models(df), get_datasets(df)
    n_models, n_datasets = len(models), len(datasets)
    fig, ax = plt.subplots(n_datasets, n_models, figsize=(n_models*3, n_datasets*3))
    # iterate through datasets and models
    for i, dataset in enumerate(datasets):
        min_y, max_y, = np.inf, -np.inf
        for j, model in enumerate(models):
            data = df.loc[dataset, model]
            scatter_plot(data, ax[i, j])
            # update min and max y from ylims
            min_y = min(min_y, ax[i, j].get_ylim()[0])
            max_y = max(max_y, ax[i, j].get_ylim()[1])
        # update the ylims for all models
        for j in range(n_models):
            ax[i, j].set_ylim(min_y, max_y)

    # on top row, set the model as the title
    for i, model in enumerate(models):
        ax[0, i].set_title(model)
    # on left column, set the dataset as the ylabel
    for j, dataset in enumerate(datasets):
        ax[j, 0].set_ylabel(dataset)
    # for all but left column, remove y scale
    for i in range(n_datasets):
        for j in range(1, n_models):
            ax[i, j].set_yticks([])
    plt.suptitle('Mutual Information vs Accuracy for each Model/Dataset')
    plt.savefig(save_path)
    plt.close()

def make_davinci_scatter(df, save_path='plots/davinci_scatter.pdf'):
    '''
    Make scatter plots for each dataset for 'gpt3-davinci'.
    '''
    datasets = get_datasets(df)
    # make a 4x2 scatter plot
    dims = (4, 2)
    fig, ax = plt.subplots(dims[0], dims[1], figsize=(dims[1]*3, dims[0]*3))
    for i, dataset in enumerate(datasets):
        ax_row, ax_col = i//dims[1], i%dims[1]
        data = df.loc[dataset, 'gpt3-davinci']
        scatter_plot(data, ax[ax_row, ax_col])
        ax[ax_row, ax_col].set_title(dataset)
    # for first column, add 'accuracy' as the ylabel
    for i in range(dims[0]):
        ax[i, 0].set_ylabel('accuracy')
    # for bottom row, add 'Mutual Information (nats)' as the xlabel
    for i in range(dims[1]):
        ax[-1, i].set_xlabel('Mutual Information (nats)')
    plt.suptitle('Mutual Information vs Accuracy for each Dataset with GPT-3')
    plt.savefig(save_path)
    plt.close()

def get_corrs(df):
    '''
    Get correlation matrix between accuracy and mutual information for all models. Return a df with the correlations.
    '''
    models, datasets = get_models(df), get_datasets(df)
    # make a df with the correlations
    corrs = pd.DataFrame(index=datasets, columns=models)
    for model in models:
        for dataset in datasets:
            data = df.loc[dataset, model]
            corrs.loc[dataset, model] = data['accuracy'].corr(data['mutual_inf'])
    return corrs

def get_concordance_index(df):
    '''
    Get the concordance index between accuracy and mutual information for all models. Return a df with the concordance index.
    '''
    models, datasets = get_models(df), get_datasets(df)
    # make a df with the concordance index
    ci = pd.DataFrame(index=datasets, columns=models)
    for model in models:
        for dataset in datasets:
            data = df.loc[dataset, model]
            ci.loc[dataset, model] = concordance_index(data['accuracy'].tolist(), data['mutual_inf'].tolist())
    return ci

def generate_all():
    '''
    Generate all plots.
    '''
    df = get_data()
    make_big_scatter(df)
    make_davinci_scatter(df)

if __name__ == '__main__':
    generate_all()
    df = get_data()
    print(get_corrs(df))
    print(get_concordance_index(df))