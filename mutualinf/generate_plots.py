from collections import defaultdict
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
import warnings


# Color constants
MAIN_COLOR = "#4091c9"
HIGHLIGHT_COLOR = "#ef3c2d"


def get_summary(df, model_name):
    ds_dict = defaultdict(list)

    for idx, row in df.iterrows():
        ds_df = row[model_name]

        n_templates = len(ds_df.index)

        if n_templates != 20:
            warnings.warn(f"{idx} has {n_templates} templates (!= 20)")

        ds_dict["dataset"].extend([idx] * len(ds_df.index))
        ds_dict["template_name"].extend(ds_df.index.tolist())
        ds_dict["accuracy"].extend(ds_df["accuracy"])
        ds_dict["mutual_inf"].extend(ds_df["mutual_inf"])

    return pd.DataFrame(ds_dict)


def cover_plot(df):

    # For GPT-3 and each individual dataset, make a cluster of bars

    df = get_summary(df, "gpt3-davinci")

    ds_agg = df.groupby("dataset")["accuracy"].agg(["min",
                                                    "mean",
                                                    "median",
                                                    "max"])
    ds_agg.reset_index(level=0, inplace=True)
    print(ds_agg)

    ds_mi = df.groupby("dataset").apply(lambda x: x.nlargest(1, "mutual_inf"))
    ds_agg["mi_max"] = ds_mi["accuracy"].values

    plot_data = defaultdict(list)

    for _, row in ds_agg.iterrows():
        plot_data["dataset"] += [row["dataset"]] * 5
        plot_data["values"] += [row["min"], row["mean"], row["median"],
                                row["mi_max"], row["max"]]
        plot_data["hues"] += ["min", "mean", "median", "mi_max", "max"]

    plot_data = pd.DataFrame(plot_data)
    print(plot_data)

    # Make the plot
    colors = ["#9dcee2", "#4091c9", "#1368aa", "#ef3c2d", "#033270"]
    sns.set_palette(sns.color_palette(colors))
    sns.catplot(x="dataset", y="values", hue="hues",
                data=plot_data, kind="bar", saturation=1)
    plt.show()


def davinci_box_whisker(df):

    prepped_df = get_summary(df, "gpt3-davinci")

    sns.boxplot(x="accuracy", y="dataset", data=prepped_df, color=MAIN_COLOR, orient="h")

    mi_ds_dict = defaultdict(list)
    for ds_name in prepped_df["dataset"].unique().tolist():

        ds_part = prepped_df[prepped_df["dataset"] == ds_name]

        mi_ds_dict["dataset"].extend([ds_name for _ in range(1)])
        mi_cutoff = ds_part["mutual_inf"].nlargest(1).iloc[-1]
        mi_ds_dict["accuracy"].extend(ds_part[ds_part["mutual_inf"] >= mi_cutoff]["accuracy"])
    mi_ds = pd.DataFrame(mi_ds_dict)
    mi_ds = mi_ds.groupby("dataset").mean()
    mi_ds.reset_index(level=0, inplace=True)
    sns.swarmplot(x="accuracy", y="dataset", data=mi_ds, color=HIGHLIGHT_COLOR, size=10, alpha=0.5)
    plt.show()


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
            # add horizontal lines in at every tenth in the lims
            tenths = np.linspace(0, 1, 11)
            for t in tenths:
                if t > min_y and t < max_y:
                    ax[i, j].axhline(t, color='black', alpha=0.1)


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

def heatmap(df, save_path, scale_min=None, scale_max=None, title=None):
    '''
    Generate a heatmap.
    '''
    plt.figure(figsize=(10, 10))
    if scale_min is None:
        scale_min = df.min().min()
    if scale_max is None:
        scale_max = df.max().max()
    if title is not None:
        plt.title(title)
    # plt.imshow(df.values.astype(float), cmap='viridis', interpolation='nearest', vmin=scale_min, vmax=scale_max)
    # seaborn heatmap
    sns.heatmap(
        df.values.astype(float),
        cmap='viridis',
        vmin=scale_min,
        vmax=scale_max,
        annot=True,
        square=True,
        cbar=True
    )
    # columns
    plt.xticks(np.arange(len(df.columns))+0.5, df.columns, rotation=90)
    # index
    plt.yticks(np.arange(len(df.index)), df.index)
    plt.savefig(save_path)

def correlation_heatmap(df, save_path='plots/correlation_heatmap.pdf', scale_min=0, scale_max=1, title=None):
    '''
    Make a correlation heatmap.
    '''
    corrs = get_corrs(df)
    if title is None:
        title = 'Correlation between Accuracy and Mutual Information'
    heatmap(corrs, save_path, scale_min=scale_min, scale_max=scale_max, title=title)

def concordance_heatmap(df, save_path='plots/concordance_heatmap.pdf', scale_min=.5, scale_max=1, title=None):
    '''
    Make a concordance heatmap.
    '''
    concs = get_concordance_index(df)
    if title is None:
        title = 'Correlation between Concordance and Mutual Information'
    heatmap(concs, save_path, scale_min=scale_min, scale_max=scale_max, title=title)

def make_transfer_heatmap(df_mi, df_oracle, save_path=None, scale_min=0, scale_max=1, title=None):
    '''
    Make a plot showing transfer ability.
    '''
    # two plots, one for MI and one for Oracle
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # MI
    ax[0].set_title('Mutual Information')
    # seaborn heatmap on ax[0]
    sns.heatmap(
        df_mi.values.astype(float),
        cmap='viridis',
        vmin=scale_min,
        vmax=scale_max,
        annot=True,
        square=True,
        cbar=True,
        ax=ax[0],
    )
    ax[0].set_xlabel('Inference Model')
    ax[0].set_ylabel('Selection Model')
    # set xticks and yticks
    ax[0].set_xticks(np.arange(len(df_mi.columns))+0.5)
    ax[0].set_xticklabels(df_mi.columns, rotation=90)
    ax[0].set_yticks(np.arange(len(df_mi.index)))
    ax[0].set_yticklabels(df_mi.index, rotation=0)

    # oracle
    ax[1].set_title('Test Accuracy')
    # seaborn heatmap on ax[1]
    sns.heatmap(
        df_oracle.values.astype(float),
        cmap='viridis',
        vmin=scale_min,
        vmax=scale_max,
        annot=True,
        square=True,
        cbar=True,
        ax=ax[1],
    )
    ax[1].set_xlabel('Inference Model')
    # set just xticks
    ax[1].set_xticks(np.arange(len(df_oracle.columns))+0.5)

    # set sup title
    if title is None:
        title = 'Transferability'
    plt.suptitle(title)

    # save
    if save_path is None:
        save_path =  f'plots/transfer_heatmap_{dataset}.pdf'
    plt.savefig(save_path)
    plt.close()


def make_transfer_plots(df, save_dir='plots'):
    '''
    Make transfer plots for each dataset.
    '''
    datasets = get_datasets(df)
    transfer_oracle = get_transfer_oracle(df)
    transfer_mi = get_transfer_mutual_information(df)
    for dataset in datasets:
        make_transfer_heatmap(transfer_mi[dataset], transfer_oracle[dataset], save_path=f'{save_dir}/transfer_heatmap_{dataset}.pdf', title=f'Transferability for {dataset}')


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

def get_transfer_oracle(df):
    '''
    For each dataset, generate a graph across model sizes with accuracy by model size, after selecting prompty via oracle.
    '''
    datasets = get_datasets(df)
    models = get_models(df)
    # keep track of template selected by the oracle
    template_dict = {}
    for dataset in datasets:
        dataset_dict = {}
        for model in models:
            df_exp = df.loc[dataset, model]
            # sort by accuracy
            df_exp = df_exp.sort_values('accuracy', ascending=False)
            # get highest accuracy index
            dataset_dict[model] = df_exp.index[0]
        template_dict[dataset] = dataset_dict
    
    transfer_dict = {}
    for dataset in datasets:
        # make df with models as index and models as columns
        df_transfer = pd.DataFrame(index=models, columns=models)
        for inference_model in models:
            df_exp = df.loc[dataset, inference_model]
            for select_model in models:
                # get the best template for the select model
                template = template_dict[dataset][select_model]
                # score is accuracy, scaled to 0-1 between mean and best
                acc = df_exp.loc[template, 'accuracy']
                best_acc = df_exp['accuracy'].max()
                mean_acc = df_exp['accuracy'].mean()
                scaled_acc = (acc - mean_acc) / (best_acc - mean_acc)
                df_transfer.loc[select_model, inference_model] = scaled_acc
        transfer_dict[dataset] = df_transfer
    return transfer_dict

def get_transfer_mutual_information(df):
    '''
    For each dataset, generate a graph across model sizes with mutual information by model size, after selecting prompt by mutual information.
    '''
    datasets = get_datasets(df)
    models = get_models(df)
    # keep track of template selected by the oracle
    template_dict = {}
    for dataset in datasets:
        dataset_dict = {}
        for model in models:
            df_exp = df.loc[dataset, model]
            # sort by mutual information
            df_exp = df_exp.sort_values('mutual_inf', ascending=False)
            # get highest mutual information index
            dataset_dict[model] = df_exp.index[0]
        template_dict[dataset] = dataset_dict

    transfer_dict = {}
    for dataset in datasets:
        # make df with models as index and models as columns
        df_transfer = pd.DataFrame(index=models, columns=models)
        for inference_model in models:
            df_exp = df.loc[dataset, inference_model]
            for select_model in models:
                # get the best template for the select model
                template = template_dict[dataset][select_model]
                # score is accuracy, scaled to 0-1 between mean and best
                acc = df_exp.loc[template, 'accuracy']
                best_acc = df_exp['accuracy'].max()
                mean_acc = df_exp['accuracy'].mean()
                scaled_acc = (acc - mean_acc) / (best_acc - mean_acc) * 1
                df_transfer.loc[select_model, inference_model] = scaled_acc
        transfer_dict[dataset] = df_transfer
    return transfer_dict



def generate_all():
    '''
    Generate all plots.
    '''
    df = get_data()
    make_big_scatter(df)
    make_davinci_scatter(df)
    correlation_heatmap(df)
    concordance_heatmap(df)
    make_transfer_plots(df)

if __name__ == '__main__':
    # generate_all()
    df = get_data()
    # davinci_box_whisker(df)
    cover_plot(df)
    # print(get_corrs(df))
    # print(get_concordance_index(df))
