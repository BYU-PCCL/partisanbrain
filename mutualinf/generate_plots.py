from collections import defaultdict
from ensembling_utils import prep_all_ensembling_data
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
import cmocean

import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import warnings
from pdb import set_trace as breakpoint


# Color constants
MAIN_COLOR = "#4091c9"
HIGHLIGHT_COLOR = "#ef3c2d"

BLUE_1 = "#9dcee2"  # Min
BLUE_2 = "#4091c9"  # Mean
BLUE_3 = "#1368aa"  # Median
RED_1 = "#ef3c2d"  # Ours
BLUE_4 = "#033270"  # Max

# times new roman
plt.rcParams["font.family"] = "Times"

x_text_rotate = 40
y_text_rotate = 0
# cmap = 'Blues'
cmap = 'RdBu'


def make_ensembling_kde_plot(save_fname="plots/ensembling_kde_plot.pdf"):

    # Make sure ensembling data has been gathered
    prep_all_ensembling_data()

    # Generate plot
    dss = {"rocstories": "ROCStories",
           "squad": "SQuAD",
           "common_sense_qa": "CommonsenseQA",
           "boolq": "BoolQ",
           "imdb": "IMDB",
           "anes": "ANES",
           "copa": "COPA",
           "wic": "WiC"}

    # Make a grid of plots
    n_rows = 2
    n_cols = 4
    sub_dim = 2.5
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(n_cols*sub_dim, n_rows*sub_dim))

    # For each dataset in ds_names add a plot to the grid
    for i, ds_name in enumerate(sorted(dss.keys())):

        data_fname = f"ensembling_data/{ds_name}.pkl"

        with open(data_fname, "rb") as f:
            ensembling_data = pickle.load(f)

        plt_ax = axes[i // n_cols, i % n_cols]

        kde_vals = np.array(ensembling_data["kde_vals"])

        sns.kdeplot(x=kde_vals,
                    fill=True,
                    color=BLUE_1,
                    ax=plt_ax,
                    shade=False)

        kdeline = plt_ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()

        # Get index of first True in ys > 0.01
        idx = np.where(ys > 0.25)[0][0]
        low_bound = xs[idx-1]

        # Get index of first True in reversed(ys) > 0.01
        idx = np.where(ys[::-1] > 0.25)[0][0]
        high_bound = xs[-idx+1]

        plt_ax.axvline(x=ensembling_data["avg_acc"], color=BLUE_2)
        plt_ax.axvline(x=ensembling_data["ensemble_acc"], color=BLUE_4)
        plt_ax.axvline(x=ensembling_data["top_k_acc"], color=RED_1)
        plt_ax.fill_between(kdeline.get_xdata(),
                            0,
                            kdeline.get_ydata(),
                            facecolor=BLUE_1,
                            alpha=0.2)

        plt_ax.set(xlim=(low_bound, high_bound))
        plt_ax.yaxis.label.set_visible(False)
        plt_ax.set_title(dss[ds_name])

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none",
                    which="both",
                    top=False,
                    bottom=False,
                    left=False,
                    right=False)
    plt.xlabel("Accuracy")
    plt.ylabel("Density")

    fig.tight_layout(h_pad=0.5, w_pad=0)

    plt.savefig(save_fname, bbox_inches="tight")


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


def cover_plot(df, save_path='plots/cover_plot.pdf'):

    # For GPT-3 and each individual dataset, make a cluster of bars

    datasets = get_datasets(df)
    # df = get_summary(df, "gpt3-davinci")
    # df = get_summary(df, "175B (GPT-3)")
    df = get_summary(df, "GPT-3: 175B")

    ds_agg = df.groupby("dataset")["accuracy"].agg(["min",
                                                    "mean",
                                                    "median",
                                                    "max"])
    # order
    ds_agg = ds_agg.loc[datasets]
    ds_agg.reset_index(level=0, inplace=True)

    ds_mi = df.groupby("dataset").apply(lambda x: x.nlargest(1, "mutual_inf"))
    ds_mi = ds_mi.loc[datasets]
    ds_agg["mi_max"] = ds_mi["accuracy"].values

    plot_data = defaultdict(list)

    for _, row in ds_agg.iterrows():
        plot_data["dataset"] += [row["dataset"]] * 5
        plot_data["values"] += [row["min"], row["mean"], row["median"],
                                row["mi_max"], row["max"]]
        # plot_data["hues"] += ["min", "mean", "mediar", "mi_max", "max"]
        plot_data["hues"] += ["Min", "Mean", "Median", "MI Choice", "Max"]

    plot_data = pd.DataFrame(plot_data)

    # Make the plot
    # size of plot
    # TWO COLUMN PLOT
    # height, aspect = 5, 2.5
    # ONE COLUMN PLOT
    height, aspect = 4.5, 1.35

    # get axis for big plot
    colors = [BLUE_1, BLUE_2, BLUE_3, RED_1, BLUE_4]
    sns.set_palette(sns.color_palette(colors))
    sns.catplot(
        x="dataset",
        y="values",
        hue="hues",
        data=plot_data,
        kind="bar",
        saturation=1,
        height=height,
        aspect=aspect,
        legend=False,
        row_order=get_datasets(df),
    )
    # bottom right
    plt.legend(loc="lower right")
    # rotate xticks, right justification
    plt.xticks(rotation=x_text_rotate, ha="right")
    # ylabel accuracy
    plt.ylabel('Accuracy')
    # xlabel none
    plt.xlabel('')
    plt.title('Mutual Information performance on GPT-3 Davinci')
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


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
    plt.close()

def box_whisker(df, dataset, orientation='v', absolute_scaling=False, ax=False, save=True):
    '''
    Make a box and whisker plot of 'mutual_inf' vs 'accuracy' for a given dataset.
    '''
    models = get_models(df)
    df_ds = df.loc[dataset]
    acc_models = []
    accs = []
    # acc_lists = [df_ds[model]['accuracy'].astype(float).tolist() for model in models]
    mutual_inf_accs = []
    for model in models:
        df_exp = df_ds[model]
        # get max mutualinf
        index = df_exp['mutual_inf'].idxmax()
        mutual_inf_accs.append(df_exp.loc[index]['accuracy'].astype(float))
        # add accs
        accs.extend(df_exp['accuracy'].astype(float).tolist())
        acc_models.extend([model] * len(df_exp['accuracy'].astype(float).tolist()))
    if not ax:
        ax = plt.gca()
    if orientation == 'v':
        sns.boxplot(
            x = acc_models,
            y = accs,
            color = MAIN_COLOR,
            orient = "v",
            showfliers = True,
            ax = ax,
        )
        sns.swarmplot(
            x = models,
            y = mutual_inf_accs,
            color = HIGHLIGHT_COLOR,
            size = 10,
            alpha = 0.5,
            ax = ax,
        )
        if absolute_scaling:
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1])
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        # rotate xticks, right justification
        ax.set_xticklabels(models, rotation=x_text_rotate, ha="right")
    elif orientation == 'h':
        sns.boxplot(
            x = accs,
            y = acc_models,
            color = MAIN_COLOR,
            orient = "h",
            showfliers = True,
            ax = ax,
        )
        sns.swarmplot(
            x = mutual_inf_accs,
            y = models,
            color = HIGHLIGHT_COLOR,
            size = 10,
            alpha = 0.5,
            ax = ax,
        )
        if absolute_scaling:
            xlim = ax.get_xlim()
            ax.set_xlim(0, xlim[1])
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Model")
        ax.set_yticklabels(models, rotation=y_text_rotate, ha="right")
    else:
        raise ValueError('orientation must be "v" or "h"')
    ax.set_title(dataset)
    if save:
        path = f'plots/box_whisker_{dataset}.pdf'
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()

def make_all_box_whisker(df, orientation='v', absolute_scaling=False):
    datasets = get_datasets(df)
    for dataset in datasets:
        box_whisker(df, dataset, orientation, absolute_scaling)

def make_grouped_box_whisker(df, orientation='v', absolute_scaling=False):
    datasets = get_datasets(df)
    # shape of subplot grid
    grid_shape = (2, 4)
    # make figure
    fig, axs = plt.subplots(
        nrows=grid_shape[0],
        ncols=grid_shape[1],
        figsize=(grid_shape[1] * 2.5, grid_shape[0] * 2.5),
    )

    for ax, dataset in zip(axs.flatten(), datasets):
        box_whisker(df, dataset, orientation, absolute_scaling, ax=ax, save=False)
    if orientation == 'v':
        # turn off xticks and xlabel on first row
        for ax in axs[0]:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        # turn off ylabel on all but first column
        for col in range(1, grid_shape[1]):
            for ax in axs[:, col]:
                ax.set_ylabel('')
        # turn off xlabel on last row
        for ax in axs[-1]:
            ax.set_xlabel('')
    elif orientation == 'h':
        # turn off yticks and ylabel on all but first column
        for col in range(1, grid_shape[1]):
            for ax in axs[:, col]:
                ax.set_yticklabels([])
                ax.set_ylabel('')
        # turn off xlabel on top row
        for ax in axs[0]:
            ax.set_xlabel('')
                
    plt.tight_layout()
    plt.savefig('plots/grouped_box_whisker.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


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
    fig, ax = plt.subplots(n_datasets, n_models, figsize=(n_models*2, n_datasets*2))
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
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def make_davinci_scatter(df, save_path='plots/davinci_scatter.pdf'):
    '''
    Make scatter plots for each dataset for 'gpt3-davinci'.
    '''
    datasets = get_datasets(df)
    # make a 4x2 scatter plot
    dims = (4, 2)
    fig, ax = plt.subplots(dims[0], dims[1], figsize=(dims[1]*2.5, dims[0]*2.5))
    for i, dataset in enumerate(datasets):
        ax_row, ax_col = i//dims[1], i%dims[1]
        # data = df.loc[dataset, 'gpt3-davinci']
        # data = df.loc[dataset, '175B (GPT-3)']
        data = df.loc[dataset, 'GPT-3: 175B']
        scatter_plot(data, ax[ax_row, ax_col])
        ax[ax_row, ax_col].set_title(dataset)
    # for first column, add 'accuracy' as the ylabel
    for i in range(dims[0]):
        ax[i, 0].set_ylabel('accuracy')
    # for bottom row, add 'Mutual Information (nats)' as the xlabel
    for i in range(dims[1]):
        ax[-1, i].set_xlabel('Mutual Information (nats)')
    plt.suptitle('Mutual Information vs Accuracy for each Dataset with GPT-3')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def heatmap(df, save_path, scale_min=None, scale_max=None, title=None, override_cmap=None, round=2):
    '''
    Generate a heatmap.
    '''
    plt.figure(figsize=(6, 6))
    if scale_min is None:
        scale_min = df.min().min()
    if scale_max is None:
        scale_max = df.max().max()
    if title is not None:
        plt.title(title)
    # plt.imshow(df.values.astype(float), cmap=cmap, interpolation='nearest', vmin=scale_min, vmax=scale_max)
    # seaborn heatmap
    data = df.values.astype(float)
    # round
    data = np.round(data, round)
    sns.heatmap(
        data,
        cmap=cmap if override_cmap is None else override_cmap,
        vmin=scale_min,
        vmax=scale_max,
        annot=True,
        square=True,
        cbar=True,
        cbar_kws={'shrink': 0.68},
    )
    # columns
    plt.xticks(np.arange(len(df.columns))+0.5, df.columns, rotation=x_text_rotate, ha='right')
    # index
    plt.yticks(np.arange(len(df.index))+0.5, df.index, rotation=y_text_rotate, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def correlation_heatmap(df, save_path='plots/correlation_heatmap.pdf', scale_min=-1, scale_max=1, title=None):
    '''
    Make a correlation heatmap.
    '''
    corrs = get_corrs(df)
    if title is None:
        title = 'Correlation between MI and Accuracy'
    heatmap(corrs, save_path, scale_min=scale_min, scale_max=scale_max, title=title)

def concordance_heatmap(df, save_path='plots/concordance_heatmap.pdf', scale_min=0, scale_max=1, title=None):
    '''
    Make a concordance heatmap.
    '''
    concs = get_concordance_index(df)
    if title is None:
        title = 'Concordance between MI and Accuracy'
    heatmap(concs, save_path, scale_min=scale_min, scale_max=scale_max, title=title)

def combined_corr_conc_heatmap(df, save_path='plots/corr_conc_heatmap.pdf'):
    '''
    Make a combined correlation and concordance heatmap.
    '''
    # correlation
    corrs = get_corrs(df)
    corrs = corrs.values.astype(float)
    # round corrs
    corrs = np.round(corrs, 2)

    # concordance
    concs = get_concordance_index(df)
    concs = concs.values.astype(float)
    # round concs
    concs = np.round(concs, 2)
    # make a 2x1 grid
    fig, ax = plt.subplots(2, 1, figsize=(5.5, 8))


    # correlation heatmap
    sns.heatmap(
        corrs,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=True,
        square=True,
        cbar=True,
        cbar_kws={'shrink': 1},
        ax=ax[0],
    )
    # set title
    ax[0].set_title('Correlation between MI and Accuracy')
    # rows are datasets
    ax[0].set_yticks(np.arange(len(df.index))+0.5)
    ax[0].set_yticklabels(df.index, rotation=y_text_rotate, ha='right')
    # xticks off
    ax[0].set_xticks([])

    # concordance heatmap
    sns.heatmap(
        concs,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        square=True,
        cbar=True,
        cbar_kws={'shrink': 1},
        ax=ax[1],
    )
    # set title
    ax[1].set_title('Concordance between MI and Accuracy')

    # columns
    plt.xticks(np.arange(len(df.columns))+0.5, df.columns, rotation=x_text_rotate, ha='right')
    # index
    plt.yticks(np.arange(len(df.index))+0.5, df.index, rotation=y_text_rotate, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def make_transfer_heatmap(df_mi, df_oracle, save_path=None, scale_min=-1, scale_max=1, title=None, round=2):
    '''
    Make a plot showing transfer ability.
    '''
    # two plots, one for MI and one for Oracle
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.75))
    # MI
    ax[0].set_title('Mutual Information')
    # seaborn heatmap on ax[0]
    data = df_mi.values.astype(float)
    # round
    data = np.round(data, round)
    sns.heatmap(
        data,
        cmap=cmap,
        vmin=scale_min,
        vmax=scale_max,
        annot=True,
        square=True,
        cbar=True,
        ax=ax[0],
        cbar_kws={'shrink': .88},
    )
    ax[0].set_xlabel('Inference Model')
    ax[0].set_ylabel('Selection Model')
    # set xticks and yticks
    ax[0].set_xticks(np.arange(len(df_mi.columns))+0.5)
    ax[0].set_xticklabels(df_mi.columns, rotation=x_text_rotate, ha='right')
    ax[0].set_yticks(np.arange(len(df_mi.index))+0.5)
    ax[0].set_yticklabels(df_mi.index, rotation=y_text_rotate, ha='right')

    # oracle
    ax[1].set_title('Test Accuracy')
    # seaborn heatmap on ax[1]
    data = df_oracle.values.astype(float)
    # round
    data = np.round(data, round)
    sns.heatmap(
        data,
        cmap=cmap,
        vmin=scale_min,
        vmax=scale_max,
        annot=True,
        square=True,
        cbar=True,
        ax=ax[1],
        cbar_kws={'shrink': 0.88},
    )
    ax[1].set_xlabel('Inference Model')
    # set just xticks
    ax[1].set_xticks(np.arange(len(df_oracle.columns))+0.5)
    ax[1].set_xticklabels(df_oracle.columns, rotation=x_text_rotate, ha='right')
    # turn off yticks
    ax[1].set_yticks([])

    # set sup title
    if title is None:
        title = 'Transferability'
    plt.suptitle(title)

    # tight layout
    plt.tight_layout()

    # save
    if save_path is None:
        save_path =  f'plots/transfer_heatmap_{dataset}.pdf'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
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

def make_average_transfer_heatmap(df, save_dir='plots'):
    '''
    Make average transfer heatmap, averaged across datasets.
    '''
    datasets = get_datasets(df)
    transfer_oracle = get_transfer_oracle(df)
    transfer_mi = get_transfer_mutual_information(df)
    # average together
    transfer_oracle_avg = sum([transfer_oracle[dataset] for dataset in datasets])/len(datasets)
    transfer_mi_avg = sum([transfer_mi[dataset] for dataset in datasets])/len(datasets)
    make_transfer_heatmap(
        transfer_mi_avg,
        transfer_oracle_avg,
        save_path=f'{save_dir}/transfer_heatmap_average.pdf',
        title='Transferability (Averaged over Datasets)'
    )

def make_average_transfer_difference_heatmap(df, save_dir='plots'):
    datasets = get_datasets(df)
    transfer_oracle = get_transfer_oracle(df)
    transfer_mi = get_transfer_mutual_information(df)
    # average together
    transfer_oracle_avg = sum([transfer_oracle[dataset] for dataset in datasets])/len(datasets)
    transfer_mi_avg = sum([transfer_mi[dataset] for dataset in datasets])/len(datasets)
    diff = transfer_mi_avg - transfer_oracle_avg
    # make diagonal 0
    diff.values[np.diag_indices_from(diff)] = 0
    heatmap(diff,
        save_path=f'{save_dir}/transfer_heatmap_difference.pdf',
        title='Transferability Difference (Averaged over Datasets)',
        scale_min=-.5,
        scale_max=.5,
        override_cmap='RdBu',
    )
    
def normalize_accs(df):
    '''
    For each model and dataset pair, normalize the accuracies to [0, 1], where 0 is the average prompt and 1 is the max accuracy.
    '''
    normed_accs = df.copy()
    models, datasets = get_models(df), get_datasets(df)
    for model in models:
        for dataset in datasets:
            # df.loc[dataset, model] = (df.loc[dataset, model] - df.loc[dataset, model].mean()) / (df.loc[dataset, model].max() - df.loc[dataset, model].min())
            normed_accs.loc[dataset, model]['accuracy'] = (normed_accs.loc[dataset, model]['accuracy'] - normed_accs.loc[dataset, model]['accuracy'].mean()) / (normed_accs.loc[dataset, model]['accuracy'].max() - df.loc[dataset, model]['accuracy'].min())
    return normed_accs

def plot_normalized_accs(df, save_path='plots/normalized_accs.pdf'):
    '''
    plot with model being x-axis, normalized accuracy is y-axis, and dataset is color
    '''
    df = df.copy()
    datasets, models = get_datasets(df), get_models(df)
    norm = normalize_accs(df)
    plt.figure(figsize=(10, 10))
    for i, dataset in enumerate(datasets):
        accs = []
        # get just the top mutual information accuracy for each model
        for model in models:
            df_exp = norm.loc[dataset, model]
            arg_max = df_exp['mutual_inf'].idxmax()
            accs.append(df_exp.loc[arg_max, 'accuracy'])
        plt.plot(models, accs, label=dataset, color=f'C{i}')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_acc_diffs(df, save_path='plots/acc_diffs.pdf'):
    '''
    plot with model being x-axis, accuracy boost is y-axis, and dataset is color
    '''
    datasets, models = get_datasets(df), get_models(df)
    plt.figure(figsize=(10, 10))
    for i, dataset in enumerate(datasets):
        accs = []
        # get just the top mutual information accuracy for each model
        for model in models:
            df_exp = df.loc[dataset, model]
            arg_max = df_exp['mutual_inf'].idxmax()
            accs.append(df_exp.loc[arg_max, 'accuracy'] - df_exp['accuracy'].mean())
        plt.plot(models, accs, label=dataset, color=f'C{i}')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_models_vs_mi_gain(df, save_path='plots/models_v_mi_gain.pdf'):
    '''
    plot with model being x-axis, and 3 lines on the y-axis:
        1. Average concordance index (with error bars)
        2. Average accuracy boost (with error bars)
        3. Average normalized accuracy boost (with error bars)
    '''
    datasets, models = get_datasets(df), get_models(df)
    plt.figure(figsize=(12, 10))
    # ylim 0 to 1
    plt.ylim(0, 1)

    # plot average concordance index
    ci = get_concordance_index(df)
    ci_mean, ci_std = ci.mean(), ci.std()
    # plot with error bars
    plt.errorbar(models, ci_mean, yerr=ci_std, label='Average Concordance Index', color='C0')

    # plot average accuracy boost
    acc_boost_means, acc_boost_stds = [], []
    for model in models:
        acc_boosts = []
        for dataset in datasets:
            df_exp = df.loc[dataset, model]
            arg_max = df_exp['mutual_inf'].idxmax()
            acc_boosts.append(df_exp.loc[arg_max, 'accuracy'] - df_exp['accuracy'].mean())
        acc_boost_means.append(np.mean(acc_boosts))
        acc_boost_stds.append(np.std(acc_boosts))
    # plot with error bars
    plt.errorbar(models, acc_boost_means, yerr=acc_boost_stds, label='Average Accuracy Boost', color='C1')

    # plot average normalized accuracy boost
    normed_acc_boost_means, normed_acc_boost_stds = [], []
    df_norm = normalize_accs(df)
    for model in models:
        normed_acc_boosts = []
        for dataset in datasets:
            df_exp = df_norm.loc[dataset, model]
            arg_max = df_exp['mutual_inf'].idxmax()
            normed_acc_boosts.append(df_exp.loc[arg_max, 'accuracy'])
        normed_acc_boost_means.append(np.mean(normed_acc_boosts))
        normed_acc_boost_stds.append(np.std(normed_acc_boosts))
    # plot with error bars
    plt.errorbar(models, normed_acc_boost_means, yerr=normed_acc_boost_stds, label='Average Normalized Accuracy Boost', color='C2')

    plt.legend()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
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
    make_average_transfer_heatmap(df)
    make_average_transfer_difference_heatmap(df)
    make_all_box_whisker(df)
    make_grouped_box_whisker(df, orientation='v')
    cover_plot(df)
    combined_corr_conc_heatmap(df)

if __name__ == '__main__':
    # make_big_scatter(get_data())
    # generate_all()
    make_average_transfer_heatmap(get_data())
    # generate_all()
    # make_ensembling_kde_plot()
