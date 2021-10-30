from ensemble import ensemble, get_accuracies
from generate_plots import get_data, BLUE_1, BLUE_2, BLUE_4, RED_1
from itertools import combinations
from math import comb
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm


def get_ensemble_acc(five_templates):
    return ensemble(five_templates)["accuracy"].mean()


def ensembling_barplot(df, parallel=False):

    # Get average accuracy for each template_name in df
    template_accs = df.groupby("template_name").mean()["accuracy"]

    kde_vals = []
    idx = template_accs.index

    # Use all available CPUs in parallel
    if parallel:
        p = Pool()
        df_subsets = [df[df["template_name"].isin(c)] for c in combinations(idx, 5)]
        kde_vals = list(tqdm.tqdm(p.imap(get_ensemble_acc, df_subsets), total=comb(len(idx), 5)))
    else:
        for c in tqdm.tqdm(combinations(idx[:15], 5), total=comb(len(idx[:15]), 5)):
            five_templates = df[df["template_name"].isin(c)]
            five_acc = ensemble(five_templates)["accuracy"].mean()
            kde_vals.append(five_acc)

    sns.kdeplot(x=kde_vals, fill=True, color=BLUE_1)

    avg_acc, ensemble_acc, top_k_acc = get_accuracies(df)

    plt.axvline(x=avg_acc, color=BLUE_2)
    plt.axvline(x=ensemble_acc, color=BLUE_4)
    plt.axvline(x=top_k_acc, color=RED_1)

    # Save plot
    plt.savefig("saved.pdf", bbox_inches="tight")


if __name__ == "__main__":
    import pandas as pd
    import sys
    data_path = "exp_results_gpt3-davinci_23-10-2021_processed.pkl"  # sys.argv[1]
    df = pd.read_pickle(data_path)
    ensembling_barplot(df)
