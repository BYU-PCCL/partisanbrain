from ensemble import ensemble, get_accuracies
from generate_plots import BLUE_1, BLUE_2, BLUE_4, RED_1
from itertools import combinations
from math import comb

import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
import tqdm


def make_davinci_ensemble_plot():
    pass


def prep_all_ensembling_data():

    save_dir = "ensembling_data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    proc_fnames = glob.glob("data/*/*_processed.pkl")

    # Only davinci
    davinci_proc_fnames = [f for f in proc_fnames if "davinci" in f]

    # Skip old_boolq
    valid_fnames = [f for f in davinci_proc_fnames if "old_boolq" not in f]

    for fname in valid_fnames:

        ds_name = fname.split("/")[1]
        save_fname = f"{save_dir}/{ds_name}.pkl"

        save_ensembling_data(fname, save_fname)


def plot_from_ensembling_data(data_fname):

    with open(data_fname, "rb") as f:
        ensembling_data = pickle.load(f)

    print(len(ensembling_data["kde_vals"]))

    p = sns.kdeplot(x=ensembling_data["kde_vals"], fill=True, color=BLUE_1)
    p.set_xlabel("Accuracy")

    plt.axvline(x=ensembling_data["avg_acc"], color=BLUE_2)
    plt.axvline(x=ensembling_data["ensemble_acc"], color=BLUE_4)
    plt.axvline(x=ensembling_data["top_k_acc"], color=RED_1)

    plt.savefig("saved.pdf", bbox_inches="tight")


def save_ensembling_data(in_fname, out_fname):

    df = pd.read_pickle(in_fname)

    ensembling_data = dict()

    # Get average accuracy for each template_name in df
    template_accs = df.groupby("template_name").mean()["accuracy"]

    kde_vals = []
    idx = template_accs.index

    for c in tqdm.tqdm(combinations(idx, 5), total=comb(len(idx), 5)):
        five_templates = df[df["template_name"].isin(c)]
        five_acc = ensemble(five_templates)["accuracy"].mean()
        kde_vals.append(five_acc)

    avg_acc, ensemble_acc, top_k_acc = get_accuracies(df)

    ensembling_data["kde_vals"] = kde_vals
    ensembling_data["avg_acc"] = avg_acc
    ensembling_data["ensemble_acc"] = ensemble_acc
    ensembling_data["top_k_acc"] = top_k_acc

    # Save ensembling data with pickle
    with open(out_fname, "wb") as f:
        pickle.dump(ensembling_data, f)


if __name__ == "__main__":
    prep_all_ensembling_data()

    # import pandas as pd

    # save_ensembling_data(in_fname="exp_results_gpt3-davinci_23-10-2021_processed.pkl",
    #                      out_fname="temp.pkl")
    # plot_from_ensembling_data("ensembling_data/imdb.pkl")
