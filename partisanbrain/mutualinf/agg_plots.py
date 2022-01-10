from collections import defaultdict

import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DS_NAMES = {
    "imdb": "IMDB",
}


def get_model_date(fname):
    name = fname.split("/")[-1]
    name = name.replace("processed_exp_results_", "")
    name = name.replace(".pkl", "")
    model_name, date = name.split("_")
    return model_name, date


def get_processed_exps(dir_name, keep_duplicates=True):
    fnames = glob.glob(f"data/{dir_name}/processed_exp_results_*.pkl")
    model_date_pairs = [get_model_date(f) for f in fnames]
    dfs = [pd.read_pickle(f) for f in fnames]
    exps = defaultdict(list)
    for i in range(len(dfs)):
        model, date = model_date_pairs[i]
        exps[model].append((date, dfs[i]))

    if not keep_duplicates:
        # deduplicated_exps = defaultdict(list)
        for k, v in exps.keys():
            if len(v) > 1:
                msg = ("More than one experiment "
                       f"result found for {k}. We don't have a way "
                       "to handle that yet.")
                raise NotImplementedError(msg)

    return exps


def get_df_acc_stats(df):

    # Choose template name with best mean
    acc_mns = df.groupby("template_name")["accuracy"].mean()

    # Choose template with best mutual information
    a = df.groupby("template_name").mean()
    mi = a[a["mutual_inf"] == a["mutual_inf"].max()]["accuracy"].values[0]

    return acc_mns.min(), acc_mns.mean(), acc_mns.median(), acc_mns.max(), mi


def make_agg_plot(exp_map):

    plot_data = defaultdict(list)

    for model_name in exp_map.keys():
        l, m, md, h, mia = get_df_acc_stats(exp_map[model_name][0][1])
        plot_data["model_name"] += [model_name] * 5
        plot_data["values"] += [l, m, md, mia, h]
        plot_data["hues"] += ["l", "m", "md", "mia", "h"]

    plot_data = pd.DataFrame(plot_data)
    print(plot_data)

    # Make the plot
    # colors = ["#4BA6AA", "#BDBDBC", "#FBE8D4", "#FE3D63", "#8E3759"]
    # colors = list(reversed(["#F94144", "#F8961E", "#90BE6D", "#43AA8B", "#577590"]))  # "#F3722C"
    # colors = list(reversed(["#aacc00", "#80b918", "#55a630", "#2b9348", "#007f5f"]))
    colors = ["#9dcee2", "#4091c9", "#1368aa", "#ef3c2d", "#033270"]
    # colors = ["#cccccc", "#a5a5a5", "#7f7f7f", "#e5383b", "#595959"]
    sns.set_palette(sns.color_palette(colors))
    sns.catplot(x="model_name", y="values", hue="hues", data=plot_data, kind="bar", saturation=1)
    plt.show()

    # import numpy as np
    # # sns.set_palette(sns.color_palette("Blues"))
    # uniform_data = np.random.rand(10, 12)
    # sns.heatmap(uniform_data, cmap=sns.color_palette("Blues", as_cmap=True))
    # plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="Name of directory of dataset to make plots for")
    args = parser.parse_args()

    exp_map = get_processed_exps(args.dataset)
    make_agg_plot(exp_map)
