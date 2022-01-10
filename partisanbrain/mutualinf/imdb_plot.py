import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# "data/imdb/exp_results_gpt3-davinci_23-10-2021_processed.pkl"
df = pd.read_pickle("data/imdb/imdb_post_token_set_processed.pkl")


# Get mean mutual_inf and accuracy for each template_name in df
df_mean = df.groupby("template_name").mean()

# Reset index
df_mean = df_mean.reset_index()

# Make a scatterplot of mutual_inf and accuracy
plt.scatter(df_mean["mutual_inf"], df_mean["accuracy"])
# sns.lmplot(x="mutual_inf", y="accuracy", data=df_mean)
plt.show()
