from collections import defaultdict

import os
import pandas as pd


# Load IMDB GPT-3 results
df = pd.read_pickle("data/imdb/exp_results_gpt3-davinci_23-10-2021.pkl")

template_token_sets = defaultdict(set)
template_example = dict()

for idx, row in df.iterrows():

    template_name = row["template_name"]

    if template_name not in template_example.keys():
        template_example[template_name] = row["prompt"]

    # Extract tokens from GPT-3 response
    resp_tokens = [k.strip().lower() for k in row["resp"].keys()]
    template_token_sets[template_name].update(resp_tokens)

# Make save directory if does not exist
if not os.path.exists("imdb_exp_sets"):
    os.makedirs("imdb_exp_sets")

for template_name, tokens in template_token_sets.items():
    with open(f"imdb_exp_sets/{template_name}.txt", "w") as f:
        f.write(template_example[template_name])
        f.write("\n")
        f.write("-" * 50)
        f.write("\n")
        f.write("\n".join(tokens))
