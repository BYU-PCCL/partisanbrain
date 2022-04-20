import numpy as np
from sklearn.model_selection import ParameterGrid
from generate_output import Generator
from neuron_selection import NeuronSelector
from lda_neuron_selection import LdaNeuronSelector
from sentiment_analysis import SentimentClassifier
from transformers import GPT2Tokenizer
from gpt2 import GPT2LMHeadModel
import torch
from perplexity import PerplexityAnalyzer
import pandas as pd
from tqdm import tqdm
import argparse
import os
import label_toxicity

args = argparse.ArgumentParser()
args.add_argument("-a", "--activations", type=str)
args.parse_args()

attribute = os.path.split(args.activations)[-1].split(".")[0]
activation_filename = args.activations


torch.manual_seed(0)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FORCE_EMOTION = "negative"

# We have to structure it like this to optimize our runtime
hyperparams = [
    {
        "selection_method": [
            "correlation",
        ],
        "n_neurons": [100, 200, 500, 1000, 5000],
        "percentile": [0.5, 0.8, 1],
    },
    {
        "selection_method": [
            "logistic_regression",
        ],
        "n_neurons": [100, 200, 500, 1000, 5000],
        "percentile": [0.5, 0.8, 1],
    },
    {
        "selection_method": [
            "pca_correlation",
        ],
        "n_neurons": [1, 2, 5, 10, 20, 50],
        "percentile": [0.5, 0.8, 1],
    },
    {
        "selection_method": [
            "pca_log_reg",
        ],
        "n_neurons": [1, 2, 5, 10, 20, 50],
        "percentile": [0.5, 0.8, 1],
    },
]
lda_hyperparams = {
    "selection_method": ["lda"],
    "layer_selection_method": ["correlation", "logistic_regression"],
    "n_layers": [1, 10, 25, 49],
    "percentile": [0.5, 0.8, 1],
}

selection_method_to_force_with = {
    "correlation": "per_neuron",
    "logistic_regression": "per_neuron",
    "pca_log_reg": "transform",
    "pca_correlation": "transform",
    "lda": "projection",
}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
tokenizer.padding_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.to(DEVICE)
model.eval()
prompt = "I watched a new movie yesterday. I thought it was"

# Read in the data
output = np.load(activation_filename)

X = output["activations"]
y = output["targets"].squeeze()

perplexity_df = pd.read_csv("data/wiki.csv")
perplexity_analyzer = PerplexityAnalyzer(model, tokenizer, df=perplexity_df)

results = []
neuron_selector = NeuronSelector(input_filename=activation_filename, device=DEVICE)
for params in tqdm(ParameterGrid(hyperparams)):
    if not params["selection_method"].startswith("pca"):
        neuron_selector.set_samples(X=X, y=y)
    neurons_per_layer = neuron_selector.get_neurons_per_layer(
        n_neurons=params["n_neurons"],
        percentile=params["percentile"],
        method=params["selection_method"],
    )

    # Generate 1000 sentences
    generator = Generator(
        model,
        tokenizer,
        force_emotion=FORCE_EMOTION,
        neurons_per_layer=neurons_per_layer,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )
    df = generator.generate_samples(
        prompt=prompt,
        n_sequences=1000,
    )

    # Classify the sentiment of the generated sentences
    classifier = SentimentClassifier(df=df)
    classifier.classify_sentiment()
    value_counts = classifier.get_value_counts()
    n_negative = value_counts.NEGATIVE

    # Measure the perplexity
    average_perplexity = perplexity_analyzer.get_average_perplexity(
        neurons_per_layer=neurons_per_layer,
        force_emotion=FORCE_EMOTION,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )

    output_dir = os.path(f"output/hyperparam/{attribute}/dfs")
    if not os.path.exists(output_dir):
        os.path.mkdir(output_dir)
    output_filename = f"{params['selection_method']}_{params['n_neurons']}_{int(params['percentile'] * 100)}.csv"
    output_fp = os.path.join(output_dir, output_filename)
    classifier.mod_df.to_csv(output_fp, index=False)

    res_dict = {
        **params,
        "n_negative": n_negative,
        "average_perplexity": average_perplexity,
        "output_filename": output_filename,
    }

    results.append(res_dict)
    pd.DataFrame(results).to_csv(
        f"output/hyperparam/{attribute}/results.csv", index=False
    )

results = []
lda_neuron_selector = LdaNeuronSelector(filename=activation_filename, device=DEVICE)
for params in tqdm(ParameterGrid(lda_hyperparams)):
    neurons_per_layer = lda_neuron_selector.get_lda_neurons_per_layer(
        n_layers=params["n_layers"],
        percentile=params["percentile"],
        method=params["layer_selection_method"],
    )

    # Generate 1000 sentences
    generator = Generator(
        model,
        tokenizer,
        force_emotion=FORCE_EMOTION,
        neurons_per_layer=neurons_per_layer,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )
    df = generator.generate_samples(
        prompt=prompt,
        n_sequences=1000,
    )

    # Classify the sentiment of the generated sentences
    # classifier = SentimentClassifier(df=df)
    # classifier.classify_sentiment()
    # value_counts = classifier.get_value_counts()
    scored_df = label_toxicity.score_df(df, attribute.upper())
    n_negative = scored_df.label.value_counts()[0]

    # Measure the perplexity
    average_perplexity = perplexity_analyzer.get_average_perplexity(
        neurons_per_layer=neurons_per_layer,
        force_emotion=FORCE_EMOTION,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )

    output_dir = f"output/hyperparam/dfs/"
    if not os.path.exists(output_dir):
        os.path.mkdir(output_dir)

    output_filename = f"{params['selection_method']}_{params['layer_selection_method']}_{params['n_layers']}_{int(params['percentile'] * 100)}.csv"
    output_fp = os.path.join(output_dir, output_filename)

    scored_df.to_csv(output_fp, index=False)

    res_dict = {
        **params,
        "n_negative": n_negative,
        "average_perplexity": average_perplexity,
        "output_filename": output_filename,
    }

    results.append(res_dict)
    pd.DataFrame(results).to_csv(
        f"output/hyperparam/{attribute}/lda_results.csv", index=False
    )
