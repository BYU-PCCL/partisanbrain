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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FORCE_EMOTION = "negative"

hyperparams = {
    "selection_method": [
        "correlation",
        "logistic_regression",
        "pca_correlation",
        "pca_log_reg",
    ],
    "n_neurons": [100, 200, 500, 1000, 5000],
    "percentile": [0.5, 0.8, 1],
}
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
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.to(DEVICE)
model.eval()
prompt = "I watched a new movie yesterday. I thought it was"

# Read in the data
output = np.load("output/output.npz")

X = output["activations"]
y = output["targets"].squeeze()

perplexity_df = pd.read_csv("data/wiki.csv")

neuron_selector = NeuronSelector(input_filename="output/output.npz", device=DEVICE)
for params in ParameterGrid(hyperparams):
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
    classifier = SentimentClassifier(df)
    classifier.classify_sentiment()
    value_counts = df.get_value_counts()

    # Measure the perplexity
    perplexity_analyzer = PerplexityAnalyzer(model, tokenizer, df=perplexity_df)
    average_perplexity = perplexity_analyzer.get_average_perplexity(
        neurons_per_layer=neurons_per_layer,
        force_emotion=FORCE_EMOTION,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )

lda_neuron_selector = LdaNeuronSelector(filename="output/output.npz", device=DEVICE)
for params in ParameterGrid(lda_hyperparams):
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
    classifier = SentimentClassifier(df)
    classifier.classify_sentiment()
    value_counts = df.get_value_counts()

    # Measure the perplexity
    perplexity_analyzer = PerplexityAnalyzer(model, tokenizer, df=perplexity_df)
    average_perplexity = perplexity_analyzer.get_average_perplexity(
        neurons_per_layer=neurons_per_layer,
        force_emotion=FORCE_EMOTION,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )
