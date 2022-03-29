import numpy as np
from sklearn.model_selection import ParameterGrid
from generate_output import Generator
from lda_generate_output import Generator as LdaGenerator
from neuron_selection import select_neurons_per_layer, get_samples
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
        "pca_log_reg",
        "pca_correlation",
    ],
    "n_neurons": [100, 200, 500, 1000, 2000, 5000],
    "percentile": [50, 80, 100, "means"],
}
lda_hyperparams = {
    "selection_method": ["lda"],
    "layer_selection_method": ["correlation", "logistic_regression"],
    "layers": [1, 10, 25],
    "percentile": [50, 80, 100, "means"],
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
sample_info = get_samples()

perplexity_df = pd.read_csv("data/wiki.csv")

for params in ParameterGrid(hyperparams):
    neurons_per_layer = select_neurons_per_layer(
        n_neurons=params["n_neurons"],
        method=params["selection_method"],
        sample_info=sample_info,
        percentile=params["percentile"],
    )
    # Generate 1000 sentences
    generator = Generator(
        model,
        tokenizer,
        force_emotion=FORCE_EMOTION,
        percentile=params["percentile"],
        neurons_per_layer=neurons_per_layer,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )
    df = generator.generate_samples(
        prompt=prompt,
        n_sequences=1000,
        n_neurons=params["n_neurons"],
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
    lda_neuron_selector.percentile = params["percentile"]
    neurons_per_layer = lda_neuron_selector.get_lda_neurons_per_layer(
        layers=params["layers"],  # TODO Fix this
    )
    # Generate 1000 sentences
    generator = LdaGenerator(
        model,
        tokenizer,
        layers=params["layers"],
        force_emotion=FORCE_EMOTION,
        percentile=params["percentile"],
        neurons_per_layer=neurons_per_layer,
        force_with=selection_method_to_force_with[params["selection_method"]],
    )
    series = generator.generate_samples(
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
