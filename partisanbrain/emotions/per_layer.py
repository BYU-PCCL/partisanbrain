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

torch.manual_seed(0)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FORCE_EMOTION = "negative"
MODEL_SHAPE = (49, 1600)

params = {
    "selection_method": "pca_correlation",
    "n_neurons": 10,
    "percentile": 0.8,
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
output = np.load("output/sentiment_activations.npz")

X = output["activations"]
y = output["targets"].squeeze()

perplexity_df = pd.read_csv("data/wiki.csv")
perplexity_analyzer = PerplexityAnalyzer(model, tokenizer, df=perplexity_df)

results = []
neuron_selector = NeuronSelector(
    input_filename="output/sentiment_activations.npz", device=DEVICE
)
for layer in range(MODEL_SHAPE[0]):
    neurons_per_layer = neuron_selector.get_neurons_per_layer(
        n_neurons=params["n_neurons"],
        percentile=params["percentile"],
        method=params["selection_method"],
        layer=layer,
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

    output_filename = f"output/hyperparam/dfs/{params['selection_method']}_{params['n_neurons']}_{int(params['percentile'] * 100)}_{params['layer']}.csv"
    classifier.mod_df.to_csv(output_filename, index=False)

    res_dict = {
        **params,
        "n_negative": n_negative,
        "average_perplexity": average_perplexity,
        "output_filename": output_filename,
    }

    results.append(res_dict)
    pd.DataFrame(results).to_csv("output/hyperparam/layer_results.csv", index=False)
