from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm


N_NEURONS = 78400
MODEL_SHAPE = (49, 1600)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_neurons_per_layer(mask_filename):
    with open(mask_filename) as file:
        neurons_per_layer = json.load(file)

    neurons_per_layer = {int(k): v for k, v in neurons_per_layer.items()}

    return neurons_per_layer


def get_rand_neurons_per_layer():
    rand_neurons = np.random.randint(low=0, high=N_NEURONS, size=100)
    rand_neurons_per_layer = {}

    for neuron in rand_neurons:
        layer, neuron = np.unravel_index(neuron, MODEL_SHAPE)
        layer, neuron = int(layer), int(neuron)
        if layer in rand_neurons_per_layer:
            rand_neurons_per_layer[layer].append(neuron)
        else:
            rand_neurons_per_layer[layer] = [neuron]

    return rand_neurons_per_layer


def get_likelihood_sequence(input, log_probs):
    return [
        log_probs[:, i, token_index].cpu().detach().numpy()[0]
        for i, token_index in enumerate(input.squeeze()[1:])
    ]


def get_probs(data_filename, mask_filename):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    neurons_per_layer = get_neurons_per_layer(mask_filename)
    rand_neurons_per_layer = get_rand_neurons_per_layer()

    df = pd.read_csv(data_filename)

    # Use these to store the output token distributions
    masked_log_probs = []
    no_masked_log_probs = []
    rand_masked_log_probs = []

    # Use these to store the input likelihoods
    masked_likelihood = []
    no_masked_likelihood = []
    rand_masked_likelihood = []

    for i, row in tqdm(df.iterrows()):
        input = tokenizer.encode(row.sentence.strip(), return_tensors="pt")
        input = input.to(DEVICE)

        with torch.no_grad():
            output = model(input)
        log_probs = torch.nn.functional.softmax(output.logits, dim=0)
        no_masked_log_probs.append(log_probs[:, -1, :].detach().cpu().numpy())
        no_masked_likelihood.append(get_likelihood_sequence(input, log_probs))

        with torch.no_grad():
            output = model(input, neurons_per_layer=neurons_per_layer)
        log_probs = torch.nn.functional.softmax(output.logits, dim=0)
        masked_log_probs.append(log_probs[:, -1, :].detach().cpu().numpy())
        masked_likelihood.append(get_likelihood_sequence(input, log_probs))

        with torch.no_grad():
            output = model(input, neurons_per_layer=rand_neurons_per_layer)
        log_probs = torch.nn.functional.softmax(output.logits, dim=0)
        rand_masked_log_probs.append(log_probs[:, -1, :].detach().cpu().numpy())
        rand_masked_likelihood.append(get_likelihood_sequence(input, log_probs))

    return {
        "masked_log_probs": masked_log_probs,
        "no_masked_log_probs": no_masked_log_probs,
        "rand_masked_log_probs": rand_masked_log_probs,
        "masked_likelihood": masked_likelihood,
        "no_masked_likelihood": no_masked_likelihood,
        "rand_masked_likelihood": rand_masked_likelihood,
    }


if __name__ == "__main__":
    data_filename = "data/train_data_binary.csv"
    mask_filename = "neurons_per_layer.json"

    output_dict = get_probs(data_filename, mask_filename)

    np.savez("output/masked_log_probs.npz", *output_dict["masked_log_probs"])
    np.savez("output/no_masked_log_probs.npz", *output_dict["no_masked_log_probs"])
    np.savez("output/rand_masked_log_probs.npz", *output_dict["rand_masked_log_probs"])
    np.savez("output/masked_likelihood.npz", *output_dict["masked_likelihood"])
    np.savez("output/no_masked_likelihood.npz", *output_dict["no_masked_likelihood"])
    np.savez(
        "output/rand_masked_likelihood.npz", *output_dict["rand_masked_likelihood"]
    )
