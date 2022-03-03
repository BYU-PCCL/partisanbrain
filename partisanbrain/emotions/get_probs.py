from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
from neuron_selection import select_neurons_per_layer, get_samples


N_NEURONS = 1000
MODEL_SHAPE = (49, 1600)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_likelihood_sequence(input, log_probs):
    return [
        log_probs[:, i, token_index].item()
        for i, token_index in enumerate(input.squeeze()[1:])
    ]


def get_probs(data_filename, n_neurons=N_NEURONS):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    sample_info = get_samples(data_filename)

    neurons_per_layer = select_neurons_per_layer(
        n_neurons=n_neurons, method="correlation", sample_info=sample_info
    )

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
        log_probs = torch.nn.functional.log_softmax(output.logits, dim=2)
        no_masked_log_probs.append(log_probs[:, -1, :].detach().cpu().numpy())
        no_masked_likelihood.append(get_likelihood_sequence(input, log_probs))

        with torch.no_grad():
            output = model(input, neurons_per_layer=neurons_per_layer)
        log_probs = torch.nn.functional.log_softmax(output.logits, dim=2)
        masked_log_probs.append(log_probs[:, -1, :].detach().cpu().numpy())
        masked_likelihood.append(get_likelihood_sequence(input, log_probs))

        # Recalculate the random neurons we mask each time
        rand_neurons_per_layer = select_neurons_per_layer(
            n_neurons=n_neurons, method="random", sample_info=sample_info
        )
        with torch.no_grad():
            output = model(input, neurons_per_layer=rand_neurons_per_layer)
        log_probs = torch.nn.functional.log_softmax(output.logits, dim=2)
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

    output_dict = get_probs(data_filename)

    np.savez("output/masked_log_probs.npz", *output_dict["masked_log_probs"])
    np.savez("output/no_masked_log_probs.npz", *output_dict["no_masked_log_probs"])
    np.savez("output/rand_masked_log_probs.npz", *output_dict["rand_masked_log_probs"])
    np.savez("output/masked_likelihood.npz", *output_dict["masked_likelihood"])
    np.savez("output/no_masked_likelihood.npz", *output_dict["no_masked_likelihood"])
    np.savez(
        "output/rand_masked_likelihood.npz", *output_dict["rand_masked_likelihood"]
    )
