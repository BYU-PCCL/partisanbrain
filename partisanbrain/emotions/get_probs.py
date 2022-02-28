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


def get_mask_per_layer(mask_filename):
    with open(mask_filename) as file:
        mask_per_layer = json.load(file)

    mask_per_layer = {int(k): v for k, v in mask_per_layer.items()}

    return mask_per_layer


def get_rand_mask_per_layer():
    rand_neurons = np.random.randint(low=0, high=N_NEURONS, size=100)
    rand_mask_per_layer = {}

    for neuron in rand_neurons:
        layer, neuron = np.unravel_index(neuron, MODEL_SHAPE)
        layer, neuron = int(layer), int(neuron)
        if layer in rand_mask_per_layer:
            rand_mask_per_layer[layer].append(neuron)
        else:
            rand_mask_per_layer[layer] = [neuron]

    return rand_mask_per_layer


def get_probs(data_filename, mask_filename):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    mask_per_layer = get_mask_per_layer(mask_filename)
    rand_mask_per_layer = get_rand_mask_per_layer()

    df = pd.read_csv(data_filename)

    masked_log_probs = []
    no_masked_log_probs = []
    rand_masked_log_probs = []

    for i, row in tqdm(df.iterrows()):
        input = tokenizer.encode(row.sentence.strip(), return_tensors="pt")
        input.to(DEVICE)

        with torch.no_grad():
            output = model(input)
        log_probs = torch.nn.functional.softmax(output.logits, dim=0)
        no_masked_log_probs.append(log_probs.detach().cpu().numpy())

        with torch.no_grad():
            output = model(input, mask_per_layer=mask_per_layer)
        log_probs = torch.nn.functional.softmax(output.logits, dim=0)
        masked_log_probs.append(log_probs.detach().cpu().numpy())

        with torch.no_grad():
            output = model(input, mask_per_layer=rand_mask_per_layer)
        log_probs = torch.nn.functional.softmax(output.logits, dim=0)
        rand_masked_log_probs.append(log_probs.detach().cpu().numpy())

    masked_log_probs = np.vstack(masked_log_probs)
    no_masked_log_probs = np.vstack(rand_masked_log_probs)
    rand_masked_log_probs = np.vstack(rand_masked_log_probs)

    return masked_log_probs, no_masked_log_probs, rand_masked_log_probs


if __name__ == "__main__":
    data_filename = "data/train_data_binary.csv"
    mask_filename = "mask_per_layer.json"

    masked_log_probs, no_masked_log_probs, rand_masked_log_probs = get_probs(
        data_filename, mask_filename
    )
    np.savez("output/masked_log_probs.npz", *masked_log_probs)
    np.savez("output/no_masked_log_probs.npz", *no_masked_log_probs)
    np.savez("output/rand_masked_log_probs.npz", *rand_masked_log_probs)
