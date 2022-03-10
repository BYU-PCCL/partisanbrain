import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np


def harvest_activations(model, tokenizer, danger_filename, safe_filename):
    NUM_LAYERS = model.config.n_layer + 1
    activations = []
    targets = []

    with open(danger_filename, "r") as danger_file:
        for line in danger_file.readlines():
            encoded_input = tokenizer(line.strip(), return_tensors="pt")
            # encoded_input.to("cuda:0")
            output = model(**encoded_input, output_hidden_states=True)

            danger_activations = []
            for n_layer in range(NUM_LAYERS):
                danger_activations.append(
                    np.atleast_2d(
                        output.hidden_states[n_layer][0, -1, :].detach().cpu().numpy()
                    )
                )
            danger_activations = np.vstack(danger_activations)
            activations.append(danger_activations)
            targets.append(np.ones(shape=(1, 1)))

    with open(safe_filename, "r") as safe_file:
        for line in safe_file.readlines():
            encoded_input = tokenizer(line.strip(), return_tensors="pt")
            # encoded_input.to("cuda:0")
            output = model(**encoded_input, output_hidden_states=True)

            safe_activations = []
            for n_layer in range(NUM_LAYERS):
                safe_activations.append(
                    np.atleast_2d(
                        output.hidden_states[n_layer][0, -1, :].detach().cpu().numpy()
                    )
                )
            safe_activations = np.vstack(safe_activations)
            activations.append(safe_activations)
            targets.append(np.zeros(shape=(1, 1)))

    activations = np.array(activations)  # instance, layer, neuron
    targets = np.vstack(targets)

    return activations, targets


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2Model.from_pretrained("gpt2-xl")
    model.eval()

    danger_filename = "data/danger.txt"
    safe_filename = "data/safe.txt"

    with torch.no_grad():
        activations, targets = harvest_activations(
            model=model,
            tokenizer=tokenizer,
            danger_filename=danger_filename,
            safe_filename=safe_filename,
        )

    # Save the activations and targets to run a regression on them later
    np.savez("output/output.npz", activations=activations, targets=targets)
