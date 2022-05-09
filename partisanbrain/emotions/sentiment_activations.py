import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def harvest_activations(model, tokenizer, filename):
    NUM_LAYERS = model.config.n_layer + 1
    activations = []
    targets = []

    df = pd.read_csv(filename)

    for i, row in df.iterrows():
        label, sentence = row.label, row.sentence
        encoded_input = tokenizer(sentence.strip(), return_tensors="pt")
        encoded_input.to(DEVICE)
        output = model(**encoded_input, output_hidden_states=True)

        row_activations = []
        for n_layer in range(NUM_LAYERS):
            row_activations.append(
                np.atleast_2d(
                    output.hidden_states[n_layer][0, -1, :].detach().cpu().numpy()
                )
            )
        row_activations = np.vstack(row_activations)
        activations.append(row_activations)
        targets.append(np.atleast_2d(label))

    activations = np.array(activations)  # instance, layer, neuron
    targets = np.vstack(targets)

    return activations, targets


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2Model.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    filename = "data/train_data_binary.csv"

    with torch.no_grad():
        activations, targets = harvest_activations(
            model=model,
            tokenizer=tokenizer,
            filename=filename,
        )

    np.savez(
        "output/sentiment_activations.npz", activations=activations, targets=targets
    )
