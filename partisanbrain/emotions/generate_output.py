from neuron_selection import select_neurons_per_layer
from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import torch


N_NEURONS = 1000
BATCH_SIZE = 100
N_SEQUENCES = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def convert_to_text(self, output):
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def write_output(self, filename, outputs, labels):
        sentences = [self.convert_to_text(output) for output in outputs]
        df = pd.DataFrame(
            data=np.array([labels, sentences]).T, columns=["label", "sentence"]
        )
        df.to_csv(filename, index=False)

    def generate(self, input, n_sequences=N_SEQUENCES, neurons_per_layer=None):
        outputs = self.model.generate(
            input,
            max_length=30,
            do_sample=True,
            num_return_sequences=n_sequences,
            early_stopping=True,
            neurons_per_layer=neurons_per_layer,
        )
        return outputs

    def generate_samples(self, prompt, n_sequences=N_SEQUENCES):
        neurons_per_layer = select_neurons_per_layer(
            n_neurons=N_NEURONS, method="correlation"
        )

        input = self.tokenizer.encode(prompt.strip(), return_tensors="pt")
        input = input.to(DEVICE)

        normal_list = []
        altered_list = []

        n_batches = n_sequences // BATCH_SIZE
        gen_sequences = min(n_sequences, BATCH_SIZE)

        for i in range(n_batches):
            normal_outputs = self.generate(input=input, n_sequences=gen_sequences)
            altered_outputs = self.generate(
                input=input,
                n_sequences=gen_sequences,
                neurons_per_layer=neurons_per_layer,
            )

            normal_list.append(normal_outputs.cpu())
            altered_list.append(altered_outputs.cpu())

        outputs = torch.concat([*normal_list, *altered_list], dim=0)
        labels = [0] * n_sequences + [1] * n_sequences

        filename = "output/generated_sentences.csv"
        self.write_output(filename, outputs, labels)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    prompt = "I watched a new movie yesterday. I thought it was"

    generator = Generator(model, tokenizer)
    generator.generate_samples(prompt)
