from lda_neuron_selection import LdaNeuronSelector
from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import torch


N_NEURONS = 100
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

    def generate_samples(
        self,
        prompt,
        output_filename,
        n_sequences=N_SEQUENCES,
    ):
        lda_selector = LdaNeuronSelector(filename="output/output.npz")
        neurons_per_layer = lda_selector.get_lda_neurons_per_layer(layers=25)

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

        self.write_output(output_filename, outputs, labels)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    prompt = "I watched a new movie yesterday. I thought it was"
    output_filename = "output/generated_sentences.csv"

    generator = Generator(model, tokenizer)
    generator.generate_samples(prompt=prompt, output_filename=output_filename)
