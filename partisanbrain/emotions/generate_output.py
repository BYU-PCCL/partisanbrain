from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import torch
import argparse


N_NEURONS = 100
BATCH_SIZE = 100
N_SEQUENCES = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator:
    def __init__(
        self,
        model,
        tokenizer,
        neurons_per_layer=None,
        force_emotion="positive",
        percentile=0.8,
        force_with="per_neuron",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.force_emotion = force_emotion
        self.percentile = percentile
        self.force_with = force_with
        self.neurons_per_layer = (
            neurons_per_layer
            if neurons_per_layer
            else select_neurons_per_layer(percentile=self.percentile)
        )

    def convert_to_text(self, output):
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def get_df(self, outputs, labels):
        sentences = [self.convert_to_text(output) for output in outputs]
        df = pd.DataFrame(
            data=np.array([labels, sentences]).T, columns=["label", "sentence"]
        )
        return df

    def write_output(self, filename, outputs, labels):
        df = self.get_df(outputs, labels)
        df.to_csv(filename, index=False)

    def generate(self, input, n_sequences=N_SEQUENCES):
        outputs = self.model.generate(
            input,
            max_length=30,
            do_sample=True,
            num_return_sequences=n_sequences,
            early_stopping=True,
            neurons_per_layer=self.neurons_per_layer,
            force_emotion=self.force_emotion,
            force_with=self.force_with,
        )
        return outputs

    def generate_samples(
        self,
        prompt,
        output_filename=None,
        n_sequences=N_SEQUENCES,
        n_neurons=N_NEURONS,
    ):
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
            )

            normal_list.append(normal_outputs.cpu())
            altered_list.append(altered_outputs.cpu())

        outputs = torch.concat([*normal_list, *altered_list], dim=0)
        labels = [0] * n_sequences + [1] * n_sequences

        if output_filename:
            self.write_output(output_filename, outputs, labels)

        df = self.get_df(outputs, labels)
        return df


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.to(DEVICE)
    model.eval()

    prompt = "I watched a new movie yesterday. I thought it was"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--emotion", default="default")
    parser.add_argument("-p", "--percentile", type=float, default=0.8)
    parser.add_argument("-s", "--sentences", type=int, default=1000)
    parser.add_argument("-n", "--neurons", type=int, default=100)
    args = parser.parse_args()

    output_filename = f"output/{args.emotion}.csv"

    generator = Generator(
        model,
        tokenizer,
        force_emotion=args.emotion,
        percentile=args.percentile,
    )
    generator.generate_samples(
        prompt=prompt,
        output_filename=output_filename,
        n_sequences=args.sentences,
        n_neurons=args.neurons,
    )
