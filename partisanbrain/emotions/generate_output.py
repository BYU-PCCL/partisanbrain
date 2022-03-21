from lda_neuron_selection import LdaNeuronSelector
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
        self, model, tokenizer, force_emotion="positive", percentile=0.8, layers=25
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.force_emotion = force_emotion
        self.percentile = percentile
        self.layers = layers

    def convert_to_text(self, output):
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def write_output(self, filename, outputs):
        sentences = [self.convert_to_text(output) for output in outputs]
        series = pd.Series(data=sentences)
        series.to_csv(filename, index=False)

    def generate(self, input, n_sequences=N_SEQUENCES, neurons_per_layer=None):
        outputs = self.model.generate(
            input,
            max_length=30,
            do_sample=True,
            num_return_sequences=n_sequences,
            early_stopping=True,
            neurons_per_layer=neurons_per_layer,
            force_emotion=self.force_emotion,
        )
        return outputs

    def generate_samples(
        self,
        prompt,
        output_filename,
        n_sequences=N_SEQUENCES,
    ):
        lda_selector = LdaNeuronSelector(
            filename="output/output.npz", device=DEVICE, percentile=self.percentile
        )

        if self.force_emotion != "default":
            neurons_per_layer = lda_selector.get_lda_neurons_per_layer(
                layers=self.layers
            )
        else:
            neurons_per_layer = None

        input = self.tokenizer.encode(prompt.strip(), return_tensors="pt")
        input = input.to(DEVICE)

        n_batches = n_sequences // BATCH_SIZE
        gen_sequences = min(n_sequences, BATCH_SIZE)

        final_batch_sequences = n_sequences % BATCH_SIZE
        if final_batch_sequences == 0:
            final_batch_sequences = BATCH_SIZE
        else:
            n_batches += 1

        outputs = []
        for i in range(n_batches):
            batch_outputs = self.generate(
                input=input,
                n_sequences=(
                    gen_sequences if i < n_batches - 1 else final_batch_sequences
                ),
                neurons_per_layer=neurons_per_layer,
            )

            outputs.append(batch_outputs.cpu())

        self.write_output(output_filename, outputs)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl", pad_token_id=tokenizer.eos_token_id
    )
    model.to(DEVICE)
    model.eval()

    prompt = "I watched a new movie yesterday. I thought it was"
    output_filename = "output/generated_sentences.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--emotion", default="default")
    parser.add_argument("-p", "--percentile", type=float, default=0.8)
    parser.add_argument(
        "-l", "--layers", nargs="+", type=int, default=list(range(25, 49))
    )
    parser.add_argument("-s", "--sentences", type=int, default=1000)
    args = parser.parse_args()

    generator = Generator(
        model,
        tokenizer,
        force_emotion=args.emotion,
        percentile=args.percentile,
        layers=args.layers,
    )
    generator.generate_samples(
        prompt=prompt, output_filename=output_filename, n_sequences=args.sentences
    )
