from audioop import avg
import torch
import numpy as np
import pandas as pd
from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PerplexityAnalyzer:
    def __init__(self, model, tokenizer, df=None, filename=None):
        self.df = df if df is not None else pd.read_csv(filename)
        self.model = model
        self.tokenizer = tokenizer

    def get_likelihood_sequence(self, input, log_probs):
        return [
            log_probs[:, i, token_index].item()
            for i, token_index in enumerate(input.squeeze()[1:])
        ]

    def process_sentence(
        self,
        sentence,
        neurons_per_layer=None,
        force_emotion=None,
        force_with=None,
    ):
        input = self.tokenizer.encode(sentence.strip(), return_tensors="pt")
        input = input.to(DEVICE)

        with torch.no_grad():
            output = self.model(
                input,
                neurons_per_layer=neurons_per_layer,
                force_emotion=force_emotion,
                force_with=force_with,
            )

        log_probs = torch.nn.functional.log_softmax(output.logits, dim=2)
        likelihood_sequence = self.get_likelihood_sequence(input, log_probs)

        return likelihood_sequence

    def get_perplexity(self, likelihood_sequence):
        # TODO double check if we need to add or minus 1 from t
        t = len(likelihood_sequence)
        perplexity = np.exp((-1 / t) * sum(likelihood_sequence))

        return perplexity

    def get_average_perplexity(
        self,
        neurons_per_layer=None,
        force_emotion=None,
        force_with=None,
    ):
        perplexities = []
        for i, row in self.df.iterrows():
            likelihood = self.process_sentence(
                row.sentence,
                neurons_per_layer=neurons_per_layer,
                force_emotion=force_emotion,
                force_with=force_with,
            )
            perplexity = self.get_perplexity(likelihood)
            perplexities.append(perplexity)

        return np.mean(perplexities)


if __name__ == "__main__":

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

    perp_analyzer = PerplexityAnalyzer(model, tokenizer)
    filename = "data/wiki.csv"

    avg_perplexity = perp_analyzer.get_average_perplexity(filename)

    print(avg_perplexity)
