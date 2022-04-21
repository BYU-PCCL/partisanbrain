import numpy as np
import string
import pandas as pd
from datasets import load_dataset


def process_sentence(sentence):
    sentence = sentence.strip()
    punctuation = list(string.punctuation)
    for punc in punctuation:
        sentence = sentence.replace(f" {punc}", punc)
    sentence = sentence.replace("''", "")

    return sentence


dataset = load_dataset("wiki_split")

simple_sentences = dataset["test"]["simple_sentence_2"]
simple_sentences = list(
    map(
        process_sentence,
        simple_sentences,
    )
)

df = pd.DataFrame(columns=["sentence"], data=simple_sentences)
df.to_csv("data/wiki.csv", index=False)
