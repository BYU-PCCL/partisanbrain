from transformers import pipeline
import pandas as pd
import torch
import argparse


DEVICE = 0 if torch.cuda.is_available() else None


class SentimentClassifier:
    def __init__(self, input_filename):
        self.df = pd.read_csv(input_filename)
        self.mod_df = self.df.copy()
        self.classifier = pipeline("sentiment-analysis", device=DEVICE)

    def classify_sentiment(self):
        results = self.classifier(self.df.sentence.to_list())
        scores = [result["score"] for result in results]
        sentiments = [result["label"] for result in results]

        self.mod_df["score"] = scores
        self.mod_df["sentiment"] = sentiments

    def get_value_counts(self):
        return self.mod_df[mod_df["label"] == 1].sentiment.value_counts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--emotion", default="default")
    args = parser.parse_args()
    input_filename = f"output/{args.emotion}.csv"

    classifier = SentimentClassifier(input_filename)
    classifier.classify_sentiment()

    value_counts = classifier.get_value_counts()

    print(args.emotion[0].upper() + args.emotion[1:])
    print(value_counts, end="\n\n")
