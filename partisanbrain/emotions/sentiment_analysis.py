from transformers import pipeline
import pandas as pd


class SentimentClassifier:
    def __init__(self, input_filename):
        self.df = pd.read_csv(input_filename)
        self.mod_df = self.df.copy()
        self.classifier = pipeline("sentiment-analysis")

    def classify_sentiment(self):
        results = self.classifier(self.df.sentence.to_list())
        scores = [result[0]["score"] for result in results]
        sentiments = [result[0]["label"] for result in results]

        self.mod_df["score"] = scores
        self.mod_df["semtiment"] = sentiments

    def get_value_counts(self):
        altered_mask = self.mod_df.label == 1
        normal_mask = self.mod_df.label == 0

        normal_value_counts = self.mod_df[normal_mask].sentiment.value_counts()
        altered_value_counts = self.mod_df[altered_mask].sentiment.value_counts()

        return normal_value_counts, altered_value_counts


if __name__ == "__main__":
    input_filename = "output/generated_sentences.csv"

    classifier = SentimentClassifier(input_filename)
    classifier.classify_sentiment()

    normal_value_counts, altered_value_counts = classifier.get_value_counts()

    print("Unaltered")
    print(normal_value_counts, end="\n\n")

    print("Altered")
    print(altered_value_counts)
