from collections import defaultdict
from infra_modules import Dataset

import pandas as pd

class ImdbDataset(Dataset):
    def __init__(self, sample_seed=0, n=None):
        self.token_set_dict = {'postive' : ['positive', 'good', 'happy', 'great', 'excellent'], 'negative' : ['negative', 'bad', 'poor', 'sad', 'depressing']}
        super().__init__(sample_seed=0, n=None)

    def _modify_raw_data(self, df):
        mod_df = df.rename(columns={'sentiment':'ground_truth'})
        return mod_df
        

    def _get_templates(self):
        templates = {
            'review_follow_up_q0': (lambda row: (f"{row['review']} "
                                    "Was the previous review positive or negative?"), self.token_set_dict),
            'review_follow_up_q1': (lambda row: (f"{row['review']} "
                                    "Was the previous review negative or positive?"), self.token_set_dict),
            'review_follow_up_q2': (lambda row: (f"{row['review']} "
                                    "Was the sentiment of previous review positive or negative?"), self.token_set_dict),
            'review_follow_up_q3': (lambda row: (f"{row['review']} "
                                    "Was the sentiment of previous review negative or positive?"), self.token_set_dict),
            'task_review_classify0' : (lambda row: ("After reading the following review, classify it as positive or negative. \nReview: "
                                        f"{row['review']} \nClassification:"), self.token_set_dict),
            'task_review_classify1' : (lambda row: ("After reading the following review, classify it as negative or positive. \nReview: "
                                        f"{row['review']} \nClassification:"), self.token_set_dict),         
            'task_review_follow_up0' : (lambda row: ("Read the following movie review to determine the review's sentiment. "
                                        f"{row['review']} In general, was the sentiment positive or negative?"), self.token_set_dict),
            'task_review_follow_up1' : (lambda row: ("Read the following movie review to determine the review's sentiment. "
                                        f"{row['review']} In general, was the sentiment negative or positive?"), self.token_set_dict),
            'task_review_follow_up2' : (lambda row: ("Considering this movie review determine its sentiment. Review:  "
                                        f"{row['review']} In general, was the sentiment positive or negative?"), self.token_set_dict),
            'task_review_follow_up3' : (lambda row: ("Considering this movie review determine its sentiment. Review:  "
                                        f"{row['review']} In general, was the sentiment negative or positive?"), self.token_set_dict),
            'story_review_conclusion0' : (lambda row: ("Yesterday I went to see a movie. "
                                        f"{row['review']} Overall the movie was"), self.token_set_dict),
            'story_review_conclusion1' : (lambda row: ("Yesterday I went to see a movie. "
                                        f"{row['review']} Overall the sentiement of movie was"), self.token_set_dict),
            'story_review_conclusion2' : (lambda row: ("Yesterday I went to see a movie. "
                                        f"{row['review']} Between positive and negative, I would say the movie was"), self.token_set_dict),
            'story_review_conclusion3' : (lambda row: ("Yesterday I went to see a movie. "
                                        f"{row['review']} Between negative and positive, I would say the movie was"), self.token_set_dict),
            'q_and_a0' : (lambda row: ("Q: Is the sentiment of the following movie review positive or negative?"
                                        f"{row['review']} \nA:"), self.token_set_dict),
            'q_and_a1' : (lambda row: ("Q: Is the sentiment of the following movie review negative or positive?"
                                        f"{row['review']} \nA:"), self.token_set_dict),
            'dialogue0' : (lambda row: ("P1: Could you give me a review of the movie you just saw? "
                                        f"\nP2: Sure, {row['review']} "
                                        "P1: So overall was the sentiment of the movie postive or negative? "
                                        "\nP2:"), self.token_set_dict),
            'dialogue1' : (lambda row: ("P1: Could you give me a review of the movie you just saw? "
                                        f"\nP2: Sure, {row['review']} "
                                        "P1: So overall was the sentiment of the movie negative or positive? "
                                        "\nP2:"), self.token_set_dict),
            'dialogue2' : (lambda row: ("P1: How was the movie? "
                                        f"\nP2: {row['review']} "
                                        "P1: Would you say your review of the movie is positive or negative? "
                                        "\nP2:"), self.token_set_dict),
            'dialogue3' : (lambda row: ("P1: How was the movie? "
                                        f"\nP2: {row['review']} "
                                        "P1: Would you say your review of the movie is negative or positive? "
                                        "\nP2:"), self.token_set_dict),
            
        }
        return templates

if __name__ == '__main__':
    # Data is in data/imdb/raw.csv
    imdb = ImdbDataset()