from dataset import Dataset


class ImdbDataset(Dataset):
    def __init__(self, sample_seed=0, n=None):
        self.token_set_dict = {'positive' : ['positive', 'good', 'happy', 'great', 'excellent'], 'negative' : ['negative', 'bad', 'poor', 'sad', 'depressing']}
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        mod_df = df.rename(columns={'sentiment':'ground_truth'})
        # in review, replace all <br /> with newline
        mod_df['review'] = mod_df['review'].str.replace('<br />', '\n')
        return mod_df

    def _get_templates(self):
        templates = {
            'review_follow_up_q0': (lambda row: (f"{row['review']}\n\n"
                                    "Was the previous review positive or negative? The previous review was"), self.token_set_dict),

            'review_follow_up_q1': (lambda row: (f"{row['review']}\n\n"
                                    "Was the previous review negative or positive? The previous review was"), self.token_set_dict),

            'review_follow_up_q2': (lambda row: (f"{row['review']}\n\n"
                                    "Was the sentiment of previous review positive or negative? The previous review was"), self.token_set_dict),

            'review_follow_up_q3': (lambda row: (f"{row['review']}\n\n"
                                    "Was the sentiment of previous review negative or positive? The previous review was"), self.token_set_dict),

            'task_review_classify0' : (lambda row: ("After reading the following review, classify it as positive or negative. \n\nReview: "
                                        f"{row['review']} \n\nClassification:"), self.token_set_dict),

            'task_review_classify1' : (lambda row: ("After reading the following review, classify it as negative or positive. \n\nReview: "
                                        f"{row['review']} \n\nClassification:"), self.token_set_dict),         

            'task_review_follow_up0' : (lambda row: ("Read the following movie review to determine the review's sentiment.\n\n"
                                        f"{row['review']}\n\nIn general, was the sentiment positive or negative? The sentiment was"), self.token_set_dict),

            'task_review_follow_up1' : (lambda row: ("Read the following movie review to determine the review's sentiment.\n\n"
                                        f"{row['review']}\n\nIn general, was the sentiment negative or positive? The sentiment was"), self.token_set_dict),

            'task_review_follow_up2' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:  "
                                        f"{row['review']}\n\nIn general, was the sentiment positive or negative The sentiment was"), self.token_set_dict),

            'task_review_follow_up3' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:  "
                                        f"{row['review']}\n\nIn general, was the sentiment negative or positive? The sentiment was"), self.token_set_dict),

            'task_review_follow_up3' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:\n\"\"\"\n"
                                        f"{row['review']}\n\"\"\"\nIn general, was the sentiment positive or negative? The sentiment was"), self.token_set_dict),

            'task_review_follow_up4' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:\n\"\"\"\n"
                                        f"{row['review']}\n\"\"\"\nIn general, was the sentiment negative or positive? The sentiment was"), self.token_set_dict),

            'task_review_follow_up4' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:\n\"\"\"\n"
                                        f"{row['review']}\n\"\"\"\nIn general, what was the sentiment of the review? The sentiment was"), self.token_set_dict),

            'story_review_conclusion0' : (lambda row: ("Yesterday I went to see a movie. "
                                        f"{row['review']} Between positive and negative, I would say the movie was"), self.token_set_dict),

            'q_and_a0' : (lambda row: ('''Q: Is the sentiment of the following movie review positive or negative?\n"""\n'''
                                        f"{row['review']}\n\"\"\"\nA: The sentiment of the movie review was"), self.token_set_dict),

            'q_and_a1' : (lambda row: ('''Q: Is the sentiment of the following movie review negative or positive?\n"""\n'''
                                        f"{row['review']}\n\"\"\"\nA: The sentiment of the movie review was"), self.token_set_dict),

            'q_and_a2' : (lambda row: ("Q: Is the sentiment of the following movie review positive or negative?\n"
                                        f"{row['review']} \nA (positive or negative):"), self.token_set_dict),

            'q_and_a3' : (lambda row: ("Q: Is the sentiment of the following movie review negative or positive?\n"
                                        f"{row['review']} \nA (negative or positive):"), self.token_set_dict),

            'dialogue0' : (lambda row: ("P1: Could you give me a review of the movie you just saw? "
                                        f"\nP2: Sure, {row['review']} "
                                        "\nP1: So, overall, would you give it a positive or negative review? "
                                        "\nP2: I would give it a"), self.token_set_dict),

            'dialogue1' : (lambda row: ("P1: Could you give me a review of the movie you just saw? "
                                        f"\nP2: Sure, {row['review']} "
                                        "\nP1: So overall was the sentiment of the movie negative or positive? "
                                        "\nP2: I would give it a"), self.token_set_dict),

            'dialogue2' : (lambda row: ("P1: How was the movie? "
                                        f"\nP2: {row['review']} "
                                        "\nP1: Would you say your review of the movie is positive or negative? "
                                        "\nP2: I would say my review of the movie is"), self.token_set_dict),

            'dialogue3' : (lambda row: ("P1: How was the movie? "
                                        f"\nP2: {row['review']} "
                                        "\nP1: Would you say your review of the movie is negative or positive? "
                                        "\nP2: I would say my review review of the movie is"), self.token_set_dict),
            
        }
        return templates

if __name__ == '__main__':
    # Data is in data/imdb/raw.csv
    imdb = ImdbDataset()
