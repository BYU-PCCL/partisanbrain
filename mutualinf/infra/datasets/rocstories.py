from collections import defaultdict
from infra_modules import Dataset

import pandas as pd

class RocstoriesDataset(Dataset):
    def __init__(self):
        self.token_set_dict = None
        super().__init__()

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):
        templates = {
            'd_story_a0' : (lambda row: ("Fill in the blank with the missing word or phrase. "
                f"{row['story_with_mask']} \nAnswer:"), self.token_set_dict),
            'd_story_a1' : (lambda row: ("Fill in the blank with the missing word or phrase. "
                f"{row['story_with_mask']} Missing Word/Phrase:"), self.token_set_dict),
            'd_story_a2' : (lambda row: ("Fill in the blank with the missing word or phrase to complete the sentence. "
                f"{row['story_with_mask']} \nAnswer:"), self.token_set_dict),
            'd_story_a3' : (lambda row: ("Guess the word in the blank to complete the story.\nStory: "
                f"{row['story_with_mask']} \nAnswer:"), self.token_set_dict),
            'd_story_a4' : (lambda row: ("Pick the best word to replace the blank .\nStory: "
                f"{row['story_with_mask']} \nAnswer:"), self.token_set_dict),
            'd_story_a5' : (lambda row: ("Read the following sentences, and try to guess which word goes in the blank. "
                f"{row['story_with_mask']} \nAnswer:"), self.token_set_dict),
            'story_d0' : (lambda row: (f"{row['story_with_mask']} \n"
                "Fill in the blank with the missing word or phrase to complete the sentence. "), self.token_set_dict),
            'story_d1' : (lambda row: (f"{row['story_with_mask']} \n"
                "Fill in the blank with the missing word or phrase."), self.token_set_dict),
            'story_d2' : (lambda row: (f"{row['story_with_mask']} \n"
                "Put the best word in the blank to complete the story. \nWord:"), self.token_set_dict),
            'story_d3' : (lambda row: (f"{row['story_with_mask']} \n"
                "\nThe missing word in the story should be:"), self.token_set_dict),
            'story_d4' : (lambda row: (f"{row['story_with_mask']} \n"
                "Choose a word to replace the blank. \nWord:"), self.token_set_dict),
            'story_q0' : (lambda row: (f"{row['story_with_mask']} \n"
                "What word goes in the blank?"), self.token_set_dict),
            'story_q1' : (lambda row: (f"{row['story_with_mask']} \n"
                "Which word fills in the blank best?"), self.token_set_dict),
            'story_q2' : (lambda row: (f"{row['story_with_mask']} \n"
                "Which word should we put in the blank to complete the story?"), self.token_set_dict),
            'story_q3' : (lambda row: (f"{row['story_with_mask']} \n"
                "What word fits in the blank in the previous story we just read?"), self.token_set_dict),
            'story_q4' : (lambda row: (f"{row['story_with_mask']} \n"
                "What word would you choose to replace the blank in the above story?"), self.token_set_dict),
            'dialogue0' : (lambda row: ("P1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"{row['story_with_mask']} \nP2: The word which fits best is"), self.token_set_dict),
            'dialogue1' : (lambda row: ("P1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"{row['story_with_mask']} \nP2:"), self.token_set_dict),
            'dialogue2' : (lambda row: ("P1: What word do you think fits best in the following story? \n"
                f"{row['story_with_mask']} \nP2: The word which fits best is"), self.token_set_dict),
            'dialogue3' : (lambda row: ("P1: What word do you think fits best in the following story? \n"
                f"{row['story_with_mask']} \nP2:"), self.token_set_dict)  
        }

        return templates

if __name__ == '__main__':
    # Data is in data/rocstories/raw.csv
    roc = RocstoriesDataset()