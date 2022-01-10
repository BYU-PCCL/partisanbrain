from collections import defaultdict
from infra_modules import Dataset

import pandas as pd

class LambadaDataset(Dataset):
    def __init__(self, sample_seed=0, n=None):
        self.token_set_dict = None
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        # make column "story_with_mask" which is column "text" + ____.
        df['story_with_mask'] = df['text'] + ' ____.'
        return df

    def _get_templates(self):
        # function to get all string before '_'
        beginning = lambda s: (s.split('_')[0]).strip()

        templates = {
            'd_story_a0' : (lambda row: ("Fill in the blank with the missing word.\n"
                f"{row['story_with_mask']} \nAnswer: \""), self.token_set_dict),

            'd_story_a1' : (lambda row: ("Fill in the blank with the missing word.\n\nPassage: I like to eat peanut butter and jelly ____.\nMissing Word: sandwiches\n\n"
                f"Passage: {row['story_with_mask']}\nMissing Word: \""), self.token_set_dict),

            'd_story_a1' : (lambda row: ("Fill in the blank with the missing word to complete the sentence.\n\nPassage: I like to eat peanut butter and jelly ____.\nMissing Word: sandwiches\n\n"
                f"Passage: {row['story_with_mask']}\nMissing Word: \""), self.token_set_dict),

            'd_story_a4' : (lambda row: ("Pick the best word to replace the blank.\nStory: "
                f"{row['story_with_mask']} \nAnswer: \""), self.token_set_dict),

            'd_story_a5' : (lambda row: ("Read the following sentences, and try to guess which word goes in the blank.\n"
                f"{row['story_with_mask']} \nAnswer: \""), self.token_set_dict),

            'story_d0' : (lambda row: (f"{row['story_with_mask']} \n"
                "Fill in the blank with the missing word or phrase to complete the sentence.\nWhat is the missing word? The missing word is \""), self.token_set_dict),

            'stery_d1' : (lambda row: (f"{row['story_with_mask']} \n"
                "Fill in the blank with the missing word or phrase.\nWhat is the missing word? The missing word is \""), self.token_set_dict),

            'story_d2' : (lambda row: (f"It was a cold night. The wind was ____ around the courtyard as I stepped out of the car and into the darkness.\nWord: whistling\n\n{row['story_with_mask']} \n"
                "\nWord:"), self.token_set_dict),

            'story_d3' : (lambda row: (f"{row['story_with_mask']} \n"
                "\nThe missing word in the story should be: \""), self.token_set_dict),

            'story_q1' : (lambda row: (f"{row['story_with_mask']} \n"
                "Which word fills in the blank best?\nThe word that fills in the blank best is \""), self.token_set_dict),

            'story_q2' : (lambda row: (f"{row['story_with_mask']} \n"
                "Which word should we put in the blank to complete the story? Let's use the word \""), self.token_set_dict),
            
            'repeat1': (lambda row: (f"Fill in the blank for the following sentences.\n\n"
                "\"It was a cold night. The wind was whistling around the courtyard as I stepped out of the car and into the ____.\" -> \"It was a cold night. The wind was whistling around the courtyard as I stepped out of the car and into the darkness.\"\n"
                f"\"{row['story_with_mask']}\" -> "
                f"\"{row['text']}"), self.token_set_dict),

            'clip': (lambda row: (
                f"{row['text']}"), self.token_set_dict),

            'dialogue0' : (lambda row: ("P1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"{row['story_with_mask']} \nP2: The word which fits best is \""), self.token_set_dict),

            'dialogue1' : (lambda row: (
                "P1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"It was a cold night. The wind was ____ around the courtyard as I stepped out of the car and into the darkness."
                "\nP2: whistling"
                "\nP1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"{row['story_with_mask']} \nP2:"
                ), self.token_set_dict),

            'dialogue2' : (lambda row: ("P1: What word do you think fits best in the following story? \n"
                f"{row['story_with_mask']} \nP2: The word which fits best is \""), self.token_set_dict),

            'dialogue3' : (lambda row: (
                "P1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"I like to eat peanut butter and jelly ____."
                "\nP2: sandwiches"
                "\nP1: I'm going to tell you a story, but leave a word out. Once I'm done telling the story, pick the word that best fits in the blank. \n"
                f"{row['story_with_mask']} \nP2:"
                ), self.token_set_dict),
            
            'openai-0shot': (lambda row: (
                'Fill in blank:\n\n'
                f'{row["story_with_mask"]} ->'
            ), self.token_set_dict),

            'openai-1shot': (lambda row: (
                'Fill in blank:\n\n'
                'Alice was friends with Bob. Alice went to visit her friend ____. -> Bob\n'
                f'{row["story_with_mask"]} ->'
            ), self.token_set_dict),

            'openai-2shot': (lambda row: (
                'Fill in blank:\n\n'
                'Alice was friends with Bob. Alice went to visit her friend ____. -> Bob\n'
                'George bought some baseball equipment, a ball, a glove, and a ____. -> bat\n'
                f'{row["story_with_mask"]} ->'
            ), self.token_set_dict),

            'openai-1shot-v2': (lambda row: (
                'Fill in blank:\n\n'
                'She held the torch in front of her.\n\nShe caught her breath.\n\n"Chris? There\'s a step."\n\n"What?"\n\n"A step. Cut in the rock. About fifty feet ahead." She moved faster. They both moved faster. "In fact," she said, raising the torch higher, "there\'s more than a ____. -> step\n\n'
                f'{row["story_with_mask"]} ->'
            ), self.token_set_dict),
        }
        #print(f'[DEBUG]] template_size: {len(templates)}')
        return templates

if __name__ == '__main__':
    # Data is in data/rocstories/raw.csv
    roc = LambadaDataset()