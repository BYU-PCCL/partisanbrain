from collections import defaultdict
from infra_modules import Dataset

import pandas as pd

'''
NOTE: trim LM output to first character only. 
@krogers need to upload cleaned dataset to pccfs2
'''

SHOTS = [
    """I will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n
    Premise: The man broke his toe. 
    1) He got a hole in his sock.
    2) He dropped a hammer on his foot. 
    Answer: Sentence 2) is the better alternative. 
    """,
    """I will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n
    Premise: The man broke his toe. 
    1) He got a hole in his sock.
    2) He dropped a hammer on his foot. 
    The most plausible alternative is: Sentence 2).
    """,
    """The man broke his toe.
    Which of the following alternatives is most plausible for the previous sentence?\n
    Sentence 1) He got a hole in his sock.
    Sentence 2) He dropped a hammer on his foot. 
    The most plausible alternative is sentence 2).
    """,
    """P1: Here\'s a premise: The man broke his toe.
    Which sentence provides the better alternative?
    1. He got a hole in his sock, or 
    2. He dropped a hammer on his foot. 
    P2: The better alternative is sentence
    """,
    """Solve the following COPA task.

    Premise: The man broke his toe.
    Choice 1. He got a hole in his sock.
    Choice 2. He dropped a hammer on his foot.
    Answer: Choice 2.
    """

]

class CopaDataset(Dataset):
    def __init__(self, sample_seed=0, n=None):
        self.token_set_dict = None
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):

        templates = {
            '0_shot_w_instruction0' : (lambda row : ("I will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
            f"Premise: {row['premise']}\n"
            f"1) {row['choice1']}\n"
            f"2) {row['choice2']}\nAnswer:"), self.token_set_dict),

            '1_shot_w_instruction0' : (lambda row : (SHOTS[0] + "\nI will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
            f"Premise: {row['premise']}\n"
            f"1) {row['choice1']}\n"
            f"2) {row['choice2']}\nAnswer:"), self.token_set_dict),

            '0_shot_w_instruction1' : (lambda row : ("I will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
            f"Premise: {row['premise']}\n"
            f"1) {row['choice1']}\n"
            f"2) {row['choice2']}\nThe most plausible alternative is: Sentence"), self.token_set_dict),

            '1_shot_w_instruction1' : (lambda row : (SHOTS[1] + "\nI will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
            f"Premise: {row['premise']}\n"
            f"1) {row['choice1']}\n"
            f"2) {row['choice2']}\nThe most plausible alternative is: Sentence"), self.token_set_dict),

            '0_shot_qa0' : (lambda row : (
            f"{row['premise']}\n"
            f"Which of the following alternatives is most plausible for the previous sentence?\n"
            f"Sentence 1) {row['choice1']}\n"
            f"Sentence 2) {row['choice2']}\nThe most plausible alternative is sentence"), self.token_set_dict),

            '1_shot_qa0' : (lambda row : (SHOTS[2] + ""
            f"{row['premise']}\n"
            f"Which of the following alternatives is most plausible for the previous sentence?\n"
            f"Sentence 1) {row['choice1']}\n"
            f"Sentence 2) {row['choice2']}\nThe most plausible alternative is sentence"), self.token_set_dict),

            '0_shot_dialogue' : (lambda row : (f"P1: Here\'s a premise: {row['premise']}."
            f"Which sentence provides the better alternative? 1. {row['choice1'].strip('.')}, or 2. {row['choice2']}"
            f"P2: The better alternative is sentence"), self.token_set_dict), 

            '1_shot_dialogue' : (lambda row : (SHOTS[3] + f"\nP1: Here\'s a premise: {row['premise']}."
            f"Which sentence provides the better alternative? 1. {row['choice1'].strip('.')}, or 2. {row['choice2']}"
            f"P2: The better alternative is sentence"), self.token_set_dict), 

            '0_shot_copa' : (lambda row : ("Solve the following COPA task.\n\n"
            f"Premise: {row['premise']}\n"
            f"Choice 1. {row['choice1']}\n"
            f"Choice 2. {row['choice2']}\nAnswer: Choice"), self.token_set_dict), 

            '1_shot_copa' : (lambda row : (SHOTS[4] + "Solve the following COPA task.\n\n"
            f"Premise: {row['premise']}\n"
            f"Choice 1. {row['choice1']}\n"
            f"Choice 2. {row['choice2']}\nAnswer: Choice"), self.token_set_dict),
        }


if __name__ == '__main__':
    copa = CopaDataset()