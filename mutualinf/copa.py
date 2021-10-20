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
        self.token_set_dict = {}
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        return df

    # Wanted to create something that gives you why
    # def _because_so_maker(self,word):
    #     if(word == "cause"):
    #         return "because"
    #     else:
    #         return "so"
    
    # def _what_why_maker(self,word):
    #     if(word == "cause"):
    #         return "What caused"
    #     else:
    #         return "Why"

    def _get_templates(self):

        templates = {
            '0_shot_w_instruction0' : (
                lambda row : (  "I will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
                                f"Premise: {row['premise']}\n"
                                f"1) {row['choice1']}\n"
                                f"2) {row['choice2']}\nAnswer:"), self.token_set_dict),

            '1_shot_w_instruction0' : (
                lambda row : (  SHOTS[0] + "\nI will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
                                f"Premise: {row['premise']}\n"
                                f"1) {row['choice1']}\n"
                                f"2) {row['choice2']}\nAnswer:"), self.token_set_dict),

            '0_shot_w_instruction1' : (
                lambda row : (  "I will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
                                f"Premise: {row['premise']}\n"
                                f"1) {row['choice1']}\n"
                                f"2) {row['choice2']}\nThe most plausible alternative is: Sentence"), self.token_set_dict),

            '1_shot_w_instruction1' : (
                lambda row : (  SHOTS[1] + "\nI will give you a premise and you will choose either sentence 1) or 2) which is the better plausible alternative.\n"
                                f"Premise: {row['premise']}\n"
                                f"1) {row['choice1']}\n"
                                f"2) {row['choice2']}\nThe most plausible alternative is: Sentence"), self.token_set_dict),

            '0_shot_qa0' : (
                lambda row : (
                            f"{row['premise']}\n"
                            f"Which of the following alternatives is most plausible for the previous sentence?\n"
                            f"Sentence 1) {row['choice1']}\n"
                            f"Sentence 2) {row['choice2']}\nThe most plausible alternative is sentence"), self.token_set_dict),

            '1_shot_qa0' : (
                lambda row : (  SHOTS[2] + ""
                                f"{row['premise']}\n"
                                f"Which of the following alternatives is most plausible for the previous sentence?\n"
                                f"Sentence 1) {row['choice1']}\n"
                                f"Sentence 2) {row['choice2']}\nThe most plausible alternative is sentence"), self.token_set_dict
                                ),

            '0_shot_dialogue' : (
                lambda row : (  f"P1: Here\'s a premise: {row['premise']}."
                                f"Which sentence provides the better alternative? 1. {row['choice1'].strip('.')}, or 2. {row['choice2']}"
                                f"P2: The better alternative is sentence"), self.token_set_dict), 

            '1_shot_dialogue' : (
                lambda row : (  SHOTS[3] + f"\nP1: Here\'s a premise: {row['premise']}."
                                f"Which sentence provides the better alternative? 1. {row['choice1'].strip('.')}, or 2. {row['choice2']}"
                                f"P2: The better alternative is sentence"), self.token_set_dict), 

            '0_shot_copa' : (
                lambda row : (  "Solve the following COPA task.\n\n"
                                f"Premise: {row['premise']}\n"
                                f"Choice 1. {row['choice1']}\n"
                                f"Choice 2. {row['choice2']}\nAnswer: Choice"), self.token_set_dict), 

            '1_shot_copa' : (
                lambda row : (  SHOTS[4] + "Solve the following COPA task.\n\n"
                                f"Premise: {row['premise']}\n"
                                f"Choice 1. {row['choice1']}\n"
                                f"Choice 2. {row['choice2']}\nAnswer: Choice"), self.token_set_dict),

            'what_is_cause_effect_of_premise' : (

                lambda row : (  f'What is the {row.question} of the following premise:"{row.premise}"\n\n'
                                f'Choice 1. {row.choice1}\n'
                                f'Choice 2. {row.choice2}\n'
                                f'Answer: Choice'), self.token_set_dict
                            ),

            'what_is_cause_effect_of_premise_given_smthn' : (

                lambda row : (  f'Choose between Choice 1 or Choice 2\n'
                                f'What is the {row.question} of the following premise:"{row.premise}"\n\n'
                                f'Choice 1. {row.choice1}\n'
                                f'Choice 2. {row.choice2}\n'
                                f'Answer: Choice'), self.token_set_dict
                            ),
                            
            'what_is_cause_effect_of_premise_given1or2' : (

                lambda row : (  f'What is the {row.question} of the following premise:"{row.premise}"\n\n'
                                f'If asked to choose between '
                                f'Choice 1:"{row.choice1}" or '
                                f'Choice 2: "{row.choice2}"\n'
                                f'My answer would be: Choice'), self.token_set_dict
                            ),


            'Based_on_this_premise_1' : (

                lambda row : (  f'Based on this premise:"{row.premise}"\n\n'
                                f'If asked to choose between\n'
                                f'Choice 1:"{row.choice1}"\n'
                                f'or\n'
                                f'Choice 2: "{row.choice2}"\n'
                                f'My answer would be: Choice'), self.token_set_dict
                            ),

            'Based_on_this_premise_2' : (

                lambda row : (  f'Based on this premise:"{row.premise}"\n\n'
                                f'If asked to pick between\n'
                                f'Choice 1:"{row.choice1}" '
                                f'or'
                                f'Choice 2: "{row.choice2}" to get the {row.question}\n'
                                f'I would say:"Choice'), self.token_set_dict
                            ),

            'I_want_to_figure_out' : (

                lambda row : (  f'I want to figure out what is the {row.question} of:"{row.premise}"\n'
                                f'Choice 1:"{row.choice1}" or '
                                f'choice 2:"{row.choice2}"\n'
                                f'I would say:"choice'), self.token_set_dict
                            ),

            'DirectChoice' : (

                lambda row : (  f'What {row.question}ed the Premise:"{row.premise}"\n'
                                f'Choose between "1" or "2"\n'
                                f'1:"{row.choice1}"\n'
                                f'2:"{row.choice2}"\n'
                                f'Answer:"'), self.token_set_dict
                            ),   

             'Introducing_questions_then_premise_1' : (

                lambda row : (  f'If asked to pick between '
                                f'choice 1:"{row.choice1}" or '
                                f'choice 2:"{row.choice2}" to see what was the {row.question} of this premise:"{row.premise}"\n'
                                f'I would say:"choice'), self.token_set_dict
                            ),

             'my_fav_approach' : (

                lambda row : (  f'Read the following premise and answer by choosing "{row.question}1" or "{row.question}2"\n'
                                f'Premise:"{row.premise}"\n'
                                f'{row.question}1:"{row.choice1}"\n'
                                f'{row.question}2:"{row.choice2}"\n'
                                f'Answer:"{row.question}'), self.token_set_dict
                            ),

             'my_fav_approach_flipped_order' : (

                lambda row : (  f'Read the following premise and pick "{row.question}2" or "{row.question}1"\n'
                                f'Premise:"{row.premise}"\n'
                                f'{row.question}1:"{row.choice1}"\n'
                                f'{row.question}2:"{row.choice2}"\n'
                                f'Answer:"{row.question}'), self.token_set_dict
                            ),

        }

        return templates


if __name__ == '__main__':
    copa = CopaDataset()