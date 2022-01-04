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
"""\"The man broke his toe.\"
Which of the following alternatives is most plausible for the previous sentence?\n
Sentence 1) He got a hole in his sock.
Sentence 2) He dropped a hammer on his foot. 
The most plausible alternative is sentence 2).
""",
"""P1: Here\'s a premise: "The man broke his toe."
Which sentence provides the better alternative?
1. "He got a hole in his sock", or 
2. "He dropped a hammer on his foot." 
P2: The better alternative is sentence
""",
"""Solve the following COPA tasks by choosing the sentence which makes the most sense after the premise.

Premise: The man broke his toe.
Choice 1. He got a hole in his sock.
Choice 2. He dropped a hammer on his foot.
Answer: Choice 2."""

]

question_dict = {
    'cause': 'What was the CAUSE of this?',
    'effect': 'What happened as a RESULT?',
}

COPA_BACKDROP = '''The Choice Of Plausible Alternatives (COPA) evaluation provides researchers with a tool for assessing progress in open-domain commonsense causal reasoning. COPA consists of 1000 questions, split equally into development and test sets of 500 questions each. Each question is composed of a premise and two alternatives, where the task is to select the alternative that more plausibly has a causal relation with the premise. The correct alternative is randomized so that the expected performance of randomly guessing is 50%.

Examples

Premise: The man broke his toe. What was the CAUSE of this?
Alternative 1: He got a hole in his sock. 
Alternative 2: He dropped a hammer on his foot.
Answer: Alternative 2

Premise: I tipped the bottle. What happened as a RESULT?
Alternative 1: The liquid in the bottle froze.
Alternative 2: The liquid in the bottle poured out.
Answer: Alternative 2

Premise: I knocked on my neighbor's door. What happened as a RESULT?
Alternative 1: My neighbor invited me in.
Alternative 2: My neighbor left his house.
Answer: Alternative 1'''

EXPLANATION = '''For the following premises, choose the alternative that is either a cause or result of the premise, and justify your answer.

Premise: The man broke his toe. What was the CAUSE of this?
Alternative 1: He got a hole in his sock. 
Alternative 2: He dropped a hammer on his foot.
Answer: Alternative 2. Getting a hole in your sock would not break your toe, unless there is additional information. Dropping a hammer (which is a heavy object), on the other hand, would almost certaintly break your toe. Thus, the best answer is Alternative 2.

Premise: I tipped the bottle. What happened as a RESULT?
Alternative 1: The liquid in the bottle froze.
Alternative 2: The liquid in the bottle poured out.
Answer: Alternative 2. Tipping a bottle causes liquid to fall out, not to freeze. Freezing is caused by being placed in a cold place. Pouring out (Alternative 2) is correct because it makes the most sense.

Premise: I knocked on my neighbor's door. What happened as a RESULT?
Alternative 1: My neighbor invited me in.
Alternative 2: My neighbor left his house.
Answer: Alternative 1. When you knock on a neighbor's door, it is likely that if they are home they will answer and invite you in. It does not make much sense, however, that a neighbor would leave their house without explanation. Therefore, Alternative 1 is the best result of the premise.'''

class CopaDataset(Dataset):
    def __init__(self, sample_seed=0, n=None):
        self.token_set_dict = {'1': ['1'], '2': ['2']}
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        # in 'ground_truth', map 0 to '1' and 1 to '2'
        df['ground_truth'] = df['ground_truth'].map({0: '1', 1: '2'})
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
                                f"\"{row['premise']}\"\n"
                                f"Which of the following alternatives is most plausible for the previous sentence?\n"
                                f"Sentence 1) {row['choice1']}\n"
                                f"Sentence 2) {row['choice2']}\nThe most plausible alternative is sentence"), self.token_set_dict
                                ),

            '0_shot_dialogue' : (
                lambda row : (  f"P1: Here\'s a premise: {row['premise']}."
                                f"Which sentence provides the better alternative? 1. \"{row['choice1'].strip('.')}\", or 2. \"{row['choice2']}\""
                                f"P2: The better alternative is sentence"), self.token_set_dict), 

            '1_shot_dialogue' : (
                lambda row : (  SHOTS[3] + f"\nP1: Here\'s a premise: \"{row['premise']}\"."
                                f"Which sentence provides the better alternative? 1. \"{row['choice1'].strip('.')}\", or 2. \"{row['choice2']}\""
                                f"P2: The better alternative is sentence"), self.token_set_dict), 

            '0_shot_copa' : (
                lambda row : (  "Solve the following COPA task by choosing the sentence which makes the most sense after the premise.\n\n"
                                f"Premise: {row['premise']}\n"
                                f"Choice 1. {row['choice1']}\n"
                                f"Choice 2. {row['choice2']}\nAnswer: Choice"), self.token_set_dict), 

            '1_shot_copa' : (
                lambda row : (  SHOTS[4] + "\n\n"
                                f"Premise: {row['premise']}\n"
                                f"Choice 1. {row['choice1']}\n"
                                f"Choice 2. {row['choice2']}\nAnswer: Choice"), self.token_set_dict),

            'what_is_cause_effect_of_premise' : (

                lambda row : (  f'What is the {row.question} of the following premise: "{row.premise}"\n\n'
                                f'Choice 1. {row.choice1}\n'
                                f'Choice 2. {row.choice2}\n'
                                f'Answer: Choice'), self.token_set_dict
                            ),

            'what_is_cause_effect_of_premise_given1or2' : (

                lambda row : (  f'What is the {row.question} of the following premise: "{row.premise}"\n\n'
                                f'If asked to choose between '
                                f'Choice 1: "{row.choice1}" or '
                                f'Choice 2: "{row.choice2}"\n'
                                f'My answer would be: Choice'), self.token_set_dict
                            ),


            'Based_on_this_premise_1' : (

                lambda row : (  f'Based on this premise: "{row.premise}"\n\n'
                                f'If asked to choose between\n'
                                f'Choice 1: "{row.choice1}"\n'
                                f'or\n'
                                f'Choice 2: "{row.choice2}"\n'
                                f'My answer would be: Choice'), self.token_set_dict
                            ),

            'Based_on_this_premise_2' : (

                lambda row : (  f'Based on this premise: "{row.premise}"\n\n'
                                f'If asked to pick between\n'
                                f'Choice 1: "{row.choice1}" '
                                f'or '
                                f'Choice 2: "{row.choice2}" to get the {row.question}\n of the predeciding sentence, '
                                f'I would say: "Choice'), self.token_set_dict
                            ),

            'I_want_to_figure_out' : (

                lambda row : (  f'I want to figure out which {row.question} of this sentence is more probably: "{row.premise}"\n'
                                f'Choice 1: "{row.choice1}" or '
                                f'Choice 2: "{row.choice2}"\n'
                                f'I would say: "Choice'), self.token_set_dict
                            ),

            'Introducing_questions_then_premise_1' : ( 
                lambda row : (  f'If asked to pick between '
                            f'choice 1 ("{row.choice1}") or '
                            f'choice 2 ("{row.choice2}") to see what the {row.question} of this premise ("{row.premise}") was, '
                            f'I would say: "choice'), self.token_set_dict
                        ),

            'my_fav_approach' : (
                 lambda row : (  f'Read the following premise and answer by choosing "{row.question}1" or "{row.question}2"\n'
                            f'Premise: "{row.premise}"\n'
                            f'{row.question}1: "{row.choice1}"\n'
                            f'{row.question}2: "{row.choice2}"\n'
                            f'Answer: "{row.question}'), self.token_set_dict
                        ),

            'my_fav_approach_flipped_order' : (
                lambda row : (  f'Read the following premise and pick "{row.question}2" or "{row.question}1"\n'
                            f'Premise: "{row.premise}"\n'
                            f'{row.question}1: "{row.choice1}"\n'
                            f'{row.question}2: "{row.choice2}"\n'
                            f'Answer: "{row.question}'), self.token_set_dict
                        ),
        
            'complete_story1': (
                lambda row : (  f'Which one of these stories makes the most sense?\n'
                                f'Story 1: {row.premise} {row.choice1}\n'
                                f'Story 2: {row.premise} {row.choice2}\n'
                                f'Answer: Story'), self.token_set_dict
                
                ),
            
            'complete_story2': (
                lambda row : (  f'I am going to tell you two stories, one of them will make sense and the other will not.\n'
                                f'Story 1: {row.premise} {row.choice1}\n'
                                f'Story 2: {row.premise} {row.choice2}\n'
                                f'The story that makes sense is Story'), self.token_set_dict),

            'copa_backdrop': (
                lambda row : ( COPA_BACKDROP +
                    f'\n\nPremise: {row.premise} {question_dict[row.question]}\n'
                    f'Alternative 1: {row.choice1}\n'
                    f'Alternative 2: {row.choice2}\n'
                    f'Answer: Alternative'), self.token_set_dict
            ),
            
            'explanation': (
                lambda row : (
                    EXPLANATION +
                    f'\n\nPremise: {row.premise} {question_dict[row.question]}\n'
                    f'Alternative 1: {row.choice1}\n'
                    f'Alternative 2: {row.choice2}\n'
                    f'Answer: Alternative'), self.token_set_dict
            ),

        }

        return templates


if __name__ == '__main__':
    copa = CopaDataset()