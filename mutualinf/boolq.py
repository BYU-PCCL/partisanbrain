from collections import defaultdict
from dataset import Dataset

import pandas as pd

SHOTS = [

"""Passage:"Turn on red -- In Canada, left turn on red light from a one-way road into a one-way road is permitted except in some areas of Quebec, New Brunswick, and Prince Edward Island. Left turn on red light from a two-way road into a one-way road is permitted in British Columbia but only if the driver turns onto the closest lane and yields to pedestrians and cross traffic."
Question:"can you turn left on red in canada?"
Answer:"yes"

""",

"""Passage:"Lord Voldemort -- Lord Voldemort ( known as Tom Marvolo Riddle) is a fictional character and the main antagonist in J.K. Rowling's series of Harry Potter novels. Voldemort first appeared in Harry Potter and the Philosopher's Stone, which was released in 1997. Voldemort appears either in person or in flashbacks in each book and its film adaptation in the series, except the third, Harry Potter and the Prisoner of Azkaban, where he is only mentioned."
Question:"are tom riddle and lord voldemort the same person?"
Answer:"yes"

""",

"""Passage:"Clerks -- Clerks is a 1994 American independent black-and-white comedy film written, directed and co-produced by Kevin Smith. Starring Brian O'Halloran as Dante Hicks and Jeff Anderson as Randal Graves, it presents a day in the lives of two store clerks and their acquaintances."
Question:"is the movie clerks in colors?"
Answer:"no"

""",
]
class BoolqDataset(Dataset):

    def __init__(self):
        self._token_set = self._token_set = {
            'True/False_classify': {
                'True': ['true','True','yes','Yes','YES'],
                'False': ['false','False','no','No','NO'],
            },
        }
        super().__init__()

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):
        templates = {
            'No_choices_1': (
                lambda row: (
                            f'Read the following passage:"{row.passage}"\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'I would answer:"'
                        ), self._token_set['True/False_classify']
            ),

            'No_choices_2': (
                lambda row: (
                            f'Read the following passage:"{row.passage}"\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'I would respond:"'
                        ),self._token_set['True/False_classify']
            ),

            'No_choices_3': (
                lambda row: (
                            f'Based on the passage:"{row.passage}"\n\n'
                            f'Answering the question:"{row.question}?"\n'
                            f'My answer would be:"'
                        ),self._token_set['True/False_classify']
            ),

            'few_shot1': (
                lambda row: (
                    SHOTS[0] +
                    f'Passage:"{row.passage}"\n'
                    f'Question:"{row.question}?"\n'
                    f'Answer:"'
                ), self._token_set['True/False_classify']
            ),

            'few_shot2': (
                lambda row: (
                    ''.join(SHOTS[:2]) +
                    f'Passage:"{row.passage}"\n'
                    f'Question:"{row.question}?"\n'
                    f'Answer:"'
                ), self._token_set['True/False_classify']
            ),

            'few_shot3': (
                lambda row: (
                    ''.join(SHOTS[:3]) +
                    f'Passage:"{row.passage}"\n'
                    f'Question:"{row.question}?"\n'
                    f'Answer:"'
                ), self._token_set['True/False_classify']
            ),

            'QuestionPhrasing-1': (
                lambda row: (
                
                            f'Read the following passage:"{row.passage}"\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'If asked to choose "true" or "false", '
                            f'I would answer:"'
                        ),self._token_set['True/False_classify']
            ),

            'QuestionPhrasing-2': (
                lambda row: (
                            f'Read the following passage:"{row.passage}"\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'If asked to choose true/false, '
                            f'I would answer:"'
                        ),self._token_set['True/False_classify']
            ),

            'QuestionPhrasing-3': (
                lambda row: (
                            f'Read the following passage:"{row.passage}"\n\n'
                            f'Given this question:"{row.question}?"\n'
                            f'If asked to choose yes or no, '
                            f'My answer would be:"'
                        ),self._token_set['True/False_classify']
            ),

            'QuestionPhrasing-4': (
                lambda row: (
                            f'"{row.passage}"\n\n '
                            f'When picking between yes or no '
                            f'for the question:"{row.question}"\n'
                            f'I would say:"'
                        ),self._token_set['True/False_classify']
            ),

            'PromptAnswerPhrasing-1': (
                lambda row: (
                
                            f'Read the following passage:"{row.passage}"\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'If asked to choose yes or no, '
                            f'I would answer:"'
                        ),self._token_set['True/False_classify']
            ),

            'PromptAnswerPhrasing-2': (
                lambda row: (
                            f'Read the following passage:"{row.passage}"\n\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'If asked to choose "true" or "false", '
                            f'I would respond:"'
                        ),self._token_set['True/False_classify']
            ),

            'PromptAnswerPhrasing-3': (
                lambda row: (
                            f'Read the following passage:"{row.passage}\n\n'
                            f'Given this question:"{row.question}"\n\n'
                            f'If asked to choose "true" or "false", '
                            f'My answer would be:"'
                        ),self._token_set['True/False_classify']
            ),

            'Passage_Question_Directly_1': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'For the question:"{row.question}"\n'
                            f'My answer would be:"'
                        ),self._token_set['True/False_classify']
            ),

            'Passage_Question_Directly_2': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'for the question:"{row.question}"\n'
                            f'I would answer:"'
                        ),self._token_set['True/False_classify']
            ),
            
            'Passage_definingProblem_PromptingAnswer-1': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'When picking between true/false '
                            f'for the question:"{row.question}?"\n'
                            f'Answer:"'
                        ),self._token_set['True/False_classify']
            ),

            'Passage_definingProblem_PromptingAnswer-2': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'When picking between "true" or "false", '
                            f'for the question:"{row.question}?"\n'
                            f'My answer would be: "'
                        ),self._token_set['True/False_classify']
            ),

            'Passage_definingProblem_PromptingAnswer-3': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'When picking between true or false '
                            f'for the question:"{row.question}"\n'
                            f'I would answer:"'
                        ),self._token_set['True/False_classify']
            ),

            'Passage_definingProblem_PromptingAnswer-4': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'When picking between yes or no '
                            f'for the question:"{row.question}?"\n'
                            f'I would answer:"'
                        ),self._token_set['True/False_classify']
            ),

            'Passage_definingProblem_PromptingAnswer-5': (
                lambda row: (
                            f'"{row.passage}"\n\n'
                            f'When picking between yes or no '
                            f'for the question:"{row.question}?"\n'
                            f'I would respond:"'
                        ),self._token_set['True/False_classify']
            ),

            'Specific_1_yes/no': (
                lambda row: (
                            f'Based on the passage:"{row.passage}"\n\n'
                            f'And answering the question:"{row.question}?"\n'
                            f'By choosing yes or no\n'
                            f'My answer would be:"'
                        ),self._token_set['True/False_classify']
            ),

            'Specific_2_true/False': (
                lambda row: (
                            f'Based on the passage:"{row.passage}"\n\n'
                            f'And answering the question:"{row.question}"\n'
                            f'By choosing true or no\n'
                            f'My answer would be:"'
                        ),self._token_set['True/False_classify']
            ),
        }
        return templates


if __name__ == "__main__":
    # Data should be at data/example/raw.csv
    bd = BoolqDataset()

