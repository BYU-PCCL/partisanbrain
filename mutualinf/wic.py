from collections import defaultdict
from dataset import Dataset

import sys
import pandas as pd


SHOTS = [
    """Word: bright
Usage 1: He is a bright child
Usage 2: The sun is very bright today
Meaning: different

""",
    """Word: air
Usage 1: Utah has too much air pollution.
Usage 2: Open a window and let in some air.
Meaning: same

""",
    """Word: cool
Usage 1: Her pants are cool.
Usage 2: Let your food cool.
Meaning: different

""",
    """Word: fight
Usage 1: My wife and I had a fight.
Usage 2: I fight for my freedom.
Meaning: same

""",
]

LOGIC_QUESTIONS = [
    """Q: What does 2 + 2 equal?
A: 4

""",
    """Q: If you are 60 inches tall how tall are you in feet?
A: 5 feet

""",
]

FACT_QUESTIONS = [
    """Q: What year did America first land on the moon?
A: 1969

""",
    """Q: What is the average height in America?
A: 5 feet 9 inches

""",
]

YES_NO_QUESTIONS = [
    """Q: Is the United States in South America?
A: No

""",
    """Q: Is the following sentence missing a comma? Before leaving I ate breakfast.
A: Yes

""",
]


class WicDataset(Dataset):

    def __init__(self):
        self._token_set = {
            'question': {
                'True': ['yes'],
                'False': ['no'],
            },
            'true_false_classify': {
                'True': ['true', 'correct', 'implied'],
                'False': ['false', 'incorrect'],
            },
            'few_shot': {
                'True': ['same'],
                'False': ['different'],
            },
        }
        super().__init__()

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):
        templates = {
            'question0': (
                lambda row: (
                    f'{row.sentence1} {row.sentence2} '
                    f'Choose "yes" or "no". Does the word {row.word} have the same context in the previous sentences? "'
                ), self._token_set['question']
            ),
            'few_shot0': (
                lambda row: (
                    SHOTS[0] +
                    f'Word: {row.word}\n'
                    f'Usage 1: {row.sentence1}\n'
                    f'Usage 2: {row.sentence2}\n'
                    f'Meaning:'
                ), self._token_set['few_shot']
            ),
            'few_shot1': (
                lambda row: (
                    ''.join(SHOTS[:2]) +
                    f'Word: {row.word}\n'
                    f'Usage 1: {row.sentence1}\n'
                    f'Usage 2: {row.sentence2}\n'
                    f'Meaning:'
                ), self._token_set['few_shot']
            ),
            'few_shot2': (
                lambda row: (
                    ''.join(SHOTS[:3]) +
                    f'Word: {row.word}\n'
                    f'Usage 1: {row.sentence1}\n'
                    f'Usage 2: {row.sentence2}\n'
                    f'Meaning:'
                ), self._token_set['few_shot']
            ),
            'few_shot3': (
                lambda row: (
                    ''.join(SHOTS[:4]) +
                    f'Word: {row.word}\n'
                    f'Usage 1: {row.sentence1}\n'
                    f'Usage 2: {row.sentence2}\n'
                    f'Meaning:'
                ), self._token_set['few_shot']
            ),
            'question_answer_logic0': (
                lambda row: (
                    ''.join(LOGIC_QUESTIONS[:1]) +
                    f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'{row.sentence1} {row.sentence2}\n'
                    f'A:'
                ), self._token_set['question']
            ),
            'question_answer_logic1': (
                lambda row: (
                    ''.join(LOGIC_QUESTIONS[:2]) +
                    f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'{row.sentence1} {row.sentence2}\n'
                    f'A:'
                ), self._token_set['question']
            ),
            'question_answer_fact0': (
                lambda row: (
                    ''.join(FACT_QUESTIONS[:1]) +
                    f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'{row.sentence1} {row.sentence2}\n'
                    f'A:'
                ), self._token_set['question']
            ),
            'question_answer_fact1': (
                lambda row: (
                    ''.join(FACT_QUESTIONS[:2]) +
                    f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'{row.sentence1} {row.sentence2}\n'
                    f'A:'
                ), self._token_set['question']
            ),
            'question_answer_yes_no0': (
                lambda row: (
                    ''.join(YES_NO_QUESTIONS[:1]) +
                    f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'{row.sentence1} {row.sentence2}\n'
                    f'A:'
                ), self._token_set['question']
            ),
            'question_answer_yes_no1': (
                lambda row: (
                    ''.join(YES_NO_QUESTIONS[:2]) +
                    f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'{row.sentence1} {row.sentence2}\n'
                    f'A:'
                ), self._token_set['question']
            ),
            'true_false_classify0': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify1': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}", '
                    f'true or false, '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify2': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is "'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify3': (
                lambda row: (
                    f'{row.sentence1}\n'
                    f'{row.sentence2}\n\n'
                    f'True or false, the word {row.word} has the same context.\n\n'
                    f'Answer:'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify4': (
                lambda row: (
                    f'{row.sentence1}\n'
                    f'{row.sentence2}\n\n'
                    f'"True" or "False", the word {row.word} has the same context.\n\n'
                    f'Answer:'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify5': (
                lambda row: (
                    f'{row.sentence1}\n'
                    f'{row.sentence2}\n\n'
                    f'"True" or "False", the word {row.word} has the same context.\n\n'
                    f'Answer: "'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify6': (
                lambda row: (
                    f'"{row.sentence1}"\n'
                    f'"{row.sentence2}"\n\n'
                    f'True or False, the word "{row.word}" has the same context.\n\n'
                    f'Answer:'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify7': (
                lambda row: (
                    f'True or False, the word "{row.word}" has the same context in the following sentences.\n\n'
                    f'Sentence 1: {row.sentence1}\n'
                    f'Sentence 2: {row.sentence2}\n\n'
                    f'Answer:'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify8': (
                lambda row: (
                    'I am going to answer true or false questions about whether a word that appears in two sentences has the same context or not.\n\n'
                    f'True or False, the word "{row.word}" has the same context in the following sentences.\n\n'
                    f'Sentence 1: {row.sentence1}\n'
                    f'Sentence 2: {row.sentence2}\n\n'
                    f'Answer:'
                ), self._token_set['true_false_classify']
            ),
        }

        return templates


if __name__ == "__main__":
    # Data should be at data/wic/raw.csv
    wd = WicDataset()
