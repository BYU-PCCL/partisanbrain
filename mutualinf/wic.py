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


class WicDataset(Dataset):

    def __init__(self):
        self._token_set = {
            'question': {
                'True': ['yes'],
                'False': ['no'],
            },
            'true_false_classify': {
                'True': ['true'],
                'False': ['false'],
            },
            'similar_different_classify': {
                'True': ['similar'],
                'False': ['different'],
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
            # Add QA prompt



            'true_false_classify0': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify1': (
                lambda row: (
                    f'In the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify2': (
                lambda row: (
                    f'Given the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify3': (
                lambda row: (
                    f'After reading the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify4': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify5': (
                lambda row: (
                    f'In the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify6': (
                lambda row: (
                    f'Given the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify7': (
                lambda row: (
                    f'After reading the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify8': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and given options "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'true_false_classify9': (
                lambda row: (
                    f'In the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and given options "true" or "false", '
                    f'the statement "the word {row.word} has the same context" is'
                ), self._token_set['true_false_classify']
            ),
            'similar_different_classify0': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify1': (
                lambda row: (
                    f'In the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify2': (
                lambda row: (
                    f'Given the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify3': (
                lambda row: (
                    f'After reading the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify4': (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify5': (
                lambda row: (
                    f'In the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify6': (
                lambda row: (
                    f'Given the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
            'similar_different_classify7': (
                lambda row: (
                    f'After reading the following sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing between "similar" or "different", '
                    f'in the sentences I read, the context of the word "{row.word}" is'
                ), self._token_set['similar_different_classify']
            ),
        }

        return templates


if __name__ == "__main__":
    # Data should be at data/wic/raw.csv
    wd = WicDataset()
