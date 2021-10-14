from collections import defaultdict
from dataset import Dataset

import sys
import pandas as pd


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
        }
        super().__init__()

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):
        templates = {
            'question0': (
                lambda row: (
                    f'{row.sentence1} {row.sentence2} '
                    f'Does the word "{row.word}" have the same context?'
                ), self._token_set['question']
            ),
            'question1': (
                lambda row: (
                    f'{row.sentence1} {row.sentence2} '
                    f'Does the word "{row.word}" have the same context in the previous sentences?'
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
