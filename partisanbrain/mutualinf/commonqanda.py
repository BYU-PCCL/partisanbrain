from collections import defaultdict
from dataset import Dataset

import sys
import pandas as pd
import json

class CommonQADataset(Dataset):

    def __init__(self, sample_seed=0, n=None):
        self._token_set = {}
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):
        templates = {
            'question0': (
                lambda row: (
                    f'{row.question_concept}\n {row.A}\n'
                ), self._token_set
            ),
        }

        return templates


if __name__ == "__main__":
    qd = CommonQADataset()
