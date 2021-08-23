from opener import Opener

import abc
import random


class Dataset(abc.ABC):
    """Base class for datasets."""

    def __init__(self, fname, n_exemplars, opening_func=None):

        # Load data into pandas DataFrame
        self._data = Opener().open(fname, opening_func=opening_func)

        # Filter down to relevant rows and columns
        # and format them properly
        self._data = self._format(self._data)

        # Choose exemplars and make exemplar string
        exemplar_idxs = self._get_exemplar_idxs(n_exemplars)
        self._exemplars = self._data.loc[self._data.index.isin(exemplar_idxs)]
        self._data = self._data.loc[~self._data.index.isin(exemplar_idxs)]

        # Get row and exemplar backstories (so they only need
        # to be calculated once)
        self._row_backstories = {idx: self._make_backstory(row) for (idx, row)
                                 in self._data.iterrows()}
        self._exemplar_backstories = [self._make_backstory(row) for (_, row)
                                      in self._exemplars.iterrows()]

    @property
    def data(self):
        return self._data

    @property
    def exemplars(self):
        return self._exemplars

    @property
    def row_backstories(self):
        return self._row_backstories

    @property
    def exemplar_backstories(self):
        return self._exemplar_backstories

    @abc.abstractmethod
    def _format(self, df):
        """
        Here subclass should implement filtering rows and
        columns, handling NA, and anything else related to
        the dataframe. Return the resulting dataframe. DO
        NOT modify df in place - return a new dataframe. ALL
        modifying of column values (binning, etc.) should
        happen here (not in the _make_backstory or _make_prompts
        methods).
        """
        pass

    @abc.abstractmethod
    def _make_backstory(self, row):
        """
        Here subclass should use taken row to make the demographics
        based backstory. This should not include DV information. This
        method is only abstract because this method will help make
        the _make_prompts method cleaner.
        """
        pass

    def _make_prompt(self, row_idx, col_name, func):
        row_backstory = self._row_backstories[row_idx]

        exemplar_dv_strs = [func(e[col_name]) for (_, e)
                            in self._exemplars.iterrows()]
        blank_dv_str = func(None)
        prompt = ""
        for i in range(len(self._exemplars)):
            prompt += self._exemplar_backstories[i] + " "
            prompt += exemplar_dv_strs[i] + "\n"
        prompt += row_backstory + " "
        prompt += blank_dv_str

        return prompt

    @abc.abstractmethod
    def _make_prompts(self, row_idx):
        """
        Here subclass should implement converting a pandas row
        and exemplars dataframe to a list of prompt strings. This
        is a list because each subclass will have many prompts
        for each respondent because there are many DVs.
        """
        pass

    def _get_exemplar_idxs(self, n):
        return random.sample(range(len(self._data)), n)
