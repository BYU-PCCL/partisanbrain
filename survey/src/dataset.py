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

        # Choose exemplars
        exemplar_idxs = self._get_exemplar_idxs(n_exemplars)
        self._exemplars = self._data.loc[self._data.index.isin(exemplar_idxs)]
        self._data = self._data.loc[~self._data.index.isin(exemplar_idxs)]

        # Get row and exemplar backstories (so they only need
        # to be calculated once)
        self._row_backstories = {idx: self._make_backstory(row) for (idx, row)
                                 in self._data.iterrows()}
        self._exemplar_backstories = [self._make_backstory(row) for (_, row)
                                      in self._exemplars.iterrows()]

        # Make a mapping from row index to row's prompts data
        self._prompts_dict = {}
        for idx in self.kept_indices:
            self._prompts_dict[idx] = self._make_prompts(idx)

    @property
    def prompts(self):
        """{row_index: {dv_colname: prompt}}"""
        return self._prompts_dict

    @property
    def kept_indices(self):
        return self._data.index.tolist()

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

    @abc.abstractmethod
    def _get_prompt_instructions(self):
        """
        Here subclass should return a dictionary where each key
        is a column name present in the formatted self._data and
        each value is a tuple consisting of the prompt without
        value and a function for handling value
        (i.e., {column_name: (prompt_str, handler_func)}). For
        example, if we are working with a DV called partisanship
        the prompt string might be "The party I identify most
        with is" (note no space on the end). The function might be
        lambda x: x.upper() if we want values from out dataframe
        to appear capitalized when in a prompt.
        """
        pass

    def _make_prompt(self, row_idx, col_name):
        row_backstory = self._row_backstories[row_idx]
        prompt_str, dv_func = self._get_prompt_instructions()[col_name]

        exemplar_dv_strs = [prompt_str + " " + dv_func(e[col_name]) for (_, e)
                            in self._exemplars.iterrows()]
        prompt = ""
        for i in range(len(self._exemplars)):
            prompt += self._exemplar_backstories[i] + " "
            prompt += exemplar_dv_strs[i] + "\n"
        prompt += row_backstory + " "
        prompt += prompt_str

        return prompt

    def _make_prompts(self, row_idx):
        prompts = {}
        for (col_name, _) in self._get_prompt_instructions().items():
            prompts[col_name] = self._make_prompt(row_idx, col_name)
        return prompts

    def _get_exemplar_idxs(self, n):
        return random.sample(self.kept_indices, n)
