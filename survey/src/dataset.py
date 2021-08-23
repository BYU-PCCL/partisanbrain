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
        exemplars = self._data.loc[self._data.index.isin(exemplar_idxs)]
        self._data = self._data.loc[~self._data.index.isin(exemplar_idxs)]
        exemplars_list = [value for (_, value) in exemplars.iterrows()]
        exemplar_prompt_strs = [self._make_prompt(e) for e in exemplars_list]
        self._exemplar_str = "\n".join(exemplar_prompt_strs)

    @property
    def data(self):
        return self._data

    @property
    def exemplar_str(self):
        return self._exemplar_str

    @abc.abstractmethod
    def _format(self, df):
        """
        Here subclass should implement filtering rows and
        columns, handling NA, and anything else related to
        the dataframe. Return the resulting dataframe. DO
        NOT modify df in place - return a new dataframe.
        """
        pass

    @abc.abstractmethod
    def _make_backstory(self, row):
        """
        Here subclass should use taken row to make the demographics
        based backstory. This should not include DV information.
        """
        pass

    @abc.abstractmethod
    def _make_prompts(self, row, exemplar_str=None):
        """
        Here subclass should implement converting a pandas row
        and exemplars string to a list of prompt strings. This
        is a list because each subclass will have many prompts
        for each respondent because there are many DVs.
        """
        pass

    def _get_exemplar_idxs(self, n):
        return random.sample(range(len(self._data)), n)
