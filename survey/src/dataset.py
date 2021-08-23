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

    @abc.abstractmethod
    def _format(self, df):
        """
        Here subclass should implement filtering rows and
        columns, handling NA, and anything else related to
        the dataframe. Return the resulting dataframe.
        """
        pass

    @abc.abstractmethod
    def _row_to_prompt(self, row):
        """
        Here subclass should implement converting a pandas row
        to a prompt string.
        """
        pass

    def _get_exemplar_idxs(self, n):
        return random.sample(range(len(self._data)), n)


class PewDataset(Dataset):

    def __init__(self, n_exemplars):
        pew_fname = "../data/Pew Research Center Spring 2016 Global Attitudes Dataset WEB FINAL.sav"
        super().__init__(pew_fname, n_exemplars)

    def _row_to_prompt(self, row):
        return "This is a prompt"

    def _format(self, df):
        return df[df["country"] == "United States"]


if __name__ == '__main__':
    ds = PewDataset(n_exemplars=5)
