import os.path
import pandas as pd


class Opener:
    """Utility class for loading a file into pandas DataFrame"""

    def __init__(self):
        self._opening_funcs = {
            "csv": self._load_csv,
            "pkl": self._load_pickled_df,
            "pickle": self._load_pickled_df,
            "sav": pd.read_spss,
            "dta": pd.read_stata
        }

    def _load_csv(self, fname):
        return pd.read_csv(fname, encoding="unicode_escape")

    def _load_pickled_df(self, fname):
        return pd.read_pickle(fname)

    def _get_file_type(self, fname):
        return fname.split(".")[-1]

    def open(self, fname, opening_func=None):
        """
        opening_func is for a custom function to use to turn the file
        specified by fname into a pandas DataFrame. If not included,
        opening_func will be chosen from a list of reasonable
        defaults.
        """

        if opening_func is None:
            ftype = self._get_file_type(fname)

            # Get opening function if available
            try:
                opening_func = self._opening_funcs[ftype]
            except KeyError:
                msg = f"Opener class has no function for opening .{ftype} file"
                raise NotImplementedError(msg)

        # Open file if it exists
        if os.path.isfile(fname):
            return opening_func(fname)
        else:
            msg = f"Opener could not open {fname} because it doesn't exist"
            raise FileNotFoundError(msg)
