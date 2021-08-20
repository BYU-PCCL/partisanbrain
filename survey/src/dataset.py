from opener import Opener

import abc


class Dataset(abc.ABC):

    def __init__(self, fname, opening_func=None):

        # Load data into pandas DataFrame
        df = Opener().open(fname, opening_func=opening_func)

        # Reduce down to relevant rows and columns
        pass
