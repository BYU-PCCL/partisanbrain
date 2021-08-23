import pickle


class ResultExplorer:
    """For getting statistics and visualizing results of an experiment"""

    def __init__(self, result_fname):
        self._results = pickle.load(result_fname)
