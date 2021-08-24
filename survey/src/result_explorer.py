import pickle


class ResultExplorer:
    """For getting statistics and visualizing results of an experiment"""

    def __init__(self, result_fname):
        with open(result_fname, "rb") as f:
            self._results = pickle.load(f)
        print(self._results)


if __name__ == "__main__":
    re = ResultExplorer("star_wars_results.pkl")
