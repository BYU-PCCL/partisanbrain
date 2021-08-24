from collections import defaultdict

import pickle


class ResultExplorer:
    """For getting statistics and visualizing results of an experiment"""

    def __init__(self, result_fname):
        with open(result_fname, "rb") as f:
            self._results = pickle.load(f)

    def get_raw_accuracy(self):
        matches = defaultdict(int)
        totals = defaultdict(int)
        for (_, prompts_dict) in self._results.items():
            for (col_name, prt) in prompts_dict.items():
                _, response, target = prt
                top_answer = response["choices"][0]["logprobs"]["tokens"][0]
                top_answer = top_answer.strip().lower()
                target = target.lower()
                matches[col_name] += top_answer == target
                totals[col_name] += 1
        raw_acc = {k: matches[k] / totals[k] for (k, _) in matches.items()}
        print(raw_acc)


if __name__ == "__main__":
    re = ResultExplorer("star_wars_results.pkl")
    re.get_raw_accuracy()
