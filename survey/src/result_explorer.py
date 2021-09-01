import numpy as np
import pickle


class ResultExplorer:
    """For getting statistics and visualizing results of an experiment"""

    def __init__(self, result_fname, ds):
        # Results will be a dictionary. Index into first level
        # is DV name. Index into second level is row index.
        with open(result_fname, "rb") as f:
            self._results = pickle.load(f)

        self._ds = ds

    def _sample(self, dv_name, row_idx, seed=0):
        """Returns sampled GPT-3 answer lowercased"""
        resp = self._results[dv_name][row_idx][1]
        logit_dict = resp["choices"][0]["logprobs"]["top_logprobs"][0]
        answer_map = self._ds._get_col_prompt_specs()[dv_name].answer_map
        possible_vals = [val.lower() for val in list(answer_map.values())]
        logit_dict = {k: v for (k, v) in logit_dict.items()
                      if k.strip().lower() in possible_vals}
        logit_dict = {k: np.exp(v) for (k, v) in logit_dict.items()}
        val_sum = sum(list(logit_dict.values()))
        logit_dict = {k: v/val_sum for (k, v) in logit_dict.items()}
        score_dict = dict(zip(possible_vals, [0]*len(possible_vals)))
        for k, v in logit_dict.items():
            score_dict[k.strip().lower()] += v
        opt_scores = [score_dict[opt] for opt in possible_vals]
        chosen = np.random.choice(possible_vals, 1, opt_scores)[0]
        return chosen

    def make_summary_dfs(self):
        summary_dfs = dict()
        demographics = self._ds.demographics
        for (dv_name, dv_series) in self._ds.dvs.items():

            # Get demographics for this dv
            dv_demographics = demographics.loc[dv_series.index]

            # Get human responses for this dv
            human_resp = [r[-1] for r in list(self._results[dv_name].values())]

            # Get GPT-3 responses for this dv with sampling
            # gpt_3_resp = [r[1] for r in list(self._results[dv_name].values())]
            # gpt_3_resp = [r["choices"][0]["logprobs"]["top_logprobs"][0]
            #               for r in gpt_3_resp]
            # print(gpt_3_resp[0])


            # gpt_3_resp = [r[-1] for r in ]


if __name__ == "__main__":
    from pew_american_trends_78_dataset import PewAmericanTrendsWave78Dataset
    re = ResultExplorer("patsy_87_davinci_200.pkl",
                        PewAmericanTrendsWave78Dataset())
    re.make_summary_dfs()
