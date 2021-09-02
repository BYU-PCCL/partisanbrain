from stats import cramers_v_from_vecs, list_to_val_map

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns


class ResultExplorer:
    """For getting statistics and visualizing results of an experiment"""

    def __init__(self, result_fname, ds):
        # Results will be a dictionary. Index into first level
        # is DV name. Index into second level is row index.
        with open(result_fname, "rb") as f:
            self._results = pickle.load(f)

        self._ds = ds
        self._summary_dfs = self._make_summary_dfs()

    def _get_score_dict(self, dv_name, row_idx):
        resp = self._results[dv_name][row_idx][1]
        logit_dict = resp["choices"][0]["logprobs"]["top_logprobs"][0]
        answer_map = self._ds._get_col_prompt_specs()[dv_name].answer_map
        possible_vals = [val.split(" ")[0].lstrip().lower()
                         for val in list(answer_map.values())]
        logit_dict = {k: v for (k, v) in logit_dict.items()
                      if k.lstrip().lower() in possible_vals}
        logit_dict = {k: np.exp(v) for (k, v) in logit_dict.items()}
        val_sum = sum(list(logit_dict.values()))
        logit_dict = {k: v/val_sum for (k, v) in logit_dict.items()}
        score_dict = dict(zip(possible_vals, [0]*len(possible_vals)))
        # TODO: Softmax then combine or combine then softmax?
        for k, v in logit_dict.items():
            score_dict[k.lstrip().lower()] += v
        return score_dict

    def _sample(self, dv_name, row_idx):
        """Returns sampled GPT-3 answer lowercased"""
        score_dict = self._get_score_dict(dv_name, row_idx)
        possible_vals = list(score_dict.keys())
        opt_scores = [score_dict[opt] for opt in possible_vals]
        chosen = np.random.choice(possible_vals, p=opt_scores)
        return chosen

    def _make_summary_dfs(self, seed=0):
        np.random.seed(seed)
        summary_dfs = dict()
        demographics = self._ds.demographics
        for (dv_name, dv_series) in self._ds.dvs.items():

            # Get demographics for this dv
            dv_demographics = demographics.loc[dv_series.index]

            # Get human responses for this dv
            human_resp = [r[-1] for r in list(self._results[dv_name].values())]
            human_resp = [r.lstrip().lower() for r in human_resp]

            # Get GPT-3 responses for this dv with sampling
            gpt_3_resp = [self._sample(dv_name, i) for i in dv_series.index]

            # Get GPT-3 probabilities
            gpt_3_probs = [self._get_score_dict(dv_name, i)
                           for i in dv_series.index]

            # Make df and add to summary_dfs
            df = dv_demographics.assign(human=pd.Series(human_resp).values)
            df = df.assign(gpt_3=pd.Series(gpt_3_resp).values)
            df = df.assign(gpt_3_probs=pd.Series(gpt_3_probs).values)
            summary_dfs[dv_name] = df

        return summary_dfs

    def summary_dfs_to_excel(self, fname):
        with pd.ExcelWriter(fname) as writer:
            for (dv_name, df) in self._summary_dfs.items():
                df.to_excel(writer, sheet_name=dv_name, index=False)

    def get_raw_accs(self):
        raw_accs = dict()
        for dv_name in self._ds.dvs.keys():
            df = self._summary_dfs[dv_name]
            human_responses = []
            for item in df["human"]:
                human_responses.append(item.split(" ")[0].lstrip().lower())
            matches = sum(np.array(human_responses) == df["gpt_3"])
            raw_accs[dv_name] = matches / len(df)
        return raw_accs

    def get_cramers_v_values(self):
        # # Rows are demographics, columns are DVs
        # dvs = list(self._ds.dvs.keys())
        # dv_map = list_to_val_map(dvs)
        # demographic_map = list_to_val_map(list(self._ds.demographics))
        # tbl = np.zeros((len(demographic_map), len(dv_map)))
        # for dv_name in self._ds.dvs.keys():
        #     for demographic_name in list(self._ds.demographics):
        #         demographic = self._summary_dfs[dv_name][demographic_name]
        #         dv = self._summary_dfs[dv_name]["human"]
        #         v = cramers_v_from_vecs(demographic, dv)
        #         tbl[demographic_map[demographic_name], dv_map[dv_name]] = v

        # # Show plot
        # sns.heatmap(tbl,
        #             annot=True,
        #             xticklabels=dvs,
        #             yticklabels=list(self._ds.demographics))
        # plt.show()

        # Rows are demographics, columns are DVs
        dvs = list(self._ds.dvs.keys())
        dv_map = list_to_val_map(dvs)
        demographic_map = list_to_val_map(list(self._ds.demographics))
        tbl_1 = np.zeros((len(demographic_map), len(dv_map)))
        for dv_name in self._ds.dvs.keys():
            for demographic_name in list(self._ds.demographics):
                demographic = self._summary_dfs[dv_name][demographic_name]
                dv = self._summary_dfs[dv_name]["human"]
                v = cramers_v_from_vecs(demographic, dv)
                tbl_1[demographic_map[demographic_name], dv_map[dv_name]] = v

        # Rows are demographics, columns are DVs
        tbl_2 = np.zeros((len(demographic_map), len(dv_map)))
        for dv_name in self._ds.dvs.keys():
            for demographic_name in list(self._ds.demographics):
                demographic = self._summary_dfs[dv_name][demographic_name]
                dv = self._summary_dfs[dv_name]["gpt_3"]
                v = cramers_v_from_vecs(demographic, dv)
                tbl_2[demographic_map[demographic_name], dv_map[dv_name]] = v

        # Show plot
        sns.heatmap(np.abs(tbl_1-tbl_2),
                    annot=True,
                    xticklabels=dvs,
                    yticklabels=list(self._ds.demographics))
        plt.show()


if __name__ == "__main__":
    from pew_american_trends_78_dataset import PewAmericanTrendsWave78Dataset
    re = ResultExplorer("patsy_87_davinci_200.pkl",
                        PewAmericanTrendsWave78Dataset())
    # re.summary_dfs_to_excel("output.xlsx")
    # print(re.get_raw_accs())
    print(re.get_cramers_v_values())
