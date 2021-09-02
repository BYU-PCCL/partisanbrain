from stats import cramers_v_from_vecs, list_to_val_map

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from matplotlib import pyplot as plt

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
        possible_vals = [val.lower() for val in list(answer_map.values())]
        logit_dict = {k: v for (k, v) in logit_dict.items()
                      if k.strip().lower() in possible_vals}
        logit_dict = {k: np.exp(v) for (k, v) in logit_dict.items()}
        val_sum = sum(list(logit_dict.values()))
        logit_dict = {k: v/val_sum for (k, v) in logit_dict.items()}
        score_dict = dict(zip(possible_vals, [0]*len(possible_vals)))
        for k, v in logit_dict.items():
            score_dict[k.strip().lower()] += v
        return score_dict

    def _sample(self, dv_name, row_idx):
        """Returns sampled GPT-3 answer lowercased"""
        score_dict = self._get_score_dict(dv_name, row_idx)
        possible_vals = list(score_dict.keys())
        opt_scores = [score_dict[opt] for opt in possible_vals]
        chosen = np.random.choice(possible_vals, 1, opt_scores)[0]
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
            human_resp = [r.strip().lower() for r in human_resp]

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
            raw_accs[dv_name] = (sum(df["human"] == df["gpt_3"])) / len(df)
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


    def average_demographics(self):
        demographics = self._ds.demographics
        human_dic = {
            "econ_today" : {
                "Excellent": 1,
                "Good": 0.666,
                "Only fair": .333,
                "Poor": 0,

            },
            "econ_year_away" : {
                "Worse": 0,
                "About the same as now": 0.5,
                "Better": 1,
            },
            "country_satisfied" : {
                "Dissatisfied": 0,
                "Satisfied": 1,
            },
            "election_wellness" : {
                "Not at all well" : 0,
                "Not too well" :0.333,
                "Somewhat well" : 0.666,
                "Very well" : 1,
            },
            "follow_election" : {
                'Followed them almost constantly': 1.0,
                'Checked in fairly often': 0.666,
                'Checked in occasionally': 0.333,
                'Tuned them out entirely': 0,
            },
            "covid_assist_pack" : {
                'Necessary': 1.0,
                'Not necessary': 0,
            },
            "rep_dem_relationship" : {
                'Get better': 1.0,
                'Stay about the same' : 0.5,
                'Get worse' : 0,
            },
            "covid_restrict" : {
                'MORE restrictions right now': 1.0, 
                'About the same number of restrictions right now': 0.5,
                'FEWER restrictions right now':0,
            },
            "rep_dem_division" : {
                'Very concerned' : 1.0,
                'Somewhat concerned' : 0.666,
                'Not too concerned' : 0.333,
                'Not at all concerned' : 0,
            },
            "more_votes_better" : {
                "The country would be better off if more Americans voted": 1.0,
                "The country would not be better off if more Americans voted": 0,
            },
        }
        gpt3_dic = {
            "econ_today" : {
                "excellent": 1,
                "good": 0.666,
                "fair": .333,
                "poor": 0,

            },
            "econ_year_away" : {
                "worse": 0,
                "same as now": 0.5,
                "better": 1,
            },
            "country_satisfied" : {
                "dissatisfied": 0,
                "satisfied": 1,
            },
            "election_wellness" : {
                "well" : 1,
                "poorly" :0.0,
            },
            "follow_election" : {
                'yes': 1.0,
                'no':0,
            },
            "covid_assist_pack" : {
                'yes':1.0,
                'no':0.0,
            },
            "rep_dem_relationship" : {
                'better': 1.0,
                'same as now' : 0.5,
                'worse' : 0,
            },
            "covid_restrict" : {
                'increased': 1.0, 
                'maintained': 0.5,
                'decreased':0,
            },
            "rep_dem_division" : {
                'yes' : 1.0,
                'no' : 0.0,
            },
            "more_votes_better" : {
                'yes' : 1.0,
                'no' : 0.0,
            },
        }

        for dv_name, dv_series in self._ds.dvs.items():
            labels = []
            avg_gpt3s = []
            avg_humans = []
            df = self._summary_dfs[dv_name]
            i = 0
            for demo in demographics:
                for demo_val in df[demo].unique():
                    i+=1
                    demo_df = df[df[demo] == demo_val]
                    humancol = 'human'
                    gpt_3_col = 'gpt_3'
                    if i == 1:
                        bp()

                    humanmapping = demo_df[humancol].map(gpt3_dic[dv_name])
                    gpt3mapping = demo_df[gpt_3_col].map(gpt3_dic[dv_name])

                    #Check columns humancol and gpt_3_col in demo_df for nans 
                    if humanmapping.isna().any():
                        print('NANS!')
                    if gpt3mapping.isna().any():
                        print('NANS!')

                    avg_human = humanmapping.mean()
                    avg_gpt3 = gpt3mapping.mean()






                    # for key in gpt3_dic[dv_name]:
                    #     demo_df['gpt_3_probs']

                    # Make a variable called label that concatenates demo and demo_val
                    label = demo + "=" + demo_val

                    labels.append(label)
                    avg_gpt3s.append(avg_gpt3)
                    avg_humans.append(avg_human)
                    plt.scatter(avg_gpt3, avg_human)
                    # plt.annotate(label, (avg_gpt3, avg_human))
            print(f"Correlation = {np.corrcoef(avg_gpt3s, avg_humans)[0,1]}")
            plt.title(dv_name)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.savefig(f'experiments/{dv_name}.jpeg')
            plt.close()





if __name__ == "__main__":
    from pew_american_trends_78_dataset import PewAmericanTrendsWave78Dataset
    re = ResultExplorer("patsy_87_davinci_200.pkl",
                        PewAmericanTrendsWave78Dataset())
    re.average_demographics()
    # re.summary_dfs_to_excel("output.xlsx")
    # print(re.get_raw_accs())
    print(re.get_cramers_v_values())
