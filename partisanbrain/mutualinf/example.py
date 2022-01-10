from collections import defaultdict
from infra_modules import Dataset

import pandas as pd


class ExampleDataset(Dataset):

    def __init__(self, sample_seed=0, n=None):

        # Each template should have a token set
        # of the form {ground_truth_term: synonyms_list}.
        # It is a best guess at what close synonyms for
        # each ground truth term would be. If there's
        # no practical best guess (as in this example)
        # just make the token set None.
        self._token_set = None

        super().__init__(sample_seed=sample_seed,
                         n=n)

    def _modify_raw_data(self, df):
        mod_df_dict = defaultdict(list)
        for _, row in df.iterrows():
            mod_df_dict["product"].append(row["q"].split(" ")[4])
            mod_df_dict["ground_truth"].append(row["a"])
        return pd.DataFrame(mod_df_dict, index=df.index)

    def _get_templates(self):
        templates = {
            "sells": (lambda row: ("The company that sells "
                                   f"{row['product']} is"), self._token_set),
            "sold_by": (lambda row: (f"{row['product'].capitalize()} "
                                     "are sold by"), self._token_set),
        }
        return templates


if __name__ == "__main__":
    # Data should be at data/example/raw.csv
    ed = ExampleDataset()
