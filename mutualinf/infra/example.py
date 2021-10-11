from collections import defaultdict
from dataset import Dataset

import pandas as pd


class ExampleDataset(Dataset):

    def __init__(self):
        super().__init__()

    def _modify_raw_data(self, df):
        mod_df_dict = defaultdict(list)
        for _, row in df.iterrows():
            mod_df_dict["product"].append(row["q"].split(" ")[4])
            mod_df_dict["ground_truth"].append(row["a"])
        return pd.DataFrame(mod_df_dict, index=df.index)

    def _get_templates(self):
        templates = {
            "sells": lambda row: ("The company that sells "
                                  f"{row['product']} is "),
            "sold_by": lambda row: (f"{row['product'].capitalize()} "
                                    "are sold by "),
        }
        return templates


if __name__ == "__main__":
    ed = ExampleDataset()
