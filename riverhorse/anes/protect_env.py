from dataset import Dataset


class ProtectEnvDataset(Dataset):

    def __init__(self, sample_seed=0, n=None):
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        # Rename columns for clarity
        mod_df = df.rename(columns={"2348234": "something political"})
        # Replace numeric values with strings for clarity
        mod_df["bla"] = mod_df["bla"].str.replace("yikes", "yes")
        return mod_df

    def _get_templates(self):
        templates = {
            "review_follow_up_q0": (lambda row: (f"{row['review']}\n\n"
                                    "Was the previous review positive or negative? The previous review was"), self.token_set_dict)
        }
        return templates


if __name__ == "__main__":
    # Data is in data/anes/raw.csv
    ped = ProtectEnvDataset()
