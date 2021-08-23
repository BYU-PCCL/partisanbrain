from dataset import Dataset


# This is an example that will be removed
class PewDataset(Dataset):

    def __init__(self, n_exemplars):
        pew_fname = "../data/Pew Research Center Spring 2016 Global Attitudes Dataset WEB FINAL.sav"
        super().__init__(pew_fname, n_exemplars)

    def _make_backstory(self, row):
        return f"I am from {row['country']}"

    def _make_prompts(self, row, exemplars):
        return self._make_backstory(row)

    def _format(self, df):
        return df[df["country"] == "United States"]


if __name__ == '__main__':
    ds = PewDataset(n_exemplars=5)
    print(ds._make_prompts(ds.data.iloc[0], ds.exemplars))
