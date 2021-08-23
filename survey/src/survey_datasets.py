from dataset import Dataset


# This is an example that will be removed
class PewDataset(Dataset):

    def __init__(self, n_exemplars):
        pew_fname = "../data/Pew Research Center Spring 2016 Global Attitudes Dataset WEB FINAL.sav"
        super().__init__(pew_fname, n_exemplars)

    def _make_prompt(self, row, exemplars):
        return "This is a prompt"

    def _format(self, df):
        return df[df["country"] == "United States"]


if __name__ == '__main__':
    ds = PewDataset(n_exemplars=5)
