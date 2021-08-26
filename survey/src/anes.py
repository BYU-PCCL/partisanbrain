import sys
sys.path.append('src')
import dataset
from pdb import set_trace as bp
import pandas as pd

class AnesDataset(dataset.Dataset):
    def __init__(self, n_exemplars):
        survey_fname = "data/anes/anes_timeseries_2020_csv_20210719.csv"
        #Read this into a pandas df
        df = pd.read_csv(survey_fname)
        bp()
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):

        # Dropping all but relevant columns
        new_df = df[[

                        #Demographics
                     "V201507x", #Good
                     "V201018",
                     "V201510",
                     "V201200",
                     "V201607",
                     "V201458x",
                     "V201549x",
                     "V203003",
                     "V201508",
        ]]
        """
        V201321
        V201401
        V201309
        V201130 (-V201132 "handling economy")
        V201235
        V201300
        V201318
        V201324
        V201325
        V201594
        V201312
        V201416
        V201133 (-V201135 "handling foreign relations")
        V201139 (-V201141 "handling immigration")
        V201350
        V201619
        V201620
        V201006
        V201223
        V201234
        """



        # Dropping rows with NA values
        new_df = new_df.dropna(axis=0)

        # Renaming columns for convenience
        new_df = new_df.rename({"Gender": "gender",
                                "Age": "age",
                                "Which character shot first?": "shot_first",
                                ("Do you consider yourself to be a fan "
                                 "of the Star Wars film franchise?"): "fan"},
                               axis=1)

        # Removing "I don't understand this question" response
        new_df = new_df.loc[new_df["shot_first"].isin(["Han", "Greedo"])]

        # Get only top 8 rows to keep things simple for testing
        new_df = new_df.head(8)

        return new_df

    def _make_backstory(self, row):
        return f"I am a {row['age']} year old {row['gender'].lower()}."

    def _get_prompt_instructions(self):
        return {"shot_first": (("Between Han and Greedo I think the one "
                                "who shot first was"),
                               lambda x: x),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}

if __name__ == '__main__':
    import random
    ds = AnesDataset(n_exemplars=5)