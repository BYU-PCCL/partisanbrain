from dataset import Dataset


class ExampleSurveyDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "../data/StarWars.csv"
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):

        # Reduce rows (e.g., down to US)

        # Dropping all but relevant columns
        new_df = df[["Gender",
                     "Age",
                     "Which character shot first?",
                     ("Do you consider yourself to be a fan "
                      "of the Star Wars film franchise?")]]

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

        # Randomly sample columns!

        # Get only top 8 rows to keep things simple for testing
        new_df = new_df.head(105)

        return new_df

    def _make_backstory(self, row):
        return f"I am a {row['age']} year old {row['gender'].lower()}."

    def _get_prompt_instructions(self):
        return {"shot_first": (("Between Han and Greedo I think the one "
                                "who shot first was"),
                               lambda x: x),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}


####################################################################################
# Josh
####################################################################################
class PewAmericanTrendsWave78Dataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "data/ATP W78.sav"
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):
        return df

    def _make_backstory(self, row):
        return "Backstory"

    def _get_prompt_instructions(self):
        return {}


class PewAmericanTrendsWave67Dataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "data/StarWars.csv"
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):
        return df

    def _make_backstory(self, row):
        return "Backstory"

    def _get_prompt_instructions(self):
        return {}


####################################################################################
# Maren
####################################################################################

####################################################################################
# Chris
####################################################################################