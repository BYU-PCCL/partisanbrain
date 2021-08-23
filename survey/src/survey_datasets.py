from dataset import Dataset


class ExampleSurveyDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "../data/StarWars.csv"
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):
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
        return new_df

    def _get_shot_first_str(self, value):
        if len(value) != 0:
            value = " " + value
        return ("Between Han and Greedo I think the one "
                f"who shot first was{value}")

    def _get_fan_str(self, value):
        if value == "":
            return "When asked if I'm a Star Wars fan I say"
        else:
            return ("When asked if I'm a Star Wars fan "
                    f"I say {value.lower()}")

    def _make_backstory(self, row):
        return f"I am a {row['age']} year old {row['gender'].lower()}."

    def _make_prompts(self, row, exemplars):
        prompts = []
        row_backstory = self._make_backstory(row)
        exemplar_backstories = [self._make_backstory(e) for (_, e) in exemplars.iterrows()]

        # Our first prompt is for the shot_first DV
        exemplar_shot_firsts = [self._get_shot_first_str(e["shot_first"]) for (_, e)
                                in exemplars.iterrows()]
        shot_first_blank = self._get_shot_first_str("")
        prompt = ""
        for i in range(len(exemplars)):
            prompt += exemplar_backstories[i] + " "
            prompt += exemplar_shot_firsts[i] + "\n"
        prompt += row_backstory + " "
        prompt += shot_first_blank

        prompts.append(prompt)

        # Our second prompt is for the fan DV
        exemplar_fans = [self._get_fan_str(e["fan"]) for (_, e)
                         in exemplars.iterrows()]
        fan_blank = self._get_fan_str("")
        prompt = ""
        for i in range(len(exemplars)):
            prompt += exemplar_backstories[i] + " "
            prompt += exemplar_fans[i] + "\n"
        prompt += row_backstory + " "
        prompt += fan_blank

        prompts.append(prompt)

        return prompts


if __name__ == '__main__':
    ds = ExampleSurveyDataset(n_exemplars=5)
    print(ds._make_prompts(ds.data.iloc[0], ds.exemplars))
