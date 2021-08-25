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
# class PewAmericanTrendsDataset(Dataset):

#     def __init__(self, n_exemplars):
#         survey_fname = "data/ATP W78.sav"
#         super().__init__(survey_fname, n_exemplars)

#     def _format(self, df):

#         demographic_col_names = ["F_AGECAT",
#                                  "F_GENDER",
#                                  "F_PARTY_FINAL",
#                                  "F_EDUCCAT",
#                                  "F_IDEO",
#                                  "F_INCOME",
#                                  "F_RELIG",
#                                  RACE,
#                                  REGION,
#                                  "F_MARITAL"]

#         dv_col_names = ["ECON1_W78",
#                         "ECON1B_W78",
#                         "SATIS_W78",
#                         "VTADMIN_POST_US_W78",
#                         "ELECTRESULTPLAT_W78",
#                         "COVID_2ASSISTLD_W78",
#                         "POL12_W78",
#                         "COVID_OPENMORE_W78",
#                         "DIVISIONSCONC_W78",
#                         "VOTELIST_US_W78",
#                         ]

#         new_df = df[demographic_col_names]

#         return df

#     def _make_backstory(self, row):
#         age_str = row["F_AGECAT"]  # ['18-29', '30-49', '50-64', '65+', 'Refused']
#         backstory = f"I am between {} and {} years old."

#         return "Backstory"

#     def _get_prompt_instructions(self):
#         return {}    


class PewAmericanTrendsWave78Dataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "../data/ATP W78.sav"  # TODO: Remove the dots
        self._n_exemplars = n_exemplars
        super().__init__(survey_fname, n_exemplars)

        # Issues
        #   Removing "refused" is cutting a certain group of people

    def _format(self, df):

        # Selecting relevant columns
        demographic_col_names = ["F_AGECAT",
                                 "F_GENDER",
                                 "F_PARTY_FINAL",
                                 "F_EDUCCAT",
                                 "F_IDEO",
                                 "F_INC_SDT1",
                                 "F_RELIG",
                                 "F_RACETHNMOD",
                                 "F_CREGION",
                                 "F_CDIVISION",
                                 "F_MARITAL"]

        dv_col_names = ["ECON1_W78",
                        "ECON1B_W78",
                        "SATIS_W78",
                        "VTADMIN_POST_US_W78",
                        "ELECTRESULTPLAT_W78",
                        "COVID_2ASSISTLD_W78",
                        "POL12_W78",
                        "COVID_OPENMORE_W78",
                        "DIVISIONSCONC_W78",
                        "VOTELIST_US_W78"]

        new_df = df[demographic_col_names + dv_col_names]

        # Renaming columnns for convenience
        new_df = new_df.rename({"F_AGECAT": "age",
                                "F_GENDER": "gender",
                                "F_PARTY_FINAL": "party",
                                "F_EDUCCAT": "educ",
                                "F_IDEO": "ideo",
                                "F_INC_SDT1": "income",
                                "F_RELIG": "religion",
                                "F_RACETHNMOD": "race",
                                "F_CREGION": "census_reg",
                                "F_CDIVISION": "census_div",
                                "F_MARITAL": "marital",
                                "ECON1_W78": "econ_today",
                                "ECON1B_W78": "econ_year_away",
                                "SATIS_W78": "country_satisfied",
                                "VTADMIN_POST_US_W78": "election_wellness",
                                "ELECTRESULTPLAT_W78": "election_news",
                                "COVID_2ASSISTLD_W78": "covid_assist_pack",
                                "POL12_W78": "rep_dem_relationship",
                                "COVID_OPENMORE_W78": "covid_restrict",
                                "DIVISIONSCONC_W78": "rep_dem_division",
                                "VOTELIST_US_W78": "more_votes_better"},
                               axis=1)

        # Drop rows with unhelpful answers
        new_df = new_df[new_df["party"].isin(["Democrat",
                                              "Republican",
                                              "Independent"])]

        for col_name in list(new_df):
            new_df = new_df[new_df[col_name] != "Refused"]

        # Randomly sample 500 + self._n_exemplars rows
        new_df = new_df.sample(n=500+self._n_exemplars, random_state=0)
        print(len(new_df))

        return new_df

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


if __name__ == "__main__":
    PewAmericanTrendsWave78Dataset(n_exemplars=5)


####################################################################################
# Maren
####################################################################################

####################################################################################
# Chris
####################################################################################