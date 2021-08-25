from dataset import Dataset


class PewAmericanTrendsWave78Dataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "../data/ATP W78.sav"  # TODO: Remove the dots
        self._n_exemplars = n_exemplars
        super().__init__(survey_fname, n_exemplars)

        # Issues
        #   Removing "refused" is cutting a certain group of people

    def _format(self, df):

        # No need to filter rows for USA because all respondents from USA

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
        new_df = new_df[new_df["religion"] != "Other"]

        for col_name in list(new_df):
            new_df = new_df[new_df[col_name] != "Refused"]

        # Randomly sample 500 + self._n_exemplars rows
        new_df = new_df.sample(n=500+self._n_exemplars, random_state=0)
        print(new_df["religion"].unique().tolist())

        return new_df

    def _make_backstory(self, row):
        backstory = []

        # Age
        if row["age"] == "65+":
            backstory.append("I am at least 65 years old.")
        else:
            low, high = row["age"].split("-")
            backstory.append(f"I am between {low} and {high} years old.")

        # Gender
        if row["gender"] == "In some other way":
            backstory.append("I don't identify as male or female.")
        else:
            backstory.append(f"I am {row['gender'].lower()}.")

        # Party
        if row["party"] == "Independent":
            backstory.append("In terms of political parties I am independent.")
        else:
            backstory.append(f"In terms of political parties I am {row['party']}")

        # TODO: Education

        # TODO: Ideology
        if row["ideo"] in ["Conservative", "Very conservative"]:
            ideology = "conservative"
        elif row["ideo"] in ["Liberal", "Very liberal"]:
            ideology = "liberal"
        else:
            ideology = "neutral"

        backstory.append(("In terms of political ideology, "
                          f"I'd consider myself to be {ideology}."))


        # TODO: Income

        # TODO: Religiosity
        if row["religion"] == "Nothing in particular":
            backstory.append("I don't identify with any religion in particular.")
        else:
            if "Orthodox" in row["religion"]:
                religion == "Orthodox"
            elif "Mormon" in row["religion"]:
                religion = "Mormon"
            elif row["religion"] in ["Atheist", "Agnostic"]:
                religion = row["religion"].lower()
            else:
                religion = row["religion"]
            print(f"In terms of religion I am {religion}")

        # TODO: Race/Ethnicity

        # TODO: Region

        # TODO: Marital Status

        return backstory

    def _get_prompt_instructions(self):
        return {}


if __name__ == "__main__":
    PewAmericanTrendsWave78Dataset(n_exemplars=5)
