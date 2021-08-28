from dataset import Dataset
from dataset import PromptSpecs


class PewAmericanTrendsWave78Dataset(Dataset):

    def __init__(self):
        survey_fname = "../data/ATP W78.sav"  # TODO: Remove the dots
        super().__init__(survey_fname)

    def _get_dv_filter_funcs(self):
        return {"econ_today": lambda x: x[x != "Refused"]}

    def _filter_demographics(self, df):
        new_df = df[df["party"].isin(["Democrat",
                                      "Republican",
                                      "Independent"])]
        new_df = new_df[new_df["religion"] != "Other"]
        new_df = new_df[new_df["race"] != "Other"]

        for col_name in list(new_df):
            new_df = new_df[new_df[col_name] != "Refused"]
        return new_df

    def _filter_to_usa(self, df):
        """Return a new dictionary where all respondents are from USA"""
        return df

    def _get_dv_col_names(self):
        return {"ECON1_W78": "econ_today"}
        # return {"ECON1_W78": "econ_today",
        #         "ECON1B_W78": "econ_year_away",
        #         "SATIS_W78": "country_satisfied",
        #         "VTADMIN_POST_US_W78": "election_wellness",
        #         "ELECTNTFOL_W78": "follow_election",
        #         "COVID_2ASSISTLD_W78": "covid_assist_pack",
        #         "POL12_W78": "rep_dem_relationship",
        #         "COVID_OPENMORE_W78": "covid_restrict",
        #         "DIVISIONSCONC_W78": "rep_dem_division",
        #         "VOTELIST_US_W78": "more_votes_better"}

    def _get_demographic_col_names(self):
        return {"F_AGECAT": "age",
                "F_GENDER": "gender",
                "F_PARTY_FINAL": "party",
                "F_EDUCCAT": "educ",
                "F_IDEO": "ideo",
                "F_INC_SDT1": "income",
                "F_RELIG": "religion",
                "F_RACETHNMOD": "race",
                "F_CREGION": "census_reg",
                "F_MARITAL": "marital"}

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
            backstory.append("In terms of political parties "
                             f"I am a {row['party']}.")

        # Education
        if row["educ"] == "College graduate+":
            backstory.append("I went to grad school.")
        elif row["educ"] == "Some College":
            backstory.append("I've completed some college.")
        else:
            backstory.append("I didn't go to college.")

        # Ideology
        backstory.append(("In terms of political ideology, "
                          "I'd consider myself "
                          f"to be {row['ideo'].lower()}."))

        # Income
        if "to less than" in row["income"]:
            low, high = row["income"].split(" to less than ")
            backstory.append(("My annual family income is "
                              f"between {low} and {high}."))
        else:
            income = row['income'].lower()
            backstory.append(f"My annual family income is {income}.")

        # Religiosity
        if row["religion"] == "Nothing in particular":
            backstory.append(("I don't identify with any religion "
                              "in particular."))
        else:
            if "Orthodox" in row["religion"]:
                religion = "Orthodox"
            elif "Mormon" in row["religion"]:
                religion = "Mormon"
            elif row["religion"] in ["Atheist", "Agnostic"]:
                religion = row["religion"].lower()
            else:
                religion = row["religion"]
            backstory.append(f"In terms of religion I am {religion}.")

        # Race/Ethnicity
        race = row['race'].replace(' non-Hispanic', '').lower()
        backstory.append(f"I'm {race}.")

        # Region
        if row["census_reg"] == "Northeast":
            backstory.append("I live in the northeast of the United States.")
        elif row["census_reg"] == "West":
            backstory.append("I live in the western United States.")
        else:
            backstory.append(f"I live in the {row['census_reg']}.")

        # Marital Status
        if row["marital"] == "Never been married":
            backstory.append("I've never been married.")
        elif row["marital"] == "Separated":
            backstory.append(("I got married, but I'm now "
                              "separated from my partner."))
        else:
            backstory.append(f"I'm {row['marital'].lower()}.")

        # Date
        backstory.append("It's November 2020.")

        return " ".join(backstory)

    def _get_col_prompt_specs(self):
        return {"econ_today": PromptSpecs(question=("How would you rate "
                                                    "economic conditions "
                                                    "in this country today?"),
                                          answer_prefix="conditions are",
                                          answer_map={"Excellent": "excellent",
                                                      "Good": "good",
                                                      "Only fair": "fair",
                                                      "Poor": "poor"})}


if __name__ == "__main__":
    ds = PewAmericanTrendsWave78Dataset()
    # Uncomment this to see a sample of your prompts
    # First prompt for each DV
    # for dv_name in ds.dvs.keys():
    #     dv_prompts = ds.prompts[dv_name]
    #     print(dv_prompts[list(dv_prompts.keys())[0]])
    #     print()
