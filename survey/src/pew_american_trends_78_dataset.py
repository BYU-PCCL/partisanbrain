from dataset import Dataset


class PewAmericanTrendsWave78Dataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "../data/ATP W78.sav"  # TODO: Remove the dots
        self._n_exemplars = n_exemplars
        super().__init__(survey_fname, n_exemplars)

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
                                 "F_MARITAL"]

        dv_col_names = ["ECON1_W78",
                        "ECON1B_W78",
                        "SATIS_W78",
                        "VTADMIN_POST_US_W78",
                        "ELECTNTFOL_W78",
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
                                "F_MARITAL": "marital",
                                "ECON1_W78": "econ_today",
                                "ECON1B_W78": "econ_year_away",
                                "SATIS_W78": "country_satisfied",
                                "VTADMIN_POST_US_W78": "election_wellness",
                                "ELECTNTFOL_W78": "follow_election",
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
        new_df = new_df[new_df["race"] != "Other"]

        for col_name in list(new_df):
            new_df = new_df[new_df[col_name] != "Refused"]

        # Randomly sample 500 + self._n_exemplars rows
        new_df = new_df.sample(n=500+self._n_exemplars, random_state=0)

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
            backstory.append("In terms of political parties "
                             f"I am {row['party']}.")

        # Education
        if row["educ"] == "College graduate+":
            backstory.append("I went to grad school.")
        elif row["educ"] == "Some College":
            backstory.append("I've completed some college.")
        else:
            backstory.append("I didn't go to college.")

        # Ideology
        if row["ideo"] in ["Conservative", "Very conservative"]:
            ideology = "conservative"
        elif row["ideo"] in ["Liberal", "Very liberal"]:
            ideology = "liberal"
        else:
            ideology = "neutral"

        backstory.append(("In terms of political ideology, "
                          f"I'd consider myself to be {ideology}."))

        # Income
        if "to less than" in row["income"]:
            low, high = row["income"].split(" to less than ")
            backstory.append(f"My family income is between {low} and {high}.")
        else:
            backstory.append(f"My family income is {row['income']}.")

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
            backstory.append(f"In terms of religion I am {religion}")

        # Race/Ethnicity
        backstory.append(f"I'm {row['race'].replace(' non-Hispanic', '')}")

        # Region
        if row["census_reg"] == "Northeast":
            backstory.append("I'm from the northeast of the United States.")
        elif row["census_reg"] == "West":
            backstory.append("I'm from the western United States")
        else:
            backstory.append(f"I'm from the {row['census_reg']}")

        # Marital Status
        if row["marital"] == "Never been married":
            backstory.append("I've never been married.")
        elif row["marital"] == "Separated":
            backstory.append(("I got married, but I'm now "
                              "separated from my partner."))
        else:
            backstory.append(f"I'm {row['marital'].lower()}")

        # Date
        backstory.append("It's November 2020.")

        return " ".join(backstory)

    def _get_prompt_instructions(self):
        return {"econ_today": (("Between excellent, good, fair, and poor, "
                                "I'd call the the economic conditions in "
                                "the US"), lambda x: {"Excellent": "excellent",
                                                      "Good": "good",
                                                      "Only fair": "fair",
                                                      "Poor": "poor"}[x]),
                "econ_year_away": (("If I had to call the economic conditions "
                                    "in the US I expect a year from now "
                                    "better, worse, or same as now, I'd "
                                    "call them"),
                                   lambda x: {"Better": "better",
                                              "Worse": "worse",
                                              ("About the same "
                                               "as now"): "same"}[x]),
                "country_satisfied": (("If asked whether I'm satisfied or "
                                       "dissatisfied with the way things are "
                                       "going in this country today I would "
                                       "say that I'm"),
                                      lambda x: x.lower()),
                "election_wellness": (("If asked whether I think the "
                                       "elections this month in the United "
                                       "States were run well or poorly, "
                                       "I'd say they were run"),
                                      lambda x: {"Very well": "well",
                                                 "Somewhat well": "well",
                                                 "Not too well": "poorly",
                                                 ("Not at "
                                                  "all well"): "poorly"}[x]),
                "follow_election": (("If asked (yes or no) if I followed "
                                     "the results of the presidential "
                                     "election after polls closed on "
                                     "Election Day I say"),
                                    lambda x: {("Followed them almost "
                                                "constantly"): "yes",
                                               ("Checked in fairly "
                                                "often"): "yes",
                                               ("Checked in "
                                                "occasionally"): "yes",
                                               ("Tuned them out "
                                                "entirely"): "no"}[x]),
                "covid_assist_pack": (("Congress and President Trump passed "
                                       "a $2 trillion economic assistance "
                                       "package in March in response to "
                                       "the economic impact of the "
                                       "coronavirus outbreak. If asked "
                                       "whether I think another economic "
                                       "assistance package is necessary "
                                       "or it is not necessary I'd say it is"),
                                      lambda x: x.lower()),
                "rep_dem_relationship": (("If asked if relations between "
                                          "Republicans and Democrats in "
                                          "Washington a year from now will "
                                          "be better, worse, or same I'd "
                                          "say they will be"),
                                         lambda x: {"Get better": "better",
                                                    "Get worse": "worse",
                                                    ("Stay about "
                                                     "the same"): "same"}[x]),
                "covid_restrict": (("If asked if the number of "
                                    "restrictions on public activity "
                                    "because of the coronavirus "
                                    "outbreak in my area should be "
                                    "increased, decreased, or maintained, "
                                    "I'd say it should be"),
                                   lambda x: {("MORE restrictions "
                                              "right now"): "increased",
                                              ("FEWER restrictions "
                                               "right now"): "decreased",
                                              ("About the same number "
                                               "of restrictions "
                                               "right now"): "maintained"}[x]),
                "rep_dem_division": (("If asked if I'm at least somewhat "
                                      "concerned about divisions between "
                                      "Republicans and Democrats (yes or no)"
                                      "I'd say"),
                                     lambda x: {"Very concerned": "yes",
                                                "Somewhat concerned": "yes",
                                                "Not too concerned": "no",
                                                ("Not at all "
                                                 "concerned"): "no"}[x]),
                "more_votes_better": (("If asked whether the United States "
                                       "would be better off if more Americans "
                                       "voted (yes or no) I'd say"),
                                      lambda x: {("The country would not be "
                                                  "better off if more "
                                                  "Americans voted"): "yes",
                                                 ("The country would be "
                                                  "better off if more "
                                                  "Americans "
                                                  "voted"): "no"}[x])}


if __name__ == "__main__":
    p = PewAmericanTrendsWave78Dataset(n_exemplars=5)
