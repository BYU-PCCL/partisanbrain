from dataset import Dataset


class AddHealthDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "../data/"  # TODO: Remove the dots
        self._n_exemplars = n_exemplars
        super().__init__(survey_fname, n_exemplars)

        #issues:
        #missing political party demographic
    def _format(self, df):

        # No need to filter rows for USA because all respondents from USA

        # Selecting relevant columns
        demographic_col_names = ["H5OD1Y - S1Q1B BIRTH YEAR-W5",
                                 "H5OD2B - S1Q2B GENDER-W5",
                                 "H5OD11 - S1Q11 HIGHEST EDU ACHIEVED TO DATE-W5",
                                 "H5SS9 - S10Q9 POLITICAL LEANINGS-W5",
                                 "H5EC1 - S4Q1 INCOME PERS EARNINGS 16/17-W5",
                                 "H5RE1 - S12Q1 WHAT IS YOUR PRESENT RELIGION-W5",
                                 "H5OD4A - S1Q4", #race
                                 "W5REGION - Respondent census region-W5",
                                 "H5HR1 - S2Q1 CURRENT MARITAL STATUS-W5"]

        dv_col_names = ["H4DS20",
                        "H4CJ1",
                        "H4DS11",
                        "H4CJ10",
                        "H4DS5",
                        "H4HS9",
                        "H4MH19",
                        "H4PE6",
                        "H4SE2",
                        "H4PE23",
                        "H4MH24",
                        "H4GH8",
                        "H4DA1",
                        "H4DA5",
                        "H4TO1",
                        "H4MA3",
                        "H4TO34",
                        "H4ID8",
                        "H4TO33"
                        "H4RE10"]

        new_df = df[demographic_col_names + dv_col_names]

        # Renaming columnns for convenience
        new_df = new_df.rename({"H5OD1Y - S1Q1B BIRTH YEAR-W5": "age",
                                 "H5OD2B - S1Q2B GENDER-W5": "gender",
                                 "H5OD11 - S1Q11 HIGHEST EDU ACHIEVED TO DATE-W5": "education",
                                 "H5SS9 - S10Q9 POLITICAL LEANINGS-W5": "ideology",
                                 "H5EC1 - S4Q1 INCOME PERS EARNINGS 16/17-W5": "income",
                                 "H5RE1 - S12Q1 WHAT IS YOUR PRESENT RELIGION-W5": "religion",
                                 "H5OD4A - S1Q4": "race",
                                 "W5REGION - Respondent census region-W5": "region",
                                 "H5HR1 - S2Q1 CURRENT MARITAL STATUS-W5": "marital",
                                 "H4DS20": "shot_or_stabbed",
                                 "H4CJ1": "arrested",
                                 "H4DS11": "physical_fight",
                                 "H4CJ10": "convicted_of_charges",
                                 "H4DS5": "sell_drugs",
                                 "H4HS9": "counseling",
                                 "H4MH19": "sadness_family",
                                 "H4PE6": "worrying",
                                 "H4SE2": "suicide",
                                 "H4PE23": "optimism",
                                 "H4MH24": "happiness",
                                 "H4GH8": "fast_food",
                                 "H4DA1": "hours_of_tv",
                                 "H4DA5": "individual_sports",
                                 "H4TO1": "smoked_cigarette",
                                 "H4MA3": "physical_child_abuse",
                                 "H4TO34": "age_of_first_drink",
                                 "H4ID8": "car_accidents",
                                 "H4TO33": "drinking",
                                 "H4RE10": "prayer_in_private"},
                               axis=1)

        # Drop rows with unhelpful answers
        new_df = new_df[new_df["religion"] != "Other" OR "Refused" OR "Don't know"]

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

    def _get_prompt_instructions(self):
        return {"econ_today": (("Between excellent, good, fair, and poor, "
                                "I'd call the the economic conditions in "
                                "the US"), lambda x: {"Excellent": "excellent",
                                                      "Good": "good",
                                                      "Only fair": "fair",
                                                      "Poor": "poor"}[x]),
                "econ_year_away": (("If I had to call the economic conditions "
                                    "in the US I expect a year from now "
                                    "(compared to now) better, worse, or "
                                    "same, I'd call them"),
                                   lambda x: {"Better": "better",
                                              "Worse": "worse",
                                              ("About the same "
                                               "as now"): "same"}[x]),
                "country_satisfied": (("If asked whether I'm satisfied or "
                                       "dissatisfied with the way things are "
                                       "going in this country today I'd "
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
                                     "Election Day I'd say"),
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
                                       "or is not necessary I'd say it is"),
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
                                      "Republicans and Democrats (yes or no) "
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
    from experiment import Experiment
    ds = PewAmericanTrendsWave78Dataset(n_exemplars=5)
    e = Experiment(ds, gpt_3_engine="ada")
    e.run()
    e.save_results("pew_results.pkl")
