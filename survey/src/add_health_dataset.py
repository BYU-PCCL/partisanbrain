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
        # note: these codes are different than those in the spreadsheet. Those are wave 5, these are wave 4
        demographic_col_names = ["H4OD1Y", #birth year
                                 "BIO_SEX4",
                                 "H4ED2", #education
                                 "H4DA28", #political ideologu
                                 "H4EC1", #household income
                                 "H4RE1", #religion

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
        new_df = new_df.rename({"H40D1Y": "age",
                                "BIO_SEX4": "gender",
                                "H4ED2": "education",
                                "H4DA28": "ideology",
                                "H4EC1": "income",
                                "H4RE1": "religion",
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
            new_df = new_df[new_df[col_name] != "Don't know"]

        # Randomly sample 500 + self._n_exemplars rows
        new_df = new_df.sample(n=500+self._n_exemplars, random_state=0)

        return new_df

    def _make_backstory(self, row):
        backstory = []

        # Age
        backstory.append(f"I was born in {row['age']}")

        # Gender

        backstory.append(f"I am {row['gender'].lower()}.")

        # Party
        #missing

        # Education
        if row["education"] == "Some graduate school" OR "Completed a master's degree" OR "Some graduate training beyond a master's degree" OR "Completed a doctoral degree" OR "Some post baccalaureate professional education (e.g., law school, med school, nurse)" OR "Completed post baccalaureate professional education (e.g., law school, med school, nurse)":
            backstory.append("I went to grad school.")
        elif row["education"] == "Completed college (bachelor's degree)":
            backstory.append("I completed college.")
        elif row["education"] == "Some College":
            backstory.append("I've completed some college.")
        else:
            backstory.append("I didn't go to college.")

        # Ideology
        backstory.append(("In terms of political ideology, "
                          "I'd consider myself "
                          f"to be {row['ideology'].lower()}."))

        # Income
        if "to " in row["income"]:
            low, high = row["income"].split(" to ")
            backstory.append(("My annual family income is "
                              f"between {low} and {high}."))
        elif row["income"] == "Less than $5,000":
            backstory.append("My annual family income is less than $5000.")
        else:
            backstory.append("My annual family income is more than $150,000.")

        # Religiosity
        if row["religion"] == "None/atheist/agnostic":
            backstory.append(("I don't identify with any religion "
                              "in particular."))
        else:
            if "Protestant (such as Assemblies of God, Baptist, Lutheran, Methodist, Presbyterian, etc.)" in row["religion"]:
                religion = "Protestant"
            elif "Other Christian" in row["religion"]:
                religion = "Christian"
            else:
                religion = row["religion"]
            backstory.append(f"In terms of religion I am {religion}.")

        # Race/Ethnicity
        # Missing

        # Region
        # Missing

        # Marital Status
        #Missing

        # Date
        backstory.append("It's November 2020.")

        return " ".join(backstory)

    def _get_prompt_instructions(self):
        return {"shot_or_stabbed": (("Which of the following things happened in "
                                     "the past 12 months: You shot or stabbed "
                                     "someone?"), lambda x: x.lower()),
                "arrested": (("Have you ever been arrested?"),
                                   lambda x: x.lower()),
                "physical_fight": (("In the past 12 months, how often did you "
                                    "get into a serious physical fight?"),
                                      lambda x: x.lower()),
                "convicted_of_charges": (("Have you ever been convicted of or "
                                          "pled guilty to any charges other "
                                          "than a minor traffic violation?"),
                                      lambda x: x.lower()),
                "sell_drugs": (("In the past 12 months, how often did you sell "
                                "marijuana or other drugs?"),
                                    lambda x: x.lower()),
                "sadness_family": (("How often was each of the following things "
                                    "true during the past seven days: You could "
                                    "not shake off the blues, even with help "
                                    "from your family and your friends."),
                                      lambda x: {("Never or rarely "): "never",
                                                 ("Sometimes "): "sometimes",
                                                 ("A lot of the time "): "frequently",
                                                 ("Most of the time or all of the time "): "most of the time"}[x]),
                "counseling": (("In the past 12 months have you received "
                                "psychological or emotional counseling?"),
                                         lambda x: x.lower()),
                "worrying": (("How much do you agree with each statement about "
                              "you as you generally are now, not as you wish "
                              "to be in the future? I worry about things."),
                                   lambda x: {"Strongly agree": "agree",
                                              "Agree": "agree",
                                              "Neither agree nor disagree ": "neither"
                                              "Disagree": "disagree",
                                              "Strongly Disagree": "disagree"}[x]),
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
