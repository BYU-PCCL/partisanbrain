from dataset import Dataset

class PRRIDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "PRRI 2018 American Values Survey.sav"
        self.n_exemplars = n_exemplars
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):

        # Dropping all but relevant columns
        demographic_col_names = ["AGE",
                        "AGE4",
                        "AGE7",
                        "GENDER",
                        "PARTY",
                        "EDUC",
                        "IDEO",
                        "INCOME",
                        "RELIG",
                        "RACETHNICITY",
                        "REGION9",
                        "MARITAL"]

        dv_col_names = ["Q20A",
                        "Q20B",
                        "Q29",
                        "Q27D",
                        "Q19",
                        "Q30",
                        "Q31",
                        "Q35C",
                        "Q36",
                        "Q1",
                        "Q5",
                        "Q20C",
                        "Q22",
                        "Q26E",
                        "Q26F",
                        "Q27B",
                        "Q27I",
                        "Q27L",
                        "Q27G",
                        "Q20D"]

        new_df = df[demographic_col_names + dv_col_names]

        # Renaming columns for convenience
        new_df = new_df.rename({"AGE": "age",
                                "AGE4": "age4",
                                "AGE7": "age7",
                                "GENDER": "gender",
                                "PARTY": "party",
                                "EDUC": "education",
                                "IDEO": "ideology",
                                "INCOME": "income",
                                "RELIG": "religion",
                                "RACETHNICITY": "race_ethnicity",
                                "REGION9": "region",
                                "MARITAL": "marital_status",
                                "Q20A": "electing_women",
                                "Q20B": "electing_LGBTQIA",
                                "Q29": "do_more_for_LGBTQIA",
                                "Q27D": "slavery_effects",
                                "Q19": "putin_opinion",
                                "Q30": "view_on_immigration",
                                "Q31": "perspective_on_immigration",
                                "Q35C": "laws_preventing_refugees",
                                "Q36": "immigrant_citizenship",
                                "Q1": "voting_frequency",
                                "Q5": "trump_job_opinion",
                                "Q20C": "electing_minorities",
                                "Q22": "police_brutality_pattern",
                                "Q26E": "asian_discrimination",
                                "Q26F": "hispanic_discrimination",
                                "Q27B": "effect_of_effort_for_blacks",
                                "Q27I": "racial_issues_isolated",
                                "Q27L": "gender_discrimination_toward_men",
                                "Q27G": "fear_other_races",
                                "Q20D": "elect_non_religious"},
                                axis=1)

        # Dropping rows with problematic values
        new_df = new_df[new_df["religion"] != "Something else"]
        new_df = new_df[new_df["religion"] != "Skipped on web"]
        new_df = new_df[new_df["religion"] != "Don't know (VOL.)"]
        new_df = new_df[new_df["religion"] != "Refused"]

        new_df = new_df[new_df["party"].isin(["A Democrat",
                                              "A Republican",
                                              "An Independent"])]

        new_df = new_df[new_df["ideology"] != "Refused"]
        new_df = new_df[new_df["ideology"] != "Don't know (VOL.)"]
        new_df = new_df[new_df["ideology"] != "Skipped on web"]

        new_df = new_df[new_df["race_ethnicity"] != "Other, non-Hispanic"]

        return new_df

    def _make_backstory(self, row):
        backstory = []

        # Age
        age = str(int(row["age"]))
        backstory.append(f"I am {age} years old.")

        # Gender
        backstory.append(f"I am {row['gender'].lower()}.")

        # Party
        if row["party"] == "An Independent":
            backstory.append("In terms of political parties I am independent.")
        else:
            backstory.append("In terms of political parties "
                             f"I am a {row['party']}.")

        # Education
        education = row["education"]
        if education == "Bachelor's degree":
            backstory.append("I completed an undergraduate degree.")
        elif education == "Master's degree" or education == "Professional or doctorate degree":
            backstory.append("I went to grad school.")
        elif education == "Some college, no degree" or education == "Associate degree":
            backstory.append("I've completed some college.")
        else:
            backstory.append("I didn't go to college.")

        # Ideology
        backstory.append(("In terms of political ideology, "
                          "I'd consider myself "
                          f"to be {row['ideology'].lower()}."))

        # Income
        if "Less than" in row["income"]:
            backstory.append("My annual family income is less than $5,000")
        elif "or more" in row["income"]:
            backstory.append("My annual family income is $200,000 or more")
        else:
            low, high = row["income"].split("-")
            backstory.append(("My annual family income is "
                              f"between {low} and {high}."))

        # Religiosity
        if row["religion"] == "Nothing in particular":
            backstory.append(("I don't identify with any religion "
                              "in particular."))
        else:
            if "Christian" in row["religion"]:
                religion = "Christian"
            elif "Agnostic" in row["religion"]:
                religion = "Agnostic"
            elif "Jewish" in row["religion"]:
                religion = "Jewish"
            elif "Catholic" in row["religion"]:
                religion = "Catholic"
            elif "Protestant" in row["religion"]:
                religion = "Protestant"
            elif "Atheist" in row["religion"]:
                religion = "Atheist"
            elif "Muslim" in row["religion"]:
                religion = "Muslim"
            elif "Orthodox" in row["religion"]:
                religion = "Orthodox"
            elif "Unitarian" in row["religion"]:
                religion = "Unitarian"
            elif "Mormon" in row["religion"]:
                religion = "Mormon"
            else:
                religion = row["religion"]
            backstory.append(f"In terms of religion I am {religion}.")

        # Race/Ethnicity
        race = row['race_ethnicity'].replace(' non-Hispanic', '').lower()
        backstory.append(f"I'm {race}.")

        # Region
        if row["region"] == "New England" or row["region"] == "Mid Atlantic":
            backstory.append("I live in the northeast of the United States.")
        elif row["region"] == "Mountain":
            backstory.append("I live in the western United States.")
        elif row["region"] == "East North Central" or row["region"] == "West North Central":
            backstory.append("I live in the midwest of the United States.")
        elif row["region"] == "West South Central":
            backstory.append("I live in the southern United States.")
        elif row["region"] == "East South Central" or row["region"] == "South Atlantic":
            backstory.append("I live in the south eastern United States.")
        else:
            backstory.append("I live in the western United States.")

        # Marital Status
        if row["marital_status"] == "Never married":
            backstory.append("I've never been married.")
        elif row["marital_status"] == "Separated":
            backstory.append(("I got married, but I'm now "
                              "separated from my partner."))
        elif row["marital_status"] == "Living with partner":
            backstory.append("I am not married, but I am living "
                            "with my partner.")
        else:
            backstory.append(f"I'm {row['marital_status'].lower()}.")

        # Date
        backstory.append("It's November 2020.")

        return " ".join(backstory)

    def _get_prompt_instructions(self):
        return {"shot_first": (("Between Han and Greedo I think the one "
                                "who shot first was"),
                               lambda x: x),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}

PRRIDataset(n_exemplars=5)

