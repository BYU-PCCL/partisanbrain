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
                        "Q27P",
                        "Q18C",
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
                        "Q27O",
                        "Q27R",
                        "Q33",
                        "Q27N",
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
                                "Q27P": "immigrant_preference",
                                "Q18C": "putin_opinion",
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
                                "Q27O": "white_vs_black_discrimination",
                                "Q27R": "stranger_in_own_country",
                                "Q33" : "demographic_change_opinion",
                                "Q27N": "use_of_racism",
                                "Q20D": "elect_non_christian"},
                                axis=1)

        # Dropping rows with problematic values
        problematic_values = ["Refused", "Something else", "Don't know (VOL.)", "Skipped on web"]
        new_df = new_df[new_df["religion"] != "Something else"]
        new_df = new_df[new_df["party"].isin(["A Democrat",
                                              "A Republican",
                                              "An Independent"])]

        new_df = new_df[new_df["race_ethnicity"] != "Other, non-Hispanic"]

        new_df = new_df[new_df["immigrant_citizenship"] != "None of these"]

        for col_name in list(new_df):
            new_df = new_df[~new_df[col_name].isin(problematic_values)]

        new_df = new_df.dropna()

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
        backstory.append("It is between September and October 2018.")

        return " ".join(backstory)

    def _get_prompt_instructions(self):
        return {"electing_women": (("I think electing more women to office would make things"),
                               lambda x: {"Better": "better",
                                          "Worse": "worse",
                                          "Not much different": "same"}),
                "electing_LGBTQIA": ("I think electing more lesbian, gay, bisexual, and "
                                     "transgender people would make things",
                                lambda x: {"Better": "better",
                                          "Worse": "worse",
                                          "Not much different": "same"}),
                "do_more_for_LGBTQIA": (("If asked (yes or no) whether the US has done enough "
                                        "to give gay and lesbian people equal rights, I'd say"),
                                lambda x: {"Our country has made the changes needed to give "
                                           "gay and lesbian people equal rights with other "
                                           "Americans": "yes",
                                           "Our country needs to continue making changes to "
                                           "give gay and lesbian people equal rights with other "
                                           "Americans": "no"}),
                "immigrant_preference": (("If asked (yes or no) whether I think we should give preference "
                                         "to immigrants from Western Europe, who share our values I'd say"),
                                lambda x: {"Completely agree": "yes",
                                           "Mostly agree" : "yes",
                                           "Mostly disagree": "no",
                                           "Completely disagree": "no"}),
                "putin_opinion": (("If asked (yes or no) about my opinion of Russian President "
                                   "Vladimir Putin I'd say"),
                                lambda x: {"Completely agree": "yes",
                                           "Mostly agree" : "yes",
                                           "Mostly disagree": "no",
                                           "Completely disagree": "no"}),
                "view_on_immigration": (("I think the growing number of newcomers from other "
                                         " countries is "),
                                lambda x: {"Threatens traditional American "
                                           "customs and values": "bad",
                                           "Strengthens American society" : "good"}),
                "perspective_on_immigration": (("My overall perspective on immigration is that it is"),
                                lambda x: {"Immigrants today strengthen our country "
                                           "because of their hard work and talents": "good",
                                           "Immigrants today are a burden on our "
                                           "country because they take our jobs, housing, "
                                           "nd healthcare": "bad"}),
                "laws_preventing_refugees": (("If asked (yes or no) whether the US should "
                                              "pass laws preventing refugees from entering "
                                              "I'd say"),
                                lambda x: {"Strongly favor": "yes",
                                           "Favor" : "yes",
                                           "Oppose": "no",
                                           "Strongly oppose": "no"}),
                "immigrant_citizenship": (("I think that the US immigration system should deal "
                                           "with immigrants who are currently living in the US "
                                           "illegally by"),
                                lambda x: {"Allow them a way to become citizens provided "
                                           "they meet certain requirements": "Allow them "
                                           "to become citizens",
                                           "Allow them to become permanent legal residents, "
                                           "but not citizens" : "Allow them to become residents, "
                                           "not citizens",
                                           "Identify and deport them": "deport them"}),
                "voting_frequency": (("If asked how frequently I vote I'd say"),
                                lambda x: {"Always": "always",
                                           "Nearly always" : "sometimes",
                                           "In about half of elections": "sometimes",
                                           "Seldom": "sometimes",
                                           "Never": "never"}),
                "trump_job_opinion": (("If asked (yes or no) whether I approve of the job Donald "
                                       "Trump is doing as president I would say"),
                                lambda x: {"Strongly approve": "yes",
                                           "Somewhat approve" : "yes",
                                           "Somewhat disapprove": "no",
                                           "Strongly disapprove": "no"}),
                "electing_minorities": (("I think electing people from racial and ethnic minority "
                                         "groups would make things"),
                                lambda x: {"Better": "better",
                                          "Worse": "worse",
                                          "Not much different": "same"}),
                "police_brutality_pattern": (("If asked whether the recent killings of African American "
                                              "men by police are isolated incidents or a "
                                              "broader pattern I'd say"),
                                lambda x: {"Isolated incidents": "isolated incidents",
                                          "Part of a broader pattern": "broader pattern"}),
                "asian_discrimination": (("If asked (yes or no) whether I think there is "
                                          "a lot of discrimination against asians in the US "
                                          "today I'd say"),
                                lambda x: {"Yes, there is a lot of discrimination": "yes",
                                          "No, not a lot of discrimination": "no"}),
                "hispanic_discrimination": (("If asked (yes or no) whether I think there is "
                                          "a lot of discrimination against hispanics in the US "
                                          "today I'd say"),
                                lambda x: {"Yes, there is a lot of discrimination": "yes",
                                          "No, not a lot of discrimination": "no"}),
                "white_vs_black_discrimination": (("If asked (yes or no) whether I think that "
                                          "discrimination against whites has become as big a "
                                          "problem as discrimination against blacks in the US "
                                          "I'd say"),
                                lambda x: {"Completely agree": "yes",
                                          "Mostly agree": "yes",
                                          "Mostly disagree": "no",
                                          "Completely disagree": "no"}),
                "stranger_in_own_country": (("If asked (yes or no) whether I think the US has "
                                          "changed so much that I feel like a stranger in my own "
                                          "country I'd say"),
                                lambda x: {"Completely agree": "yes",
                                          "Mostly agree": "yes",
                                          "Mostly disagree": "no",
                                          "Completely disagree": "no"}),
                "demographic_change_opinion": (("By 2045, minorities will together be a majority "
                                                " in the US. If asked whether I think the impact "
                                                "of the coming demographic change will be positive "
                                                "or negative I'd say"),
                                lambda x: {"Mostly positive": "positive",
                                          "Mostly negative": "negative"}),
                "use_of_racism": (("If asked (yes or no) whether I think that minorities "
                                          "in the US use racism as an excuse more than they "
                                          "should I'd say"),
                                lambda x: {"Completely agree": "yes",
                                          "Mostly agree": "yes",
                                          "Mostly disagree": "no",
                                          "Completely disagree": "no"}),
                
                "elect_non_christian": (("I think electing more non christian people to "
                                         "office would make things"),
                               lambda x: {"Better": "better",
                                          "Worse": "worse",
                                          "Not much different": "same"})}
    
