from dataset import Dataset
from dataset import PromptSpecs

class PRRIDataset(Dataset):

    def __init__(self):
        survey_fname = "data/PRRI 2018 American Values Survey.sav"
        super().__init__(survey_fname)

    def _filter_to_usa(self, df):
        # all respondents in this survey should be from the USA
        return df

    def _get_demographic_col_names(self):
        return {"AGE": "age",
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
                "MARITAL": "marital_status"}

    def _get_dv_col_names(self):
        return {"Q20A": "electing_women",
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
                "Q20D": "elect_non_christian"}

    def _filter_demographics(self, df):
        problematic_values = ["Refused", "Something else", "Don't know (VOL.)", "Skipped on web"]
        df = df[df["religion"] != "Something else"]
        df = df[df["party"].isin(["A Democrat",
                                              "A Republican",
                                              "An Independent"])]
        return df

    def _make_backstory(self, row):
        backstory = []

        # Age
        age = str(int(row["age"]))
        backstory.append(f"I am {age} years old.")

        # Gender
        backstory.append(f"I am {row['gender'].lower()}.")

        # Party
        if row["party"] == "An Independent":
            backstory.append("In terms of political parties I am an independent.")
        else:
            backstory.append("In terms of political parties "
                             f"I am {row['party']}.")

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
        race = row['race_ethnicity'].replace(', non-Hispanic', '').lower()
        backstory.append(f"I'm {race}.")

        # Region
        if row["region"] == "New England" or row["region"] == "Mid Atlantic":
            backstory.append("I live in the Northeast of the United States.")
        elif row["region"] == "Mountain":
            backstory.append("I live in the Western United States.")
        elif row["region"] == "East North Central" or row["region"] == "West North Central":
            backstory.append("I live in the Midwest of the United States.")
        elif row["region"] == "West South Central":
            backstory.append("I live in the Southern United States.")
        elif row["region"] == "East South Central" or row["region"] == "South Atlantic":
            backstory.append("I live in the Southeast of the United States.")
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

    def _get_col_prompt_specs(self):
        return {"electing_women": PromptSpecs(("How do you think electing more "
                                               "women to political office would "
                                               "make things in the US?"),
                                              "I think things would be",
                                              {"Better": "better",
                                               "Worse": "worse",
                                               "Not much different": "the same"}),
                "electing_LGBTQIA": PromptSpecs(("How do you think electing more "
                                                "lesbian, gay, bisexual, and "
                                                "transgender people to political "
                                                "office would make things in the US?"),
                                                "I think things would be",
                                                {"Better": "better",
                                                "Worse": "worse",
                                                "Not much different": "the same"}),
                "do_more_for_LGBTQIA": PromptSpecs(("Do you think our country has made "
                                                    "the changes needed to give gay and "
                                                    "lesbian people equal rights in America?"),
                                                   "",
                                                   {"Our country has made the changes "
                                                   "needed to give gay and lesbian "
                                                   "people equal rights with other "
                                                   "Americans": "Yes",
                                                   "Our country needs to continue making "
                                                   "changes to give gay and lesbian people "
                                                   "equal rights with other Americans": "No"}),
                "immigrant_preference": PromptSpecs(("Do you think we should give preference "
                                                     "to immigrants from Western Europe. who "
                                                     "share our values?"),
                                                     "",
                                                     {"Completely agree": "Yes",
                                                     "Mostly agree" : "Yes",
                                                     "Mostly disagree": "No",
                                                     "Completely disagree": "No"}),
                "putin_opinion": PromptSpecs(("How would you describe your overall opinion of"
                                              "Russian President Vladimir Putin?"),
                                              "My opinion is",
                                              {"Very favorable": "favorable",
                                              "Mostly favorable" : "favorable",
                                              "Mostly unfavorable": "unfavorable",
                                              "Very unfavorable": "unfavorable"}),
                "view_on_immigration": PromptSpecs(("Do you think that, in general, the growing "
                                                    "number of newcomers from other countries to "
                                                    "the US is good or bad?"),
                                                    "The growing number of newcomers is",
                                                    {"Threatens traditional American "
                                                    "customs and values": "bad",
                                                    "Strengthens American society" : "good"}),
                "perspective_on_immigration": PromptSpecs(("Do you think that immigrants today "
                                                           "are good or bad for the US?"),
                                                           "I think that immigrants today are",
                                                           {"Immigrants today strengthen our country "
                                                           "because of their hard work and talents": "good",
                                                           "Immigrants today are a burden on our "
                                                           "country because they take our jobs, housing, "
                                                           "and healthcare": "bad"}),
                "laws_preventing_refugees": PromptSpecs(("Do you favor or oppose passing a "
                                                         "law to prevent refugees from "
                                                         "entering the US?"),
                                                         "I",
                                                         {"Strongly favor": "favor",
                                                         "Favor" : "favor",
                                                         "Oppose": "oppose",
                                                         "Strongly oppose": "oppose"}),
                "immigrant_citizenship": PromptSpecs(("How should the US immigration system should deal "
                                                      "with imigrants who are currently living in the "
                                                      "US illegally?"),
                                                      "The US immigration system should",
                                                      {"Allow them a way to become citizens provided "
                                                      "they meet certain requirements": "Allow them "
                                                      "to become citizens",
                                                      "Allow them to become permanent legal residents, "
                                                      "but not citizens" : "Allow them to become residents, "
                                                      "not citizens",
                                                      "Identify and deport them": "deport them"}),
                "voting_frequency": PromptSpecs(("How often would you say you vote?"),
                                                "I vote",
                                                {"Always": "always",
                                                "Nearly always" : "sometimes",
                                                "In about half of elections": "sometimes",
                                                "Seldom": "sometimes",
                                                "Never": "never"}),
                "trump_job_opinion": PromptSpecs(("Do you approve of the job Donald Trump is "
                                                  "doing as president?"),
                                                  "",
                                                  {"Strongly approve": "Yes",
                                                  "Somewhat approve" : "Yes",
                                                  "Somewhat disapprove": "No",
                                                  "Strongly disapprove": "No"}),
                "electing_minorities": PromptSpecs(("How do you think electing more people from "
                                                    "racial and ethnic minority groups to political "
                                                    "office would make things in the US?"),
                                                    "I think things would be",
                                                    {"Better": "better",
                                                    "Worse": "worse",
                                                    "Not much different": "the same"}),
                "police_brutality_pattern": PromptSpecs(("Do you think the recent killings "
                                                         "of African American men by police "
                                                         "are isolated events or part of a "
                                                         "broader pattern how how police "
                                                         "treat African Americans?"),
                                                         "I think they are",
                                                         {"Isolated incidents": "isolated incidents",
                                                         "Part of a broader pattern": "a broader pattern"}),
                "asian_discrimination": PromptSpecs(("In the US today is there a lot of discrimination "
                                                     "against Asians?"),
                                                     "",
                                                     {"Yes, there is a lot of discrimination": "Yes",
                                                     "No, not a lot of discrimination": "No"}),
                "hispanic_discrimination": PromptSpecs(("In the US today is there a lot of discrimination "
                                                     "against Hispanics?"),
                                                     "",
                                                     {"Yes, there is a lot of discrimination": "Yes",
                                                     "No, not a lot of discrimination": "No"}),
                "white_vs_black_discrimination": PromptSpecs(("Do you think that discrimination "
                                                              "against whites has become as big a"
                                                              "problem as discrimination against "
                                                              "blacks and other minorities?"),
                                                              "",
                                                              {"Completely agree": "Yes",
                                                              "Mostly agree": "Yes",
                                                              "Mostly disagree": "No",
                                                              "Completely disagree": "No"}),
                "stranger_in_own_country": PromptSpecs(("Do you think the US has changed "
                                                        " so much that you feel like a "
                                                        "stranger in your own country?"),
                                                        "",
                                                        {"Completely agree": "Yes",
                                                        "Mostly agree": "Yes",
                                                        "Mostly disagree": "No",
                                                        "Completely disagree": "No"}),
                "demographic_change_opinion": PromptSpecs(("By 2045, minorities will together be a majority "
                                                           "in the US. Do you think the impact of the "
                                                           "coming demographic change will be positive "
                                                           "or negative?"),
                                                           "I think the coming demographic change will be",
                                                           {"Mostly positive": "positive",
                                                           "Mostly negative": "negative"}),
                "use_of_racism": PromptSpecs(("Do you think racial minorities use racism "
                                              "as an excuse more than they should?"),
                                              "",
                                              {"Completely agree": "Yes",
                                              "Mostly agree": "Yes",
                                              "Mostly disagree": "No",
                                              "Completely disagree": "No"}),
                "elect_non_christian": PromptSpecs(("How do you think electing more non "
                                                    "Christian people to political office "
                                                    "would make things in the US?"),
                                                    "I think things would be",
                                                    {"Better": "better",
                                                    "Worse": "worse",
                                                    "Not much different": "the same"}),
        }

if __name__ == "__main__":
    ds = PRRIDataset()
    # Uncomment this to see a sample of your prompts
    # First prompt for each DV
    #for dv_name in ds.dvs.keys():
    #    dv_prompts = ds.prompts[dv_name]
    #    print(dv_prompts[list(dv_prompts.keys())[0]])
    #    print()
