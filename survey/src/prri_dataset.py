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
        return {"AGE"          : "age",
                "AGE4"         : "age4",
                "AGE7"         : "age7",
                "GENDER"       : "gender",
                "PARTY"        : "party",
                "EDUC"         : "education",
                "IDEO"         : "ideology",
                "INCOME"       : "income",
                "RELIG"        : "religion",
                "RACETHNICITY" : "race_ethnicity",
                "REGION9"      : "region",
                "MARITAL"      : "marital_status"}

    def _get_dv_col_names(self):
        return {"Q20A" : "electing_women",
                "Q20B" : "electing_LGBTQIA",
                "Q29"  : "do_more_for_LGBTQIA",
                "Q27P" : "immigrant_preference",
                "Q18C" : "putin_opinion",
                "Q30"  : "view_on_immigration",
                "Q31"  : "perspective_on_immigration",
                "Q35C" : "laws_preventing_refugees",
                "Q36"  : "immigrant_citizenship",
                "Q1"   : "voting_frequency",
                "Q5"   : "trump_job_opinion",
                "Q20C" : "electing_minorities",
                "Q22"  : "police_brutality_pattern",
                "Q26E" : "asian_discrimination",
                "Q26F" : "hispanic_discrimination",
                "Q27O" : "white_vs_black_discrimination",
                "Q27R" : "stranger_in_own_country",
                "Q33"  : "demographic_change_opinion",
                "Q27N" : "use_of_racism",
                "Q20D" : "elect_non_christian"}

    def _filter_demographics(self, df):
        df = df[df["party"].isin(["A Democrat",
                                  "A Republican",
                                  "An Independent"])]
        df = df[df["race_ethnicity"] != "Other, non-Hispanic"]
        df = df[df["party"] != "Other [SPECIFY]"]

        problematic_values = ["Refused", "Something else", "Don't know (VOL.)", "Skipped on web"]
        for col_name in list(df):
            df = df[~df[col_name].isin(problematic_values)]

        return df

    def _make_backstory(self, row):
        backstory = []

        # Age
        backstory.append(f"I am {int(row['age'])} years old.")

        # Gender
        backstory.append(f"I am {row['gender'].lower()}.")

        # Party
        if row["party"] == "An Independent":
            backstory.append("In terms of political party I am an Independent.")
        else:
            if "A Democrat" == row["party"]:
                backstory.append("In terms of political party "
                                 "I am a Democrat")
            elif "A Republican" == row['party']:
                backstory.append("In terms of political party "
                                 "I am a Republican")

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
            backstory.append("My annual family income is less than $5,000.")
        elif "or more" in row["income"]:
            backstory.append("My annual family income is $200,000 or more.")
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
        race = row['race_ethnicity'].replace(', non-Hispanic', '')
        if "Two plus" in race:
            backstory.append(f"I'm multi-racial.")
        else:
            backstory.append(f"I'm {race}.")

        # Region
        if row["region"] == "New England" or row["region"] == "Mid-Atlantic":
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
            backstory.append("I live in the Western United States.")

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
        return {"electing_women": PromptSpecs(
                    question="How do you think electing more women to political office would make things in the US?",
                    answer_prefix="I think things would be",
                    answer_map={"Better": "better",
                                "Worse": "worse",
                                "Not much different": "the same"}),
                "electing_LGBTQIA": PromptSpecs(
                    question="How do you think electing more lesbian, gay, bisexual, and transgender people to political office would make things in the US?",
                    answer_prefix="I think things would be",
                    answer_map={"Better": "better",
                                "Worse": "worse",
                                "Not much different": "the same"}),
                "do_more_for_LGBTQIA": PromptSpecs(
                    question="Do you think our country has made the changes needed to give gay and lesbian people equal rights in America?",
                    answer_prefix="",
                    answer_map={"Our country has made the changes "
                                "needed to give gay and lesbian "
                                "people equal rights with other "
                                "Americans": "yes",
                                "Our country needs to continue making "
                                "changes to give gay and lesbian people "
                                "equal rights with other Americans": "no"}),
                "immigrant_preference": PromptSpecs(
                    question="Do you think we should give preference to immigrants from Western Europe. who share our values?",
                    answer_prefix="",
                    answer_map={"Completely agree": "yes",
                                "Mostly agree" : "yes",
                                "Mostly disagree": "no",
                                "Completely disagree": "no"}),
                "putin_opinion": PromptSpecs(
                    question="How would you describe your overall opinion of Russian President Vladimir Putin?",
                    answer_prefix="my opinion is",
                    answer_map={"Very favorable": "favorable",
                                "Mostly favorable" : "favorable",
                                "Mostly unfavorable": "unfavorable",
                                "Very unfavorable": "unfavorable",
                                "Have not heard of": "no opinion"}),
                "view_on_immigration": PromptSpecs(
                    question="Do you think that, in general, the growing number of newcomers from other countries to the US is good or bad?",
                    answer_prefix="the growing number of newcomers is",
                    answer_map={"Threatens traditional American "
                                "customs and values": "bad",
                                "Strengthens American society" : "good"}),
                "perspective_on_immigration": PromptSpecs(
                    question="Do you think that immigrants today are good or bad for the US?",
                    answer_prefix="I think that immigrants today are",
                    answer_map={"Immigrants today strengthen our country "
                                "because of their hard work and talents": "good",
                                "Immigrants today are a burden on our country because they take our jobs, housing and health care": "bad"}),
                "laws_preventing_refugees": PromptSpecs(
                    question="Do you favor or oppose passing a law to prevent refugees from entering the US?",
                    answer_prefix="I",
                    answer_map={"Strongly favor": "favor",
                                "Favor" : "favor",
                                "Oppose": "oppose",
                                "Strongly oppose": "oppose"}),
                "immigrant_citizenship": PromptSpecs(
                    question="How should the US immigration system should deal with imigrants who are currently living in the US illegally?",
                    answer_prefix="the US immigration system should",
                    answer_map={"Allow them a way to become citizens provided "
                                "they meet certain requirements": "allow them to become citizens",
                                "Allow them to become permanent legal residents, "
                                "but not citizens" : "allow them to become residents, not citizens",
                                "Identify and deport them": "deport them"}),
                "voting_frequency": PromptSpecs(
                    question="How often would you say you vote?",
                    answer_prefix="I vote",
                    answer_map={"Always": "always",
                                "Nearly always" : "sometimes",
                                "In about half of elections": "sometimes",
                                "Seldom": "sometimes",
                                "Never": "never"}),
                "trump_job_opinion": PromptSpecs(
                    question="Do you approve of the job Donald Trump is doing as president?",
                    answer_prefix="",
                    answer_map={"Strongly approve": "yes",
                                "Somewhat approve" : "yes",
                                "Somewhat disapprove": "no",
                                "Strongly disapprove": "no"}),
                "electing_minorities": PromptSpecs(
                    question="How do you think electing more people from racial and ethnic minority groups to political office would make things in the US?",
                    answer_prefix="I think things would be",
                    answer_map={"Better": "better",
                                "Worse": "worse",
                                "Not much different": "the same"}),
                "police_brutality_pattern": PromptSpecs(
                    question="Do you think the recent killings of African American men by police are isolated events or part of a broader pattern how how police treat African Americans?",
                    answer_prefix="I think they are",
                    answer_map={"Isolated incidents": "isolated incidents",
                                "Part of a broader pattern": "a broader pattern"}),
                "asian_discrimination": PromptSpecs(
                    question="In the US today is there a lot of discrimination against Asians?",
                    answer_prefix="",
                    answer_map={"Yes, there is a lot of discrimination": "yes",
                                "No, not a lot of discrimination": "no"}),
                "hispanic_discrimination": PromptSpecs(
                    question="In the US today is there a lot of discrimination against Hispanics?",
                    answer_prefix="",
                    answer_map={"Yes, there is a lot of discrimination": "yes",
                                "No, not a lot of discrimination": "no"}),
                "white_vs_black_discrimination": PromptSpecs(
                    question="Do you think that discrimination against whites has become as big a problem as discrimination against blacks and other minorities?",
                    answer_prefix="",
                    answer_map={"Completely agree": "yes",
                                "Mostly agree": "yes",
                                "Mostly disagree": "no",
                                "Completely disagree": "no"}),
                "stranger_in_own_country": PromptSpecs(
                    question="Do you think the US has changed so much that you feel like a stranger in your own country?",
                    answer_prefix="",
                    answer_map={"Completely agree": "yes",
                                "Mostly agree": "yes",
                                "Mostly disagree": "no",
                                "Completely disagree": "no"}),
                "demographic_change_opinion": PromptSpecs(
                    question="By 2045, minorities will together be a majority in the US. Do you think the impact of the coming demographic change will be positive or negative?",
                    answer_prefix="I think the coming demographic change will be",
                    answer_map={"Mostly positive": "positive",
                                "Mostly negative": "negative"}),
                "use_of_racism": PromptSpecs(
                    question="Do you think racial minorities use racism as an excuse more than they should?",
                    answer_prefix="",
                    answer_map={"Completely agree": "yes",
                                "Mostly agree": "yes",
                                "Mostly disagree": "no",
                                "Completely disagree": "no"}),
                "elect_non_christian": PromptSpecs(
                    question="How do you think electing more non Christian people to political office would make things in the US?",
                    answer_prefix="I think things would be",
                    answer_map={"Better": "better",
                                "Worse": "worse",
                                "Not much different": "the same"}),
        }

if __name__ == "__main__":
    ds = PRRIDataset()
    backstories = ds.get_backstories_all_demos()
    for backstory in backstories:
        print(f"{backstory[0]}\n\n{backstory[1]}")
    prompts = ds.get_prompts_sample()
    for prompt in prompts:
        print(f"{prompt}\n\n")
    # Uncomment this to see a sample of your prompts
    # First prompt for each DV
    # prompts = []
    # for dv_name in ds.dvs.keys():
    #     dv_prompts = ds.prompts[dv_name]
    #     for row_idx in dv_prompts.keys():
    #         prompts.append(dv_prompts[row_idx])
    #     for prompt in prompts:
    #        print(prompt)
    #        print()

