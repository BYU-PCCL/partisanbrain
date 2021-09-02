from dataset import Dataset
from dataset import PromptSpecs
import pandas as pd


class BaylorReligionSurveyDataset(Dataset):

    def __init__(self):
        survey_fname = "data/baylor/baylor.sav"
        super().__init__(survey_fname)
        #issues:
        # no region demographic

    def _filter_demographics(self, df):
        new_df = df[~df["religion"].isin(["Other","Don't know"])]
        new_df = new_df[new_df["race"] != "No races chosen"]

        return new_df

    def _filter_to_usa(self, df):
        """Return a new dictionary where all respondents are from USA"""
        return df

    def _get_dv_col_names(self):
        return {"T7G"  : "tech_oppor",
                "MP4A" : "trans_restrooms",
                "Q61D" : "gay_is_it_choice",
                "MP4G" : "husband_salary",
                "MP4F" : "women_childcare",
                "MP4D" : "men_suited_politics",
                "MP4J" : "refugees_terrorist_threat",
                "MP4K" : "mexican_immigrants_criminals",
                "Q45"  : "life_happiness",
                "H13E" : "depressed_freq",
                "H12"  : "days_of_exercise",
                "MP4H" : "police_racial_treatment",
                "MP4I" : "racial_violence",
                "Q17"  : "bible_beliefs",
                "R22"  : "heaven",
                "Q19A" : "god_concern_for_world",
                "Q19D" : "god_concern_for_individuals",
                "Q4"   : "church_attendance",
                "MP12F": "prayer_in_school",
                "R20F" : "gods_plan"}

    def _get_demographic_col_names(self):
        return {
            "AGE"   : "age",
            "Q77"   : "gender",
            "Q32"   : "party",
            "I_EDUC": "edu",
            "Q31"   : "ideology",
            "Q95"   : "income",
            "Q1"    : "religion",
            "RACE"  : "race",
            "D9"    : "marital",
        }

    def _make_backstory(self, row):
        backstory = []

        #AGE
        backstory.append(f"I am {row['age']} years old.")

        #GENDER
        if row["gender"] == "Other":
            backstory.append("I don't identify as male or female.")
        else:
            backstory.append(f"I am {row['gender'].lower()}.")

        #POLITICAL PARTY
        backstory.append(f"In terms of partisan politics, I am a {row['party'].lower()}.")

        #EDUCATION
        if row["edu"] == "No high school degree":
            backstory.append("I did not graduate high school.")
        if row["edu"] == "High school graduate (Grade 12 with diploma or GED certificate)":
            backstory.append("I am a high school graduate.")
        if row["edu"] == "Some college":
            backstory.append("I have some college education.")
        if row["edu"] == "Four year bachelor's degree from a college or university (e.g., BS, BA, AB)":
            backstory.append("I have a bachelor's degree from a college or university.")
        if row["edu"] == "Postgraduate":
            backstory.append("I have a postgraduate degree.")

        #POLITICAL IDEOLOGY
        backstory.append(f"In terms of political ideology, I'd consider myself to be {row['ideology'].lower()}.")

        #INCOME
        backstory.append(f"My family income is {row['income']} per year.")

        #RELIGION
        #main cases
        case1 = pd.DataFrame(["Assemblies of God",
                  "Brethren",
                  "Christian & Missionary Alliance",
                  "Christian Reformed",
                  "Christian Science",
                  "Congregational",
                  "Holiness",
                  "Lutheran",
                  "Pentecostal",
                  "Unitarian Universalist"
                  ])
        case2 = pd.DataFrame(["Baha'i",
                              "Baptist",
                              "Buddhist",
                              "Hindu",
                              "Mennonite",
                              "Methodist",
                              "Muslim",
                              "Presbyterian",
                              "Seventh-Day Adventist",
                              "Sikh"
                             ])
        case3 = pd.DataFrame(["Adventist",
                              "African Methodist",
                              "Anabaptist"])
        case4 = pd.DataFrame(["Bible Church",
                              "Church of Christ",
                              "Church of God",
                              "Church of the Nazarene",
                              "Salvation Army",
                              "United Church of Christ" ])

        if row["religion"] in list(case1):
            backstory.append(f"I am a member of the {row['religion']} faith.")

        if row["religion"] in list(case2):
            backstory.append(f"In terms of religion, I am a {row['religion']}.")

        if row["religion"] in list(case3):
            backstory.append(f"In terms of religion, I am an {row['religion']}.")

        if row["religion"] in list(case4):
            backstory.append(f"I am a member of the {row['religion']}.")

        #special cases
        if row["religion"] == "Asian Folk Religion":
            backstory.append("I am part of an Asian Folk Religion.")
        elif row["religion"] == "Catholic/Roman Catholic":
            backstory.append("In terms of religion, I am a Catholic.")
        elif row["religion"] == "Episcopal/Anglican":
            backstory.append("I am a member of the Anglican faith.")
        elif row["religion"] == "Jehovah's Witnesses":
            backstory.append("In terms of religion, I am a Jehovah's Witness")
        elif row["religion"] == "Jewish":
            backstory.append("In terms of religion, I am Jewish")
        elif row["religion"] == "Latter-day Saints":
            backstory.append("In terms of religion, I am a Mormon.")
        elif row["religion"] == "Orthodox (Eastern, Russian, Greek)":
            backstory.append("I am a member of the Orthodox Catholic Church.")
        elif row["religion"] == "Quaker/Friends":
            backstory.append("In terms of religion, I am a Quaker.")
        elif row["religion"] == "Reformed Church in America/Dutch Reformed":
            backstory.append("I am a member of the Dutch Reformed Church.")
        elif row["religion"] == "Non-denominational Christian":
            backstory.append("I am a Christian.")
        elif row["religion"] == "No religion":
            backstory.append("I am not religious.")


        #RACE
        backstory.append(f"I am {row['race'].lower()}.")


        #REGION
        #missing for baylor study

        #MARITAL STATUS
        if row["marital"] == "Single/never been married":
            backstory.append("I am single.")
        elif row["marital"] == "Domestic partnership/living with partner (not legally married)":
            backstory.append("I am not married, but I am living with my partner.")
        else:
            backstory.append(f"I'm {row['marital'].lower()}.")

        return " ".join(backstory)

    def _get_col_prompt_specs(self):
        return {
            "tech_oppor": PromptSpecs(
                            question=("To what extent do you agree with the following? "
                            "Technology gives me new and better employment "
                            "opportunities."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "trans_restrooms": PromptSpecs(
                            question=("Please rate the extent to which you agree "
                            "or disagree with the following statements: "
                            "Transgender people should be allowed to "
                            "use the public restroom of their choice."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "gay_is_it_choice": PromptSpecs(
                            question=("Please rate the extent to which you agree "
                            "or disagree with the following "
                            "statements: People choose to be "
                            "gay/lesbian."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "husband_salary": PromptSpecs(
                            question=("Please rate the extent to which you agree "
                            "or disagree with the following statements: "
                            "A husband should earn a larger salary than "
                            "his wife."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "women_childcare": PromptSpecs(
                            question=("Please rate the extent to which you agree "
                            "or disagree with the following "
                            "statements: It is God's will that women "
                            "care for children."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "men_suited_politics": PromptSpecs(
                            question=("Please rate the extent to which you "
                            "agree or disagree with the following "
                            "statements: Men are better suited "
                            "emotionally for politics than women."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "refugees_terrorist_threat": PromptSpecs(
                            question=("Please rate the extent to which "
                            "you agree or disagree with the "
                            "following statements: Refugees "
                            "from the Middle East pose a "
                            "terrorist threat to the United "
                            "States."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "mexican_immigrants_criminals":PromptSpecs(
                            question=("Please rate the extent to "
                            "which you agree or disagree "
                            "with the following "
                            "statements: Illegal "
                            "immigrants from Mexico are "
                            "mostly dangerous criminals."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "life_happiness": PromptSpecs(
                            question=("In general, how happy are you with your "
                            "life as a whole these days?"),
                            answer_prefix="In general I am",
                            answer_map={"Not too happy": "sad",
                                        "Pretty happy": "happy",
                                        "Very happy": "happy"}),
            "depressed_freq": PromptSpecs(
                            question=("In the past WEEK, about how often have you "
                            "had the following feelings? I felt "
                            "depressed."),
                            answer_prefix="In the last week I have felt depressed",
                            answer_map={"Never": "never",
                                        "Hardly ever": "rarely",
                                        "Some of the time": "sometimes",
                                        "Most or all of the time": "frequently"}),
            "days_of_exercise": PromptSpecs(
                            question=("How many DAYS per WEEK do you do exercise "
                            "for at least 30 minutes?"),
                            answer_prefix="I exercise",
                            answer_map={0.0: "zero times",
                                        1.0: "one time",
                                        2.0: "two times",
                                        3.0: "three times",
                                        4.0: "four times",
                                        5.0: "five times",
                                        6.0: "six times",
                                        7.0: "seven times"}),
            "police_racial_treatment": PromptSpecs(
                            question=("Please rate the extent to which "
                            "you agree or disagree with the "
                            "following statements: Police "
                            "officers in the United States "
                            "treat blacks the same as whites."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "racial_violence": PromptSpecs(
                            question=("Please rate the extent to which you agree "
                            "or disagree with the following statements: "
                            "Police officers in the United States shoot "
                            "blacks more often because they are more "
                            "violent than whites."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "bible_beliefs": PromptSpecs(
                            question=("Which one statement comes closest to your personal beliefs about the Bible?"),
                            answer_prefix="I believe the bible is",
                            answer_map={"The Bible means exactly what it says. It should be "
                                        "taken literally, word-for-word, on all subjects": "literal",
                                        "The Bible is perfectly true, but it should not be taken literally, word-for-word. We must interpret its meaning": "true but not literal",
                                        "The Bible contains some human error": "flawed",
                                        "The Bible is an ancient book of history and legends": "legend"}),
            "heaven": PromptSpecs(
                            question=("How certain are you that you will get into Heaven?"),
                            answer_prefix="I am ",
                            answer_map={"Very certain": "certain",
                                        "Quite certain": "certain",
                                        "Somewhat certain": "certain",
                                        "Not very certain": "uncertain",
                                        "Not at all certain": "uncertain",
                                        "I don't believe in Heaven": "disbelieving"}),
            "god_concern_for_world": PromptSpecs(
                            question=("Based on your personal understanding "
                            "of God, to what extent do "
                            "you agree or disagree with "
                            "the statement 'God is "
                            "concerned with the well-being of the "
                            "world'?"),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "god_concern_for_individuals": PromptSpecs(
                            question=("Based on your personal "
                            "understanding of God, to what "
                            "extent do you "
                            "agree or disagree with the "
                            "statement 'God is "
                            "concerned with my personal "
                            "well-being'?"),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "church_attendance": PromptSpecs(
                            question=("How often do you attend religious "
                            "services at a place of worship?"),
                            answer_prefix="I attend religious services",
                            answer_map={"Never - Skip to Question 12": "never",
                                        "Less than once a year": "rarely",
                                        "Once or twice a year": "annually",
                                        "Several times a year": "sometimes",
                                        "Once a month": "monthly",
                                        "2 to 3 times a month": "biweekly",
                                        "About once a week": "weekly",
                                        "Several times a week": "frequently"}),
            "prayer_in_school": PromptSpecs(
                            question=("To what extent do you agree "
                            "or disagree with the "
                            "statement 'The federal government "
                            "should allow prayer in public schools'?"),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree",
                                        "Undecided": "am not sure"}),
            "gods_plan": PromptSpecs(
                            question=("To what extent do you agree or "
                            "disagree with the statement 'When "
                            "good or bad things happen to me, I see it as "
                            "part of God's plan for me'?"),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
        }



if __name__ == "__main__":
    ds = BaylorReligionSurveyDataset()
    bp()
    # Uncomment this to see a sample of your prompts
    # First prompt for each DV
    for dv_name in ds.dvs.keys():
        dv_prompts = ds.prompts[dv_name]
        print(dv_prompts[list(dv_prompts.keys())[0]])
        print()
