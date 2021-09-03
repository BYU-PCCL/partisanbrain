from dataset import Dataset
from dataset import PromptSpecs


class BaylorReligionSurveyDataset(Dataset):

    def __init__(self):
        survey_fname = "data/baylor.sav"
        super().__init__(survey_fname)
        # Issues:
        #   No region demographic

    def _filter_demographics(self, df):
        df = df[~df["religion"].isin(["Other", "Don't know"])]
        df = df[df["race"] != "No races chosen"]

        return df

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

        backstory.append(f"I am {int(row['age'])} years old.")

        #GENDER
        if row['gender'] == "Other (please specify)":
            backstory.append("I do not identify as male or female.")
        else:
            backstory.append(f"I am {row['gender'].lower()}.")

        #POLITICAL PARTY
        party = row['party']
        if "Leaning" in party:
            if "Democrat" in party:
                backstory.append("In terms of politics, I lean Democrat.")
            else:
                backstory.append("In terms of politics, I lean Republican.")
        elif "Independent" in party:
            backstory.append("In terms of politics, I am an Independent.")
        elif "moderate" or "strong" in party.lower():
            degree = party.split(" ")[0].lower()
            specific_party = party.split(" ")[1]
            backstory.append(f"In terms of politics, I am a {degree} {specific_party}.")
        else:
            backstory.append(f"In terms of politics, I am {party}")

        #EDUCATION
        if row["edu"] == "No high school degree":
            backstory.append("I did not graduate from high school.")
        if row["edu"] == "High school graduate (Grade 12 with diploma or GED certificate)":
            backstory.append("I graduated from high school.")
        if row["edu"] == "Some college":
            backstory.append("I went to some college.")
        if row["edu"] == "Four year bachelor's degree from a college or university (e.g., BS, BA, AB)":
            backstory.append("I have a bachelor's degree.")
        if row["edu"] == "Postgraduate":
            backstory.append("I went to graduate school.")

        #POLITICAL IDEOLOGY
        backstory.append(f"In terms of political ideology, I would consider myself to be {row['ideology'].lower()}.")

        #INCOME
        backstory.append(f"My family income is {row['income']} per year.")

        #RELIGION
        #main cases
        case1 = ["Assemblies of God",
                  "Brethren",
                  "Christian & Missionary Alliance",
                  "Christian Reformed",
                  "Christian Science",
                  "Congregational",
                  "Holiness",
                  "Lutheran",
                  "Pentecostal",
                  "Unitarian Universalist"
                  ]
        case2 = [ "Baha'i",
                  "Baptist",
                  "Buddhist",
                  "Hindu",
                  "Mennonite",
                  "Methodist",
                  "Muslim",
                  "Presbyterian",
                  "Seventh-Day Adventist",
                  "Sikh"
                 ]
        case3 = ["Adventist",
                 "African Methodist",
                 "Anabaptist"]
        case4 = ["Bible Church",
                 "Church of Christ",
                 "Church of God",
                 "Church of the Nazarene",
                 "Salvation Army",
                 "United Church of Christ" ]

        if row["religion"] in case1:
            backstory.append(f"I am a member of the {row['religion']} faith.")

        if row["religion"] in case2:
            backstory.append(f"In terms of religion, I am a {row['religion']}.")

        if row["religion"] in case3:
            backstory.append(f"In terms of religion, I am an {row['religion']}.")

        if row["religion"] in case4:
            backstory.append(f"I am a member of the {row['religion']}.")

        #special cases
        if row["religion"] == "Asian Folk Religion":
            backstory.append("I am part of an Asian Folk Religion.")
        elif row["religion"] == "Catholic/Roman Catholic":
            backstory.append("In terms of religion, I am a Catholic.")
        elif row["religion"] == "Episcopal/Anglican":
            backstory.append("I am a member of the Anglican faith.")
        elif row["religion"] == "Jehovah's Witnesses":
            backstory.append("In terms of religion, I am a Jehovah's Witness.")
        elif row["religion"] == "Jewish":
            backstory.append("In terms of religion, I am Jewish.")
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
        if "multiple" in row["race"].lower():
            backstory.append("I am multi-racial.")
        elif "american indian" in row["race"].lower():
            backstory.append("I am American Indian.")
        elif "pacific islander" in row["race"].lower():
            backstory.append("I am Pacific Islander.")
        else:
            backstory.append(f"I am {row['race'].capitalize()}.")


        #REGION
        #missing for baylor study

        #MARITAL STATUS
        if row["marital"] == "Single/never been married":
            backstory.append("I am single.")
        elif row["marital"] == "Domestic partnership/living with partner (not legally married)":
            backstory.append("I am not married, but I am living with my partner.")
        elif row['marital'] == "Separated":
            backstory.append("I am separated from my spouse.")
        else:
            backstory.append(f"I am {row['marital'].lower()}.")
        
        #WE DIDN'T USE THIS IN THE MEGA!
        backstory.append("It is spring 2017.")

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
                            answer_prefix="in general I am",
                            answer_map={"Not too happy": "sad",
                                        "Pretty happy": "happy",
                                        "Very happy": "happy"}),
            "depressed_freq": PromptSpecs(
                            question=("In the past WEEK, about how often have you "
                            "had the following feelings? I felt "
                            "depressed."),
                            answer_prefix="in the last week I have felt depressed",
                            answer_map={"Never": "never",
                                        "Hardly ever": "rarely",
                                        "Some of the time": "sometimes",
                                        "Most or all of the time": "frequently"}),
            "days_of_exercise": PromptSpecs(
                            question=("Did you exercise for at least 30 minutes any day last week?"),
                            answer_prefix="",
                            answer_map={0.0: "no",
                                        1.0: "yes",
                                        2.0: "yes",
                                        3.0: "yes",
                                        4.0: "yes",
                                        5.0: "yes",
                                        6.0: "yes",
                                        7.0: "yes"}),
            "police_racial_treatment": PromptSpecs(
                            question=("Please rate the extent to which "
                            "you agree or disagree with the "
                            "following statements: Police "
                            "officers in the United States "
                            "treat Blacks the same as Whites."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "racial_violence": PromptSpecs(
                            question=("Please rate the extent to which you agree "
                            "or disagree with the following statements: "
                            "Police officers in the United States shoot "
                            "Blacks more often because they are more "
                            "violent than Whites."),
                            answer_prefix="I",
                            answer_map={"Strongly agree": "agree",
                                        "Agree": "agree",
                                        "Disagree": "disagree",
                                        "Strongly disagree": "disagree"}),
            "bible_beliefs": PromptSpecs(
                            question=("Which one statement comes closest to your personal beliefs about the Bible?"),
                            answer_prefix="I believe the Bible is",
                            answer_map={"The Bible means exactly what it says. It should be "
                                        "taken literally, word-for-word, on all subjects": "literal",
                                        "The Bible is perfectly true, but it should not be taken literally, word-for-word. We must interpret its meaning": "true but not literal",
                                        "The Bible contains some human error": "flawed",
                                        "The Bible is an ancient book of history and legends": "legend"}),
            "heaven": PromptSpecs(
                            question=("How certain are you that you will get into Heaven?"),
                            answer_prefix="I am",
                            answer_map={"Very certain": "certain",
                                        "Quite certain": "certain",
                                        "Somewhat certain": "certain",
                                        "Not very certain": "uncertain",
                                        "Not at all certain": "uncertain",
                                        "I don't believe in Heaven": "not someone who believes in heaven"
                                        }),
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
                            question=("Do you attend religious services at a place of worship at least weekly?"),
                            answer_prefix="",
                            answer_map={"Never - Skip to Question 12": "no",
                                        "Less than once a year": "no",
                                        "Once or twice a year": "no",
                                        "Several times a year": "no",
                                        "Once a month": "no",
                                        "2 to 3 times a month": "no",
                                        "About once a week": "yes",
                                        "Several times a week": "yes"}),
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
                                        }),
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
    from experiment import Experiment
    from baylor_religion_survey_dataset import BaylorReligionSurveyDataset
    import openai

    openai.api_key = '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'

    # Set up the experiment
    ds = BaylorReligionSurveyDataset()
    print(ds._samples, "samples")
    e = Experiment(ds, gpt_3_engine="davinci")

    # Run the experiment
    e.run()

    # Save the results
    e.save_results(f"baylor_mega.pkl")
