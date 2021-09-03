from dataset import Dataset
from dataset import PromptSpecs

class AddHealthDataset(Dataset):

    def __init__(self):
        survey_fname = "data/21600-0022-Data.sav"
        super().__init__(survey_fname)

    def _filter_to_usa(self, df):
        # all respondents in this survey should be from the USA
        return df

    def _get_demographic_col_names(self):
        # note: these codes are different than those in the spreadsheet. Those are wave 5, these are wave 4
        return {"H4OD1Y" : "age",
                "BIO_SEX4" : "gender",
                "H4ED2"    : "education",
                "H4DA28"   : "ideology",
                "H4EC1"    : "income",
                "H4RE1"    : "religion",
                "H4IR4"    : "race_ethnicity",
                "H4TR1"    : "marital_status"}

    def _get_dv_col_names(self):
        return {"H4DS20"   : "shot_or_stabbed",
                "H4CJ1"    : "arrested",
                "H4DS11"   : "physical_fight",
                "H4CJ10"   : "convicted_of_charges",
                "H4DS5"    : "sell_drugs",
                "H4HS9"    : "counseling",
                "H4MH19"   : "sadness_family",
                "H4PE6"    : "worrying",
                "H4SE2"    : "suicide",
                "H4PE23"   : "optimism",
                "H4MH24"   : "happiness",
                "H4GH8"    : "fast_food",
                "H4DA1"    : "hours_of_tv",
                "H4DA5"    : "individual_sports",
                "H4TO1"    : "smoked_cigarette",
                "H4MA3"    : "physical_child_abuse",
                "H4TO34"   : "age_of_first_drink",
                "H4ID8"    : "car_accidents",
                "H4TO33"   : "drinking",
                "H4RE10"   : "prayer_in_private"}

    def _filter_demographics(self, df):
        problamatic_values = ["(9) Other", "Refused", "Don't know", "Missing"]
        for col_name in list(df):
            df = df[~df[col_name].isin(problamatic_values)]

        return df

    def _make_backstory(self, row):
        backstory = []

        # Age
        age = int(row['age'])
        backstory.append(f"I was born in {age}.")

        # Gender
        gender = row['gender'][4:].lower()
        backstory.append(f"I am {gender}.")

        # Party
        #missing

        # Education
        grad_school_values = ["(8) Some graduate school",
                              "(9) Completed a master's degree",
                              "(10) Some graduate training beyond a master's degree",
                              "(11) Completed a doctoral degree",
                              "(12) Some post baccalaureate professional education",
                              "(13) Completed post baccalaureate professional education"]
        if row["education"] in grad_school_values:
            backstory.append("I went to graduate school.")
        elif row["education"] == "(7) Completed college (bachelor's degree)":
            backstory.append("I completed college.")
        elif row["education"] == "(6) Some college":
            backstory.append("I've completed some college.")
        elif row["education"] == "(3) High school graduate":
            backstory.append("I graduated from high school.")
        elif row["education"] == "(2) Some high school":
            backstory.append("I didn't graduate from high school.")
        elif row["education"] == "(1) 8th grade or less":
            backstory.append("I didn't go to high school.")
        else:
            backstory.append("I didn't go to college.")

        # Ideology
        ideology_dic = {
            "(1) Very conservative" : "very conservative" ,
            "(2) Conservative" : "convservative" ,
            "(3) Middle-of-the-road" : "moderate" ,
            "(4) Liberal" : "liberal" ,
            "(5) Very liberal" : "very liberal" ,
        }
        backstory.append(f"In terms of political ideology, I'm {ideology_dic[row['ideology']]}.")

        # Income
        if "to " in row["income"]:

            income = row['income']
            income = income.split()[1:]
            income = " ".join(income)
            low, high = income.split(" to ")
            backstory.append(("My annual family income is "
                              f"between {low} and {high}."))
        elif row["income"] == "(1) Less than $5,000":
            backstory.append("My annual family income is less than $5000.")
        else:
            backstory.append("My annual family income is more than $150,000.")

        # Religiosity
        religion = row['religion'][4:]

        if "Non" in religion:
            backstory.append(("In terms of religion, I am atheist/agnostic."))
        else:
            if "Protestant" in religion:
                religion = "Protestant"
            elif "Other Christian" in row["religion"]:
                religion = "Christian"
            backstory.append(f"In terms of religion I am {religion}.")

        # Race/Ethnicity
        race_ethnicity = row['race_ethnicity']
        if "White" in race_ethnicity:
            race_ethnicity = "White"
        elif "Black" in race_ethnicity:
            race_ethnicity = "Black"
        elif "Indian" in race_ethnicity:
            race_ethnicity = "American Indian"
        else:
            race_ethnicity = "Asian or Pacific Islander"
        backstory.append(f"I am {race_ethnicity}.")

        # Region
        # Missing

        # Marital Status
        num_times_married = int(row['marital_status'])
        if num_times_married > 0:
            if num_times_married == 1:
                backstory.append("I have been married 1 time.")
            else:
                backstory.append(f"I have been married {num_times_married} times.")
        else:
            backstory.append("I have never been married.")

        # Date
        backstory.append("It's November 2020.")

        return " ".join(backstory)

    def _get_col_prompt_specs(self):
        return {
                "shot_or_stabbed": PromptSpecs(
                    question="Have you shot or stabbed someone in the past 12 months?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "yes",
                                "(0) No": "no"}),
                "arrested": PromptSpecs(
                    question="Have you ever been arrested?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "yes",
                                "(0) No": "no"}),
                "physical_fight": PromptSpecs(
                    question="In the past 12 months, did you get into a serious physical fight?",
                    answer_prefix="",
                    answer_map={"(1) 1 or 2 times": "yes",
                                "(2) 3 or 4 times": "yes",
                                "(3) 5 or more times": "yes",
                                "(0) Never": "no"}),
                "convicted_of_charges": PromptSpecs(
                    question="Have you ever been convicted of or pled guilty to any charges other than a minor traffic violation?",
                    answer_prefix="",
                    answer_map={"(1) Once": "yes",
                                "(2) More than once": "yes",
                                "(0) No": "no"}),
                "sell_drugs": PromptSpecs(
                    question="In the past 12 months, did you sell marijuana or other drugs?",
                    answer_prefix="",
                    answer_map={"(1) 1 or 2 times": "yes",
                                "(2) 3 or 4 times": "yes",
                                "(3) 5 or more times": "yes",
                                "(0) Never": "no"}),
                "counseling": PromptSpecs(
                    question="In the past 12 months have you received psychological or emotional counseling?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "yes",
                                "(0) No": "no"}),
                "sadness_family": PromptSpecs(
                    question="In the past seven days, have you felt as though you can't shake off the blues, even with help from your family and your friends?",
                    answer_prefix="",
                    answer_map={"(1) Sometimes": "yes",
                                "(2) A lot of the time": "yes",
                                "(3) Most of the time or all of the time": "yes",
                                "(0) Never or rarely": "no"}),
                "worrying": PromptSpecs(
                    question="Do you worry about things?",
                    answer_prefix="",
                    answer_map={"(1) Strongly agree": "yes",
                                "(2) Agree": "yes",
                                "(3) Neither agree nor disagree": "no",
                                "(4) Disagree": "no",
                                "(5) Strongly disagree": "no"}),
                "suicide": PromptSpecs(
                    question="Have you actually attempted suicide in the past 12 months?",
                    answer_prefix="",
                    answer_map={"(1) Once": "yes",
                                "(2) Twice": "yes",
                                "(3) 3 or 4 times": "yes",
                                "(4) 5 or more times": "yes",
                                "(0) None": "no"}),  
                "optimism": PromptSpecs(
                    question="Do you expect more good things to happen to you than bad?",
                    answer_prefix="",
                    answer_map={"(1) Strongly agree": "yes",
                                "(2) Agree": "yes",
                                "(3) Neither agree nor disagree": "no",
                                "(4) Disagree": "no",
                                "(5) Strongly disagree": "no"}),
                "happiness": PromptSpecs(
                    question="Did you feel happy in the past 7 days?",
                    answer_prefix="",
                    answer_map={"(1) Sometimes": "yes",
                                "(2) A lot of the time": "yes",
                                "(3) Most of the time or all of the time": "yes",
                                "(0) Never or rarely": "no"}),
                "fast_food": PromptSpecs(
                    question="Did you eat from a fast food restaurant in the past seven days?",
                    answer_prefix="",
                    answer_map={1.0: "yes",
                                2.0: "yes",
                                3.0: "yes",
                                4.0: "yes",
                                5.0: "yes",
                                6.0: "yes",
                                7.0: "yes",
                                8.0: "yes",
                                9.0: "yes",
                                10.0: "yes",
                                11.0: "yes",
                                12.0: "yes",
                                13.0: "yes",
                                14.0: "yes",
                                15.0: "yes",
                                16.0: "yes",
                                18.0: "yes",
                                20.0: "yes",
                                21.0: "yes",
                                22.0: "yes",
                                23.0: "yes",
                                28.0: "yes",
                                30.0: "yes",
                                35.0: "yes",
                                "(99) 99 or more times": "yes",
                                0.0: "no"}),
                "hours_of_tv": PromptSpecs(
                    question="Did you watch TV (including VHS, DVDs or music videos) in the past seven days?",
                    answer_prefix="",
                    answer_map={1.0: "yes",
                                2.0: "yes",
                                3.0: "yes",
                                4.0: "yes",
                                5.0: "yes",
                                6.0: "yes",
                                7.0: "yes",
                                8.0: "yes",
                                9.0: "yes",
                                10.0: "yes",
                                11.0: "yes",
                                12.0: "yes",
                                13.0: "yes",
                                14.0: "yes",
                                15.0: "yes",
                                16.0: "yes",
                                17.0: "yes",
                                18.0: "yes",
                                19.0: "yes",
                                20.0: "yes",
                                21.0: "yes",
                                22.0: "yes",
                                23.0: "yes",
                                24.0: "yes",
                                25.0: "yes",
                                26.0: "yes",
                                27.0: "yes",
                                28.0: "yes",
                                30.0: "yes",
                                32.0: "yes",
                                34.0: "yes",
                                35.0: "yes",
                                36.0: "yes",
                                37.0: "yes",
                                38.0: "yes",
                                40.0: "yes",
                                42.0: "yes",
                                43.0: "yes",
                                45.0: "yes",
                                48.0: "yes",
                                49.0: "yes",
                                50.0: "yes",
                                52.0: "yes",
                                54.0: "yes",
                                56.0: "yes",
                                60.0: "yes",
                                63.0: "yes",
                                70.0: "yes",
                                72.0: "yes",
                                75.0: "yes",
                                80.0: "yes",
                                84.0: "yes",
                                98.0: "yes",
                                100.0: "yes",
                                110.0: "yes",
                                121.0: "yes",
                                140.0: "yes",
                                150.0: "yes",
                                0.0: "no"}),
                "individual_sports": PromptSpecs(
                    question="In the past seven days, did you participate in individual sports such as running, wrestling, swimming, cross-country skiing, cycle racing, or martial arts?",
                    answer_prefix="",
                    answer_map={1.0: "yes",
                                2.0: "yes",
                                3.0: "yes",
                                4.0: "yes",
                                5.0: "yes",
                                6.0: "yes",
                                "(7) 7 or more times": "yes",
                                0.0: "no",}),
                "smoked_cigarette": PromptSpecs(
                    question="Have you ever smoked an entire cigarette?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "yes",
                                "(0) No": "no",}),
                "physical_child_abuse": PromptSpecs(
                    question="Before your 18th birthday, did a parent or adult caregiver hit you with a fist, kick you, or throw you down on the floor, into a wall, or down stairs?",
                    answer_prefix="",
                    answer_map={"(1) One time": "yes",
                                "(2) Two times": "yes",
                                "(3) Three to five times": "yes",
                                "(4) Six to ten times": "yes",
                                "(5) More than ten times": "yes",
                                "(6) This has never happened": "no"}),
                "age_of_first_drink": PromptSpecs(
                    question="How old were you when you had your first drink?",
                    answer_prefix="I was",
                    answer_map={5.0: "a child",
                                6.0: "a child",
                                7.0: "a child",
                                8.0: "a child",
                                9.0: "a child",
                                10.0: "a child",
                                11.0: "a child",
                                12.0: "a child",
                                13.0: "in my teens",
                                14.0: "in my teens",
                                15.0: "in my teens",
                                16.0: "in my teens",
                                17.0: "in my teens",
                                18.0: "in my teens",
                                19.0: "in my teens",
                                20.0: "an adult",
                                21.0: "an adult",
                                22.0: "an adult",
                                23.0: "an adult",
                                24.0: "an adult",
                                25.0: "an adult",
                                26.0: "an adult",
                                27.0: "an adult",
                                28.0: "an adult",
                                29.0: "an adult",
                                30.0: "an adult",
                                33.0: "an adult",}),
                "car_accidents": PromptSpecs(
                    question="In the past 12 months, were you involved in a motor vehicle accident?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "yes",
                                "(0) No": "no",}),
                "drinking": PromptSpecs(
                    question="Have you had a drink of beer, wine, or liquor more than two or three times?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "yes",
                                "(0) No": "no",}),
                "prayer_in_private": PromptSpecs(
                    question="How often do you pray privately, that is, when you're alone in places other than a church, synagogue, temple, mosque, or religious assembly?",
                    answer_prefix="I",
                    answer_map={"(0) Never": "never pray",
                                "(1) Less than once a month": "sometimes pray",
                                "(2) Once a month": "sometimes pray",
                                "(3) A few times a month": "sometimes pray",
                                "(4) Once a week": "often pray",
                                "(5) A few times a week": "often pray",
                                "(6) Once a day": "pray very often",
                                "(7) More than once a day": "pray very often"})
        }                

if __name__ == "__main__":
    ds = AddHealthDataset()
    backstories = ds.get_backstories_all_demos()
    for backstory in backstories:
        print(f"{backstory[0]}\n\n{backstory[1]}\n\n")
    prompts = ds.get_prompts_sample()
    for prompt in prompts:
        print(f"{prompt}\n\n")
    # Uncomment this to see a sample of your prompts
    # First prompt for each DV
    # prompts = []
    # for dv_name in ds.dvs.keys():
    #      dv_prompts = ds.prompts[dv_name]
    #      for row_idx in dv_prompts.keys():
    #          prompts.append(dv_prompts[row_idx])
    #      print(dv_prompts[list(dv_prompts.keys())[0]])
    #      print()
