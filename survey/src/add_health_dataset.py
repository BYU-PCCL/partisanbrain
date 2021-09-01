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
        grad_school_values = ["Some graduate school",
                              "Completed a master's degree",
                              "Some graduate training beyond a master's degree",
                              "Completed a doctoral degree",
                              "Some post baccalaureate professional education (e.g., law school, med school, nurse)",
                              "Completed post baccalaureate professional education (e.g., law school, med school, nurse)"]
        if row["education"] in grad_school_values:
            backstory.append("I went to grad school.")
        elif row["education"] == "Completed college (bachelor's degree)":
            backstory.append("I completed college.")
        elif row["education"] == "Some College":
            backstory.append("I've completed some college.")
        else:
            backstory.append("I didn't go to college.")

        # Ideology
        if "conservative" in row['ideology']:
            ideology = "conservative"
        elif "liberal" in row['ideology']:
            ideology = "liberal"
        else:
            ideology = "moderate"
        backstory.append(("In terms of political ideology, "
                          "I'd consider myself "
                          f"to be {ideology}."))

        # Income
        if "to " in row["income"]:
            income = row['income'][4:]
            low, high = income.split(" to ")
            backstory.append(("My annual family income is "
                              f"between {low} and {high}."))
        elif row["income"] == "Less than $5,000":
            backstory.append("My annual family income is less than $5000.")
        else:
            backstory.append("My annual family income is more than $150,000.")

        # Religiosity
        religion = row['religion'][4:]

        if "Non" in religion:
            backstory.append(("I don't identify with any religion in particular."))
        else:
            if "Protestant" in religion:
                religion = "Protestant"
            elif "Other Christian" in row["religion"]:
                religion = "Christian"
            backstory.append(f"In terms of religion I am {religion}.")

        # Race/Ethnicity
        race_ethnicity = row['race_ethnicity']
        if "White" in race_ethnicity:
            race_ethnicity = "white"
        elif "Black" in race_ethnicity:
            race_ethnicity = "black"
        elif "Indian" in race_ethnicity:
            race_ethnicity = "American Indian"
        else:
            race_ethnicity = "Asian or Pacific Islander"
        backstory.append(f"My race is {race_ethnicity}.")

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
                    answer_map={"(1) Yes": "Yes",
                                "(0) No": "No"}),
                "arrested": PromptSpecs(
                    question="Have you ever been arrested?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "Yes",
                                "(0) No": "No"}),
                "physical_fight": PromptSpecs(
                    question="In the past 12 months, did you get into a serious physical fight?",
                    answer_prefix="",
                    answer_map={"(1) 1 or 2 times": "Yes",
                                "(2) 3 or 4 times": "Yes",
                                "(3) 5 or more times": "Yes",
                                "(0) Never": "No"}),
                "convicted_of_charges": PromptSpecs(
                    question="Have you ever been convicted of or pled guilty to any charges other than a minor traffic violation?",
                    answer_prefix="",
                    answer_map={"(1) Once": "Yes",
                                "(2) More than once": "Yes",
                                "(0) No": "No"}),
                "sell_drugs": PromptSpecs(
                    question="In the past 12 months, did you sell marijuana or other drugs?",
                    answer_prefix="",
                    answer_map={"(1) 1 or 2 times": "Yes",
                                "(2) 3 or 4 times": "Yes",
                                "(3) 5 or more times": "Yes",
                                "(0) Never": "No"}),
                "counseling": PromptSpecs(
                    question="In the past 12 months have you received psychological or emotional counseling?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "Yes",
                                "(0) No": "No"}),
                "sadness_family": PromptSpecs(
                    question="In the past seven days, have you felt as though you can't shake off the blues, even with help from your family and your friends?",
                    answer_prefix="",
                    answer_map={"(1) Sometimes": "Yes",
                                "(2) A lot of the time": "Yes",
                                "(3) Most of the time or all of the time": "Yes",
                                "(0) Never or rarely": "No"}),
                "worrying": PromptSpecs(
                    question="Do you worry about things?",
                    answer_prefix="",
                    answer_map={"(1) Strongly agree": "Yes",
                                "(2) Agree": "Yes",
                                "(3) Neither agree nor disagree": "No",
                                "(4) Disagree": "No",
                                "(5) Strongly disagree": "No"}),
                "suicide": PromptSpecs(
                    question="Have you actually attempted suicide in the past 12 months?",
                    answer_prefix="",
                    answer_map={"(1) Once": "Yes",
                                "(2) Twice": "Yes",
                                "(3) 3 or 4 times": "Yes",
                                "(4) 5 or more times": "Yes",
                                "(0) None": "No"}),  
                "optimism": PromptSpecs(
                    question="Do you expect more good things to happen to you than bad?",
                    answer_prefix="",
                    answer_map={"(1) Strongly agree": "Yes",
                                "(2) Agree": "Yes",
                                "(3) Neither agree nor disagree": "No",
                                "(4) Disagree": "No",
                                "(5) Strongly disagree": "No"}),
                "happiness": PromptSpecs(
                    question="Did you feel happy in the past 7 days?",
                    answer_prefix="",
                    answer_map={"(1) Sometimes": "Yes",
                                "(2) A lot of the time": "Yes",
                                "(3) Most of the time or all of the time": "Yes",
                                "(0) Never or rarely": "No"}),
                "fast_food": PromptSpecs(
                    question="Did you eat from a fast food restaurant in the past seven days?",
                    answer_prefix="",
                    answer_map={1.0: "Yes",
                                2.0: "Yes",
                                3.0: "Yes",
                                4.0: "Yes",
                                5.0: "Yes",
                                6.0: "Yes",
                                7.0: "Yes",
                                8.0: "Yes",
                                9.0: "Yes",
                                10.0: "Yes",
                                11.0: "Yes",
                                12.0: "Yes",
                                13.0: "Yes",
                                14.0: "Yes",
                                15.0: "Yes",
                                16.0: "Yes",
                                18.0: "Yes",
                                20.0: "Yes",
                                21.0: "Yes",
                                22.0: "Yes",
                                23.0: "Yes",
                                28.0: "Yes",
                                30.0: "Yes",
                                35.0: "Yes",
                                "(99) 99 or more times": "Yes",
                                0.0: "No"}),
                "hours_of_tv": PromptSpecs(
                    question="Did you watch TV (including VHS, DVDs or music videos) in the past seven days?",
                    answer_prefix="",
                    answer_map={1.0: "Yes",
                                2.0: "Yes",
                                3.0: "Yes",
                                4.0: "Yes",
                                5.0: "Yes",
                                6.0: "Yes",
                                7.0: "Yes",
                                8.0: "Yes",
                                9.0: "Yes",
                                10.0: "Yes",
                                11.0: "Yes",
                                12.0: "Yes",
                                13.0: "Yes",
                                14.0: "Yes",
                                15.0: "Yes",
                                16.0: "Yes",
                                17.0: "Yes",
                                18.0: "Yes",
                                19.0: "Yes",
                                20.0: "Yes",
                                21.0: "Yes",
                                22.0: "Yes",
                                23.0: "Yes",
                                24.0: "Yes",
                                25.0: "Yes",
                                26.0: "Yes",
                                27.0: "Yes",
                                28.0: "Yes",
                                30.0: "Yes",
                                32.0: "Yes",
                                34.0: "Yes",
                                35.0: "Yes",
                                36.0: "Yes",
                                37.0: "Yes",
                                38.0: "Yes",
                                40.0: "Yes",
                                42.0: "Yes",
                                43.0: "Yes",
                                45.0: "Yes",
                                48.0: "Yes",
                                49.0: "Yes",
                                50.0: "Yes",
                                52.0: "Yes",
                                54.0: "Yes",
                                56.0: "Yes",
                                60.0: "Yes",
                                63.0: "Yes",
                                70.0: "Yes",
                                72.0: "Yes",
                                75.0: "Yes",
                                80.0: "Yes",
                                84.0: "Yes",
                                98.0: "Yes",
                                100.0: "Yes",
                                110.0: "Yes",
                                121.0: "Yes",
                                140.0: "Yes",
                                150.0: "Yes",
                                0.0: "No"}),
                "individual_sports": PromptSpecs(
                    question="In the past seven days, did you participate in individual sports such as running, wrestling, swimming, cross-country skiing, cycle racing, or martial arts?",
                    answer_prefix="",
                    answer_map={1.0: "Yes",
                                2.0: "Yes",
                                3.0: "Yes",
                                4.0: "Yes",
                                5.0: "Yes",
                                6.0: "Yes",
                                "(7) 7 or more times": "Yes",
                                0.0: "No",}),
                "smoked_cigarette": PromptSpecs(
                    question="Have you ever smoked an entire cigarette?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "Yes",
                                "(0) No": "No",}),
                "physical_child_abuse": PromptSpecs(
                    question="Before your 18th birthday, did a parent or adult caregiver hit you with a fist, kick you, or throw you down on the floor, into a wall, or down stairs?",
                    answer_prefix="",
                    answer_map={"(1) One time": "Yes",
                                "(2) Two times": "Yes",
                                "(3) Three to five times": "Yes",
                                "(4) Six to ten times": "Yes",
                                "(5) More than ten times": "Yes",
                                "(6) This has never happened": "No"}),
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
                    question="In the past 12 months, were you involved in a moror vehicle accident?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "Yes",
                                "(0) No": "No",}),
                "drinking": PromptSpecs(
                    question="Have you had a drink of beer, wine, or liquor more than two or three times?",
                    answer_prefix="",
                    answer_map={"(1) Yes": "Yes",
                                "(0) No": "No",}),
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
    # Uncomment this to see a sample of your prompts
    # First prompt for each DV
    prompts = []
    for dv_name in ds.dvs.keys():
         dv_prompts = ds.prompts[dv_name]
         for row_idx in dv_prompts.keys():
             prompts.append(dv_prompts[row_idx])
         print(dv_prompts[list(dv_prompts.keys())[0]])
         print()
