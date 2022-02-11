"""@author: Kyle Rogers"""

import numpy as np
from ..dataset_factory import DatasetFactory
from ..surveys.anes_survey import AnesSurvey
from pdb import set_trace as breakpoint
from .. import constants as k
from .anes_nan_dicts import dv_nan_dicts, demo_nan_dicts
from .anes_rephrasing_dicts import demo_rephrasing_dicts, dv_rephrasing_dicts


class AnesFactory(DatasetFactory):
    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj, sample_seed=sample_seed, n=n)

    def modify_data(self, df):
        """Replace unacceptable values like 'refused' to np.nan"""

        # Map bad values to np.nan
        for demo, demo_nan_dict in demo_nan_dicts.items():
            df[demo] = df[demo].replace(demo_nan_dict)

        for dv, dv_nan_dict in dv_nan_dicts.items():
            df[dv] = df[dv].replace(dv_nan_dict)

        # Only drop invalid demographics, as we'll drop invalid DV values later
        df.dropna(subset=k.DEMOGRAPHIC_COLNAMES, inplace=True)

        return df

    def get_templates(self):
        # Demographics
        # add token sets as dicts see example (map Kept the same to keep, kept, etc) -- for protecting_environment_spending
        party_dict = demo_rephrasing_dicts["party"]
        education_dict = demo_rephrasing_dicts["education"]
        ideology_dict = demo_rephrasing_dicts["ideology"]
        income_dict = demo_rephrasing_dicts["income"]
        religion_dict = demo_rephrasing_dicts["religion"]
        race_ethnicity_dict = demo_rephrasing_dicts["race_ethnicity"]
        marital_status_dict = demo_rephrasing_dicts["marital_status"]
        rephrasing = dv_rephrasing_dicts

        social_security_tokens = []
        social_security_mc_tokens = []
        vote_2016_dict = []
        vote_2016_mc_dict = []

        def make_backstory1(row):
            # create a generic backstory using values in the demographics
            return (
                f"I am {int(row['age'])} years old. I am {row['gender'].lower()}. "
                f"Politically speaking, I am {party_dict[row['party']]}. "
                f"Ideologically, I am {ideology_dict[row['ideology']]}. I {education_dict[row['education']]}. "
                f"My salary is {income_dict[row['income']]}. "
                f"{religion_dict[row['religion']]}. "
                f"I am {race_ethnicity_dict[row['race_ethnicity']]}. I am from the {row['region']}. "
                f"I {marital_status_dict[row['marital_status']]}."
            )

        def make_backstory2(row):
            return (
                f"Age: {int(row['age'])}, Gender: {row['gender']}, Political Affiliation: {row['party']}, "
                f"Education: {row['education']}, Ideology: {row['ideology']}, Total Income: {row['income']}, Religion: {row['religion']}, "
                f"Race/Ethnicity: {row['race_ethnicity']}, Region: {row['region']}, Marital Status: {row['marital_status']}"
            )

        def make_backstory3(row):
            return (
                f"Q: What is your age?\nA: {int(row['age'])}\n\nQ: What is your gender?\nA: {row['gender']}\n\n"
                f"Q: What is your political affiliation?\nA: {row['party']}\n\nQ: What is your education?\nA: {row['education']}\n\n"
                f"Q: What is your ideology?\nA: {row['ideology']}\n\nQ: What is your income?\nA: {row['income']}\n\n"
                f"Q: What is your religion?\nA: {row['religion']}\n\n"
                f"Q: What is your race/ethnicity?\nA: {row['race_ethnicity']}\n\nQ: What region of the country are you from?\nA: {row['region']}\n\n"
                f"Q: What is your marital status?\nA: {row['marital_status']}"
            )

        def make_backstory4(row):
            return (
                f"Question 1: What is your age?\n\nAnswer 1: {row['age']}\n\nQuestion 2: What is your gender?\n\nAnswer 2: {row['gender']}\n\n"
                f"Question 3: What is your political affiliation?\n\nAnswer 3: {row['party']}\n\nQuestion 4: What is your education?\n\nAnswer 4: {row['education']}\n\n"
                f"Question 5: What is your ideology?\n\nAnswer 5: {row['ideology']}\n\nQuestion 6: What is your income?\n\nAnswer 6: {row['income']}\n\n"
                f"Question 7: What is your religion?\n\nAnswer 7: {row['religion']}\n\n"
                f"Question 8: What is your race/ethnicity?\n\nAnswer 8: {row['race_ethnicity']}\n\nQuestion 9: What region of the country are you from?\n\nAnswer 9: {row['region']}\n\n"
                f"Question 10: What is your marital status?\n\nAnswer 10: {row['marital_status']}"
            )

        def make_backstory5(row):
            return (
                f"Person 1: What is your age?\nPerson 2: {row['age']}\n\nPerson 1: What is your gender?\nPerson 2: {row['gender']}\n\n"
                f"Person 1: What is your political affiliation?\nPerson 2: {row['party']}\n\nPerson 1: What is your education?\nPerson 2: {row['education']}\n\n"
                f"Person 1: What is your ideology?\nPerson 2: {row['ideology']}\n\nPerson 1: What is your income?\nPerson 2: {row['income']}\n\n"
                f"Person 1: What is your religion?\nPerson 2: {row['religion']}\n\n"
                f"Person 1: What is your race/ethnicity?\nPerson 2: {row['race_ethnicity']}\n\nPerson 1: What region of the country are you from?\nPerson 2: {row['region']}\n\n"
                f"Person 1: What is your marital status?\nPerson 2: {row['marital_status']}"
            )

        def make_backstory6(row):
            return (
                f"Q: What is your age? (in years)\n"
                f"A: {int(row['age'])}\n\n"
                f"Q: What is your gender? (Male, Female)\nA: {row['gender']}\n\n"
                f"Q: What is your political affiliation? "
                f"(Democratic Party, Republican Party, Inapplicable, None or "
                f"'independent')\nA: {row['party']}\n\n"
                f"Q: What is your education level? "
                f"(High school, Professional school, Bachelor's degree, "
                f"Master's degree, Associate's degree, Some college, "
                f"Less than high school credential)\nA: {row['education']}\n\n"
                f"Q: What is your ideology? (Slightly conservative, Liberal, "
                f"Conservative, Moderate, Slightly liberal, Extremely liberal, "
                f"Extremely conservative)\nA: {row['ideology']}\n\n"
                f"Q: What is your income? (Under $9,999, $10,000-14,999, "
                f"$15,000-19,999, $20,000-24,999, $25,000-29,999, $30,000-34,999, "
                f"$35,000-39,999, $40,000-44,999, $45,000-49,999, $50,000-59,999, "
                f"$60,000-64,999, $65,000-69,999, $70,000-74,999, $75,000-79,999, "
                f"$80,000-89,999, $90,000-99,999, $100,000-109,999, "
                f"$110,000-124,999, $125,000-149,999, $150,000-174,999, "
                f"$175,000-249,999, $250,00 or more')\nA: {row['income']}\n\n"
                f"Q: What is your religion? ('Undifferentiated Protestant, "
                f"Not religious, Mainline Protestant, Roman Catholic, "
                f"Other religion, Other Christian, Evangelical Protestant, "
                f"Jewish')\nA: {row['religion']}\n\n"
                f"Q: What is your race/ethnicity? ('White, non-Hispanic, Black, "
                f"non-Hispanic, Asian or Native Hawaiian/other Pacific Islander, "
                f"non-Hispanic alone, Hispanic, Multiple races, non-Hispanic, "
                f"Native American/Alaska Native or other race, non-Hispanic "
                f"alone')\nA: {row['race_ethnicity']}\n\n"
                f"Q: What region of the country are you from? (Northeast, Midwest, "
                f"South, West)\nA: {row['region']}\n\n"
                f"Q: What is your marital status? (Married: spouse present, "
                f"Never married, Divorced, Married: spouse absent "
                f"{{VOL - video/phone only}}, Widowed, Separated)"
                f"\nA: {row['marital_status']}"
            )

        # "vote_2016",

        return {
            "gay_marriage": {
                "surveyq_exact": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: What is your position on gay "
                        "marriage? Should "
                        "gay and lesbian couples be allowed to enter "
                        "legal marriages, civil unions, "
                        "or neither?\nAnswer 11: Gay and lesbian couples "
                        "should be allowed to enter"
                    ),
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: If asked whether gay people should "
                        "be allowed to get married, do you think they should "
                        "be able to form legal marriages, civil unions, or "
                        "neither?\nAnswer 11: Gay people should be allowed "
                        "to form"
                    ),
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked whether gay people should be allowed to "
                        "enter into either legal marriages, civil unions, "
                        "or neither, I would say they should be able to "
                        "enter into"
                    ),
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "person living in the United States in 2020."
                        "Please read "
                        "it and infer their opinion on gay marriage.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on gay marriage, and what kind of "
                        "unions gay people should be allowed to enter into "
                        "(legal marriages, civil unions, neither):"
                    ),
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "gay_marriage",
                            rephrasing["gay_marriage"]["colloquial"]["rephrasings"],
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "person living in the United States in 2020. "
                        "Please read "
                        "it and infer their opinion on gay marriage.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on gay marriage, and what kind of "
                        "unions gay people should be allowed to enter into "
                        "(legal marriages, civil unions, neither):"
                    ),
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of people "
                        "living in the United States."
                        "Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should gay couples be allowed to enter into "
                        "legal marriages or civil unions or neither?\nA:"
                    ),
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Do you think that gay people should "
                    "be allowed to enter into legal marriages, or civil unions, "
                    "or neither?\nPerson 2:",
                    rephrasing["gay_marriage"]["colloquial"]["tokens"],
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should gay people be allowed to "
                        "enter into legal marriages, or civil unions, or "
                        "neither?"
                        "\nA: Legal marriages\nB: Civil unions\nC: Neither\n\nAnswer 11:"
                    ),
                    rephrasing["gay_marriage"]["mc"]["tokens"],
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "gay_marriage",
                        rephrasing["gay_marriage"]["mc"]["rephrasings"],
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        +"\n\nQuestion 11: Should gay people be allowed to "
                        "enter into legal marriages, or civil unions, or "
                        "neither?"
                        "\nA: Legal marriages\nB: Civil unions\nC: Neither\n\nAnswer 11:"
                    ),
                    rephrasing["gay_marriage"]["mc"]["tokens"],
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "gay_marriage",
                        rephrasing["gay_marriage"]["mc"]["rephrasings"],
                        "mc",
                        5,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        +"\n\nQuestion 11: Should gay people be allowed to "
                        "enter into legal marriages, or civil unions, or "
                        "neither?"
                        "\nA: Legal marriages\nB: Civil unions\nC: Neither\n\nAnswer 11:"
                    ),
                    rephrasing["gay_marriage"]["mc"]["tokens"],
                ),
            },
            "trump_handling_relations": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "trump_hanlding_immigration": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "willing_military_force": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "restless_sleep": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "have_health_insurance": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "attn_to_politics": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "feel_voting_is_duty": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "govt_run_by_who": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (
                        " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
                    ),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "social_security_spending",
                            "task",
                            3,
                            "\n\n-----------------------\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                        "vote:"
                    ),
                    social_security_tokens,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n"
                        + make_backstory3(row)
                        + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: make_backstory5(row)
                    + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        2,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots(
                        "social_security_spending",
                        "mc",
                        7,
                        "\n\n-----------------------\n\n",
                    )
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
            },
            "vote_2016": {
                "surveyq_exact": (
                    lambda row: make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Four years ago, in 2016, Hillary Clinton ran "
                        "on the Democratic ticket against Donald Trump for the "
                        "Republicans. Which one did you vote for?\nAnswer 11:"
                    ),
                    vote_2016_dict,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        make_backstory4(row)
                        + "\n\nQuestion 11: Do you remember the presidential "
                        "election of 2016? Did you vote for Hillary Clinton "
                        "or Donald Trump?\nAnswer 11:"
                    ),
                    vote_2016_dict,
                ),
                "fpbs": (
                    lambda row: make_backstory1(row)
                    + (" In the 2016 presidential election, I voted for"),
                    vote_2016_dict,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer who that person voted for.\n\n"
                        + make_backstory2(row)
                        + ", 2016 Presidential Election "
                        "Vote:"
                    ),
                    vote_2016_dict,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots(
                            "vote_2016",
                            "task",
                            rephrasing["gay_marriage"]["colloquial"]["rephrasings"],
                            3,
                            sep="\n\n",
                        )
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer who that person voted for.\n\n"
                        + make_backstory2(row)
                        + " 2016 presidential election "
                        "vote:"
                    ),
                    vote_2016_dict,
                ),
                "anes_description": (
                    lambda row: (
                        "The American National Election Studies 2020 "
                        "Time Series Study (ANES 2020) "
                        "is a nationally representative survey of voters in "
                        "American Elections. Below are examples of respondents "
                        "answering various questions. Please complete what you "
                        "would guess the right answers to those questions to be."
                        "\n\n" + make_backstory3(row) + "\n\nQ: Who did you "
                        "vote for in the 2016 presidential election?\nA:"
                    ),
                    vote_2016_dict,
                ),
                "conversation": (
                    lambda row: make_backstory5(row) + "\n\nPerson 1: Who did "
                    "you vote for in the 2016 presidential election?\nPerson 2:",
                    vote_2016_dict,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + make_backstory4(row)
                        + "\n\nQuestion 11: Who did "
                        "you vote for in the 2016 presidential election?\nA: "
                        "Hillary Clinton\nB: Donald Trump\n\nAnswer 11:"
                    ),
                    vote_2016_mc_dict,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots("vote_2016", "mc", 2, "\n\n")
                    + "SURVEY_RESPONSE\n\n"
                    + make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Who did "
                        "you vote for in the 2016 presidential election?\nA: "
                        "Hillary Clinton\nB: Donald Trump\n\nAnswer 11:"
                    ),
                    vote_2016_dict,
                ),
                "explicit_enumeration": (
                    lambda row: make_backstory6(row)
                    + (
                        "\n\nQ: Who did you"
                        f"vote for in the 2016 presidential election? (Donald "
                        f"Trump, Hillary Clinton)\nA:"
                    ),
                    vote_2016_dict,
                ),
            },
            # What about Social Security? Should federal spending on Social Security be increased, decreased, or kept the same?
            # "social_security_spending": {
            #     "surveyq_exact": (
            #         lambda row: make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: What about Social Security? "
            #             "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         social_security_tokens,
            #     ),
            #     "surveyq_rephrase": (
            #         lambda row: (
            #             make_backstory4(row)
            #             + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         social_security_tokens,
            #     ),
            #     "fpbs": (
            #         lambda row: make_backstory1(row)
            #         + (
            #             " If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"
            #         ),
            #         social_security_tokens,
            #     ),
            #     "task": (
            #         lambda row: (
            #             "TASK: The following is a description of a "
            #             "voter in the 2016 presidential election. Please read "
            #             "it and infer their opinion on federal Social Security spending.\n\n"
            #             + make_backstory2(row)
            #             + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
            #         ),
            #         social_security_tokens,
            #     ),
            #     "task_3shot": (
            #         lambda row: (
            #             self.get_shots(
            #                 "social_security_spending",
            #                 "task",
            #                 3,
            #                 "\n\n-----------------------\n\n",
            #             )
            #             + "TASK: The following is a description of a "
            #             "voter in the 2016 presidential election. Please read "
            #             "it and infer their opinion on federal Social Security spending.\n\n"
            #             + make_backstory2(row)
            #             + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
            #             "vote:"
            #         ),
            #         social_security_tokens,
            #     ),
            #     "anes_description": (
            #         lambda row: (
            #             "The American National Election Studies 2020 "
            #             "Time Series Study (ANES 2020) "
            #             "is a nationally representative survey of voters in "
            #             "American Elections. Below are examples of respondents "
            #             "answering various questions. Please complete what you "
            #             "would guess the right answers to those questions to be."
            #             "\n\n"
            #             + make_backstory3(row)
            #             + "\n\nQ: Should federal spending "
            #             "on Social Security be increased, decreased, or kept the same?\nA:"
            #         ),
            #         social_security_tokens,
            #     ),
            #     "conversation": (
            #         lambda row: make_backstory5(row)
            #         + "\n\nPerson 1: Should federal spending "
            #         "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
            #         social_security_tokens,
            #     ),
            #     "mc": (
            #         lambda row: (
            #             "SURVEY_RESPONSE\n\n"
            #             + make_backstory4(row)
            #             + "\n\nQuestion 11: Should federal spending "
            #             "on Social Security be increased, decreased, or kept the same?"
            #             "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
            #         ),
            #         social_security_mc_tokens,
            #     ),
            #     "mc_2shot": (
            #         lambda row: self.get_shots(
            #             "social_security_spending",
            #             "mc",
            #             2,
            #             "\n\n-----------------------\n\n",
            #         )
            #         + "SURVEY_RESPONSE\n\n"
            #         + make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Should federal spending "
            #             "on Social Security be increased, decreased, or kept the same?"
            #             "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
            #         ),
            #         social_security_mc_tokens,
            #     ),
            #     "mc_5shot": (
            #         lambda row: self.get_shots(
            #             "social_security_spending",
            #             "mc",
            #             7,
            #             "\n\n-----------------------\n\n",
            #         )
            #         + "SURVEY_RESPONSE\n\n"
            #         + make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Should federal spending "
            #             "on Social Security be increased, decreased, or kept the same?"
            #             "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
            #         ),
            #         social_security_mc_tokens,
            #     ),
            # },
            # # What about protecting the environment? Should federal spending on protecting the environment be increased, decreased, or kept the same?
            # "protecting_environment_spending": {
            #     "surveyq_exact": (
            #         lambda row: make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: What about protecting the environment? "
            #             "Should federal spending on protecting the enivironment be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         protecting_environment_tokens,
            #     ),
            #     "surveyq_rephrase": (
            #         lambda row: (
            #             make_backstory4(row)
            #             + "\n\nQuestion 11: Do you think federal spending on protecting the environment should be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         protecting_environment_tokens,
            #     ),
            #     "fpbs": (
            #         lambda row: make_backstory1(row)
            #         + (
            #             " If asked if spending on protecting the environment should be increased, decreased, or kept the same, I would say that it should be"
            #         ),
            #         protecting_environment_tokens,
            #     ),
            #     "task": (
            #         lambda row: (
            #             "TASK: The following is a description of a "
            #             "voter in the 2016 presidential election. Please read "
            #             "it and infer their opinion on federal spending on protecting the environment.\n\n"
            #             + make_backstory2(row)
            #             + ", Opinion on spending on protecting the environment (increased, decreased, or kept the same):"
            #         ),
            #         protecting_environment_tokens,
            #     ),
            #     "task_3shot": (
            #         lambda row: (
            #             self.get_shots(
            #                 "protecting_environment_spending",
            #                 "task",
            #                 3,
            #                 "\n\n-----------------------\n\n",
            #             )
            #             + "TASK: The following is a description of a "
            #             "voter in the 2016 presidential election. Please read "
            #             "it and infer their opinion on federal spending on protecting the environment.\n\n"
            #             + make_backstory2(row)
            #             + ", Opinion on spending on protecting the environment (increased, decreased, or kept the same):"
            #             "vote:"
            #         ),
            #         protecting_environment_tokens,
            #     ),
            #     "anes_description": (
            #         lambda row: (
            #             "The American National Election Studies 2020 "
            #             "Time Series Study (ANES 2020) "
            #             "is a nationally representative survey of voters in "
            #             "American Elections. Below are examples of respondents "
            #             "answering various questions. Please complete what you "
            #             "would guess the right answers to those questions to be."
            #             "\n\n"
            #             + make_backstory3(row)
            #             + "\n\nQ: Should federal spending "
            #             "on protecting the environment be increased, decreased, or kept the same?\nA:"
            #         ),
            #         protecting_environment_tokens,
            #     ),
            #     "conversation": (
            #         lambda row: make_backstory5(row)
            #         + "\n\nPerson 1: Should federal spending "
            #         "on protecting the environment be increased, decreased, or kept the same?\nPerson 2:",
            #         protecting_environment_tokens,
            #     ),
            #     "mc": (
            #         lambda row: (
            #             "SURVEY_RESPONSE\n\n"
            #             + make_backstory4(row)
            #             + "\n\nQuestion 11: Should federal spending "
            #             "on protecting the environment be increased, decreased, or kept the same?"
            #             "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
            #         ),
            #         protecting_environment_mc_tokens,
            #     ),
            #     "mc_2shot": (
            #         lambda row: self.get_shots(
            #             "protecting_environment_spending",
            #             "mc",
            #             2,
            #             "\n\n-----------------------\n\n",
            #         )
            #         + "SURVEY_RESPONSE\n\n"
            #         + make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Should federal spending "
            #             "on protecting the environment be increased, decreased, or kept the same?"
            #             "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
            #         ),
            #         protecting_environment_mc_tokens,
            #     ),
            #     "mc_5shot": (
            #         lambda row: self.get_shots(
            #             "protecting_environment_spending",
            #             "mc",
            #             5,
            #             "\n\n-----------------------\n\n",
            #         )
            #         + "SURVEY_RESPONSE\n\n"
            #         + make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Should federal spending "
            #             "on protecting the environment be increased, decreased, or kept the same?"
            #             "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
            #         ),
            #         protecting_environment_mc_tokens,
            #     ),
            # },
            # "rising_temp_action": {
            #     "surveyq_exact": (
            #         lambda row: make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Do you think the federal government should be doing more about rising temperatures, should be doing less, or should be doing the same? "
            #             "\nAnswer 11:"
            #         ),
            #         rising_temp_action_tokens,
            #     ),
            #     "surveyq_rephrase": (
            #         lambda row: (
            #             make_backstory4(row)
            #             + "\n\nQuestion 11: Do you think federal government should be doing more about rising temperatures, should be doing less, or should be doing the same? \nAnswer 11: I think the federal government should be doing"
            #         ),
            #         rising_temp_action_tokens,
            #     ),
            #     "fpbs": (
            #         lambda row: make_backstory1(row)
            #         + (
            #             " If asked if the federal government should be doing more, less, or the same about rising temperatures, I would say that it should be doing"
            #         ),
            #         rising_temp_action_tokens,
            #     ),
            #     "task": (
            #         lambda row: (
            #             "TASK: The following is a description of a "
            #             "voter in the 2016 presidential election. Please read "
            #             "it and infer their opinion on federal spending on protecting the environment.\n\n"
            #             + make_backstory2(row)
            #             + ", Opinion on spending on federal government action about rising temperatures (more, less, or the same):"
            #         ),
            #         rising_temp_action_tokens,
            #     ),
            #     "task_3shot": (
            #         lambda row: (
            #             self.get_shots(
            #                 "rising_temp_action",
            #                 "task",
            #                 3,
            #                 "\n\n-----------------------\n\n",
            #             )
            #             + "TASK: The following is a description of a "
            #             "voter in the 2016 presidential election. Please read "
            #             "it and infer their opinion on federal spending on protecting the environment.\n\n"
            #             + make_backstory2(row)
            #             + ", Opinion on spending on federal government action about rising temperatures (more, less, or the same):"
            #             "vote:"
            #         ),
            #         rising_temp_action_tokens,
            #     ),
            #     "anes_description": (
            #         lambda row: (
            #             "The American National Election Studies 2020 "
            #             "Time Series Study (ANES 2020) "
            #             "is a nationally representative survey of voters in "
            #             "American Elections. Below are examples of respondents "
            #             "answering various questions. Please complete what you "
            #             "would guess the right answers to those questions to be."
            #             "\n\n"
            #             + make_backstory3(row)
            #             + "\n\nQ: Should the federal government "
            #             "be doing more, less, or the same about rising temperatures?\nA:"
            #         ),
            #         rising_temp_action_tokens,
            #     ),
            #     "conversation1": (
            #         lambda row: make_backstory5(row)
            #         + "\n\nPerson 1: Should the federal government "
            #         "be doing more, less, or the same about rising temperature?\nPerson 2: The federal government should be doing",
            #         rising_temp_action_tokens,
            #     ),
            #     "conversation2": (
            #         lambda row: make_backstory5(row)
            #         + "\n\nPerson 1: Should the federal government "
            #         "be doing the same, more, or less about rising temperature?\nPerson 2: The federal government should be doing",
            #         rising_temp_action_tokens,
            #     ),
            #     "mc": (
            #         lambda row: (
            #             "SURVEY_RESPONSE\n\n"
            #             + make_backstory4(row)
            #             + (
            #                 "\n\nQuestion 11: Do you think the federal government should be doing more "
            #                 "about rising temperatures, should be doing less, or is it currently doing the right amount?"
            #                 "\nA: More\nB: Less\nC: Currently doing the right amount\n\nAnswer 11:"
            #             )
            #         ),
            #         rising_temp_action_mc_tokens,
            #     ),
            #     "mc_2shot": (
            #         lambda row: self.get_shots(
            #             "rising_temp_action",
            #             "mc",
            #             2,
            #             "\n\n-----------------------\n\n",
            #             "_mc",
            #         )
            #         + "SURVEY_RESPONSE\n\n"
            #         + make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Do you think the federal government should be doing more "
            #             "about rising temperatures, should be doing less, or is it currently doing the right amount?"
            #             "\nA: More\nB: Less\nC: Currently doing the right amount\n\nAnswer 11:"
            #         ),
            #         rising_temp_action_mc_tokens,
            #     ),
            #     "mc_5shot": (
            #         lambda row: self.get_shots(
            #             "rising_temp_action",
            #             "mc",
            #             5,
            #             "\n\n-----------------------\n\n",
            #             "_mc",
            #         )
            #         + "SURVEY_RESPONSE\n\n"
            #         + make_backstory4(row)
            #         + (
            #             "\n\nQuestion 11: Do you think the federal government should be doing more "
            #             "about rising temperatures, should be doing less, or is it currently doing the right amount?"
            #             "\nA: More\nB: Less\nC: Currently doing the right amount\n\nAnswer 11:"
            #         ),
            #         rising_temp_action_mc_tokens,
            #     ),
            # },
        }


if __name__ == "__main__":
    factory = AnesFactory(AnesSurvey(force_recreate=True), n=500)
    factory.sample_templates(
        factory.survey_obj.df,
        dvs=[
            # "vote_2016",
            "gay_marriage",
            # "social_security_spending",
            # "protecting_environment_spending",
            # "rising_temp_action",
        ],
        playground=True,
    )
    # factory.sample_templates(factory.survey_obj.df, dvs=["social_security_spending"], playground=True)


# OLD KYLE TEMPLATES

# "protecting_environment_spending": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if federal spending on protecting the environment should be increased, decreased or stay the same,\n\n"
#             f" I would say federal spending on protecting the environment should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What about protecting the environment? Should federal spending on protecting the environment be increased, decreased, or kept the same?\n\n"
#             f"A: Federal spending on protecting the environment should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you consider environment protection to be important to you?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
#             f"A:"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should federal spending on protecting the environment should be increased, decreased or stay the same?\n"
#             f"P1: I think it should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on protecting the environment should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Federal spending on protecting the environment should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think federal spending on protecting the environment should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think environmental protection is important.\n\n"
#             f"{make_backstory1(row)} I think federal spending on protecting the environment should be"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person want federal spending on protecting the environment to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely support green reforms?\n"
#             f"2) According to the above backstory, would this person want federal spending on protecting the environment to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should federal spending on protecting the environment be increased, decreased, or kept the same?\nAnswer 11:"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you consider environment protection to be important to you?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on protecting the environment be increased, decreased, or kept the same?\nAnswer 12:"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on protecting the environment be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you consider environment protection to be important to you?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on protecting the environment be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you consider environment protection to be important to you?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
#             f"Answer 1 (Increased, Decreased, Kept the same):"
#         ),
#         protecting_environment_spending_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you consider environment protection to be important to you?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
#             f"Answer 2 (Increased, Decreased, Kept the same):"
#         ),
#         protecting_environment_spending_dict,
#     ),
# },
# # question: Do you think the federal government should be doing more about rising temperatures, should be doing less, or is it currently doing the right amount?
# "rising_temp_action": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked should the federal government be doing more, less, or the same about rising temperatures,\n\n"
#             f" I would say federal government should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: Do you think the federal government should be doing more about rising temperatures, should be doing less, or should be doing the same?\n\n"
#             f"A: The federal government should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should the federal government be doing more, less, or the same about rising temperatures?\n\n"
#             f"ANSWER:"
#         ),
#         rising_temp_action_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you consider global warming to be important to you?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should the federal government be doing more, less, or the same about rising temperatures?\n\n"
#             f"ANSWER:"
#         ),
#         rising_temp_action_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should the federal government be doing more, less, or the same about rising temperatures?\n"
#             f"A: It should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should the federal government be doing more, less, or the same about rising temperatures?\n"
#             f"P1: I think it should be doing"
#         ),
#         {},
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between more, less, or the same, regarding rising temperatures the federal government should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding rising temperatures the federal government should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding rising temperatures, I think the federal government should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think global warming is an important issue.\n\n"
#             f"{make_backstory1(row)} Regarding rising temperatures, I think the federal government should be doing"
#         ),
#         rising_temp_action_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person want the federal government to do more, less, or the same about rising temperatures?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         rising_temp_action_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely think gloabal warming is important?\n"
#             f"2) According to the above backstory, would this person want the federal government to do more, less, or the same about rising temperatures?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         rising_temp_action_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should the federal government to do more, less, or the same about rising temperatures?\nAnswer 11:"
#         ),
#         rising_temp_action_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you think global warming is important?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should the federal government to do more, less, or the same about rising temperatures?\nAnswer 12:"
#         ),
#         rising_temp_action_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should the federal government to do more, less, or the same about rising temperatures?''\n\n"
#             f" -> '''It should do"
#         ),
#         rising_temp_action_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you consider global warming to be important?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should the federal government to do more, less, or the same about rising temperatures?''\n\n"
#             f" -> '''It should do"
#         ),
#         rising_temp_action_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should the federal government to do more, less, or the same about rising temperatures?\n"
#             f"A: More\n"
#             f"B: Less\n"
#             f"C: The same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Should be doing more": ["A", "More"],
#             "Should be doing less": ["B", "Less"],
#             "Is currently doing the right amount": ["C", "the", "about"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you consider global warming to be important?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should the federal government to do more, less, or the same about rising temperatures?\n"
#             f"A: More\n"
#             f"B: Less\n"
#             f"C: The same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Should be doing more": ["A", "More"],
#             "Should be doing less": ["B", "Less"],
#             "Is currently doing the right amount": ["C", "the", "about"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should the federal government to do more, less, or the same about rising temperatures?\n"
#             f"Answer 1 (More, Less, The same):"
#         ),
#         rising_temp_action_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you consider environment protection to be important to you?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
#             f"Answer 2 (Increased, Decreased, Kept the same):"
#         ),
#         rising_temp_action_dict,
#     ),
# },
# # What about dealing with crime? Should federal spending on dealing with crime be increased, decreased, or kept the same?
# "dealing_with_crime_spending": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if federal spending on dealing with crime should be increased, decreased or stay the same,\n\n"
#             f" I would say federal spending on dealing with crime should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What about dealing with crime? Should federal spending on dealing with crime be increased, decreased, or kept the same?\n\n"
#             f"A: Federal spending on dealing with crime should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you consider public safety to be important to you?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
#             f"A:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
#             f"P1: I think it should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on dealing with crime should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Federal spending on dealing with crime should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think federal spending on dealing with crime should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think public safety is important.\n\n"
#             f"{make_backstory1(row)} I think federal spending on dealing with crime should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person want federal spending on dealing with crime to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person want federal spending on dealing with crime to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should federal spending on dealing with crime be increased, decreased, or kept the same?\nAnswer 11:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on dealing with crime be increased, decreased, or kept the same?\nAnswer 12:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on dealing with crime be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on dealing with crime be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
#             f"Answer 1 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
#             f"Answer 2 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
# },
# # Do you approve or disapprove of the way Donald Trump is handling the economy?
# "trump_handling_economy": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if I approve or disapprove of the way Donald Trump is handling the economy,\n\n"
#             f" I would say I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: Do you approve or disapprove of the way Donald Trump is handling the economy?\n\n"
#             f"A: I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Does this person approve or disapprove of the way Donald Trump is handling the economy?\n\n"
#             f"ANSWER: This person"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you want a sandwich?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Does this person approve or disapprove of the way Donald Trump is handling the economy?\n\n"
#             f"ANSWER: This person"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Does this person approve or disapprove of the way Donald Trump is handling the economy?\n"
#             f"A: This person"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
#             f"P1: I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding the way Donald Trump is handling the economy, between approve or disapprove, I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f'{make_backstory1(row)} If someone asked me "Do you approve or disapprove of the way Donald Trump is handling the economy?", I would say "I'
#         ),
#         trump_handling_economy_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} My friend asked if I approve or disapprove of the way Donald Trump is handling the economy, I said I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} My friend asked if I approve or disapprove of the way Donald Trump is handling the economy, I said I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, regarding the way Donald Trump is handling the economy, would this person approve or disapprove?\n\n"
#             f"ANSWERS:\n1) This person would"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, According to the above backstory, regarding the way Donald Trump is handling the economy, would this person approve or disapprove?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you approve or disapprove of the way Donald Trump is handling the economy?\nAnswer 11: I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Do you approve or disapprove of the way Donald Trump is handling the economy?\nAnswer 12: I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Do you approve or disapprove of the way Donald Trump is handling the economy?''\n\n"
#             f" -> '''I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Do you approve or disapprove of the way Donald Trump is handling the economy?''\n\n"
#             f" -> '''I"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
#             f"A: Approve\n"
#             f"B: Disapprove\n"
#             f"Answer 1:"
#         ),
#         {"Approve": ["A"], "Disapprove": ["B"]},
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
#             f"A: Approve\n"
#             f"B: Disapprove\n"
#             f"Answer 2:"
#         ),
#         {"Approve": ["A"], "Disapprove": ["B"]},
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
#             f"Answer 1 (Approve, Disapprove):"
#         ),
#         trump_handling_economy_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
#             f"Answer 2 (Approve, Disapprove):"
#         ),
#         trump_handling_economy_dict,
#     ),
# },
# # Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or don\'t waste very much of it?
# "govt_waste_money": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if people in government waste a lot of the money we pay in taxes, waste some of it, or none of it,\n\n"
#             f" I would say people in the government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
#             f"A: I think people in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
#             f"ANSWER: People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
#             f"ANSWER: People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"A: People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"P1: I think people in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between a lot, some, or none, people in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding tax money, people in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding tax money, I think people in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} Regarding tax money, I think people in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person feel that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
#             f"ANSWERS:\n1) This person would say they waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person feel that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
#             f"ANSWERS:\n1) Yes\n2) This person would say they waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"Answer 11: People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"Answer 12: People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?''\n\n"
#             f" -> '''People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?''\n\n"
#             f" -> '''People in government waste"
#         ),
#         govt_waste_money_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"A: A lot\n"
#             f"B: Some\n"
#             f"C: None\n"
#             f"Answer 1:"
#         ),
#         {
#             "Waste a lot": ["A", "A lot"],
#             "Waste some": ["B", "Some"],
#             "Don't waste very much": ["C", "None"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"A: A lot\n"
#             f"B: Some\n"
#             f"C: None\n"
#             f"Answer 2:"
#         ),
#         {
#             "Waste a lot": ["A", "A lot"],
#             "Waste some": ["B", "Some"],
#             "Don't waste very much": ["C", "None"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"Answer 1 (A lot, Some, None):"
#         ),
#         govt_waste_money_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
#             f"Answer 2 (A lot, Some, None):"
#         ),
#         govt_waste_money_dict,
#     ),
# },
# #
# "social_security_spending": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if federal spending on Social Security be increased, decreased, or kept the same,\n\n"
#             f" I would say federal spending on Social Security should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What about Social Security? Should federal spending on Social Security be increased, decreased, or kept the same?\n\n"
#             f"A: Federal spending on Social Security should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on Social Security be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on Social Security be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
#             f"A:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
#             f"P1: I think it should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on Social Security should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Federal spending on Social Security should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think federal spending on Social Security should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} I think federal spending on Social Security should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person want federal spending on Social Security to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person want federal spending on Social Security to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 12:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on Social Security be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on Social Security be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
#             f"Answer 1 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
#             f"Answer 2 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
# },
# #
# "aid_poor_spending": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if federal spending on aid to the poor be increased, decreased, or kept the same,\n\n"
#             f" I would say federal spending on aid to the poor should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What about aid to the poor? Should federal spending on aid to the poor be increased, decreased, or kept the same?\n\n"
#             f"A: Federal spending on aid to the poor should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
#             f"A:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
#             f"P1: I think it should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on aid to the poor should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Federal spending on aid to the poor should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think federal spending on aid to the poor should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} I think federal spending on aid to the poor should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person want federal spending on aid to the poor to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person want federal spending on aid to the poor to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should federal spending on aid to the poor be increased, decreased, or kept the same?\nAnswer 11:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on aid to the poor be increased, decreased, or kept the same?\nAnswer 12:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on aid to the poor be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on aid to the poor be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
#             f"Answer 1 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
#             f"Answer 2 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
# },
# # What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?
# "state_of_economy": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if the state of the United States economy is very good, good, neither good nor bad, bad, or very bad,\n\n"
#             f" I would say the economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n\n"
#             f"A: The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n\n"
#             f"ANSWER: The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n\n"
#             f"ANSWER: The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n"
#             f"A: The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n"
#             f"P1: I think the economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between very good, good, neither good nor bad, bad or very bad, the United States economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} The United States economy these days is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think the United States economy these days is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} I think the United States economy these days is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person think the United States economy is very good, good, neither good nor bad, bad or very bad?\n\n"
#             f"ANSWERS:\n1) This person would think the economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person think the United States economy is very good, good, neither good nor bad, bad or very bad?\n\n"
#             f"ANSWERS:\n1) Yes\n2) This person would think the economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\nAnswer 11: The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\nAnswer 12: The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?''\n\n"
#             f" -> '''The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?''\n\n"
#             f" -> '''The economy is"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\n"
#             f"A: Very good\n"
#             f"B: Good\n"
#             f"C: Neither good nor bad\n"
#             f"D: Bad\n"
#             f"E: Very bad\n"
#             f"Answer 1:"
#         ),
#         {
#             "Very good": ["A"],
#             "Good": ["B"],
#             "Neither good nor bad": ["C", "Neither"],
#             "Bad": ["D"],
#             "Very bad": ["E"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\n"
#             f"A: Very good\n"
#             f"B: Good\n"
#             f"C: Neither good nor bad\n"
#             f"D: Bad\n"
#             f"E: Very bad\n"
#             f"Answer 2:"
#         ),
#         {
#             "Very good": ["A"],
#             "Good": ["B"],
#             "Neither good nor bad": ["C", "Neither"],
#             "Bad": ["D"],
#             "Very bad": ["E"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\n"
#             f"Answer 1 (Very good, Good, Neither good nor bad, Bad, Very bad):"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\n"
#             f"Answer 2 (Very good, Good, Neither good nor bad, Bad, Very bad):"
#         ),
#         dealing_with_crime_dict,
#     ),
# },
# # Now thinking about the economy in the country as a whole, would you say that over the past year the nation\'s economy has gotten better, stayed about the same, or gotten worse?
# "economy_change": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if over the past year the nation's economy has gotten better, stayed about the same, or gotten worse,\n\n"
#             f" I would say the economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
#             f"A: The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
#             f"ANSWER: The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
#             f"ANSWER: The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"A: The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"P1: I would say the economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between gotten better, gotten worse, or stayed the same, over the past year the nation's economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Over the past year the nation's economy has gotten"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think over the past year the nation's economy has gotten"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} I think over the past year the nation's economy has gotten"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"Answer 11: The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"Answer 12: The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?''\n\n"
#             f" -> '''The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?''\n\n"
#             f" -> '''The economy has"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"A: Gotten better\n"
#             f"B: Gotten worse\n"
#             f"C: Stayed the same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Gotten better": ["A"],
#             "Gotten worse": ["B"],
#             "Stayed about the same": ["C"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"A: Gotten better\n"
#             f"B: Gotten worse\n"
#             f"C: Stayed the same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Gotten better": ["A"],
#             "Gotten worse": ["B"],
#             "Stayed about the same": ["C"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"Answer 1 (Gotten better, Gotten worse, Stayed the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
#             f"Answer 2 (Gotten better, Gotten worse, Stayed the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
# },
# #
# "worry_financial_situation": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked, so far as you and your family are concerned, how worried are you about your current financial situation,\n\n"
#             f" I would say I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: So far as you and your family are concerned, how worried are you about your current financial situation?\n\n"
#             f"A: I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n\n"
#             f"ANSWER: I would say I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n\n"
#             f"ANSWER: I would say I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
#             f"A: I would say I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
#             f"P1: I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding my family's financial situation, between extremely worried, very worried, moderately worried, a little worried, or not at all worried, I am"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding my family's financial situation, I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding my family's financial situation, I think I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} Regarding my family's financial situation, I think I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person feel extremely worried, very worried, moderately worried, a little worried, or not at all worried about his family's financial situation?\n\n"
#             f"ANSWERS:\n1) This person would be"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) would this person feel extremely worried, very worried, moderately worried, a little worried, or not at all worried about his family's financial situation?\n\n"
#             f"ANSWERS:\n1) Yes\n2) This person would be"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\nAnswer 11: I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\nAnswer 12: I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?''\n\n"
#             f" -> '''I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?''\n\n"
#             f" -> '''I am"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
#             f"A: Extremely worried\n"
#             f"B: Very worried\n"
#             f"C: Moderately worried\n"
#             f"D: A little worried\n"
#             f"E: Not at all worried\n"
#             f"Answer 1:"
#         ),
#         {
#             "Extremely worried": ["A", "extremely"],
#             "Very worried": ["B", "very"],
#             "Moderately worried": ["C", "moderately"],
#             "A little worried": ["D", "a little"],
#             "Not at all worried": ["E", "not"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
#             f"A: Extremely worried\n"
#             f"B: Very worried\n"
#             f"C: Moderately worried\n"
#             f"D: A little worried\n"
#             f"E: Not at all worried\n"
#             f"Answer 2:"
#         ),
#         {
#             "Extremely worried": ["A", "extremely"],
#             "Very worried": ["B", "very"],
#             "Moderately worried": ["C", "moderately"],
#             "A little worried": ["D", "a little"],
#             "Not at all worried": ["E", "not"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
#             f"Answer 1 (Extremely worried, Very worried, Moderately worried, A little worried, Not at all worried):"
#         ),
#         worried_financial_situation_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
#             f"Answer 2 (Extremely worried, Very worried, Moderately worried, A little worried, Not at all worried):"
#         ),
#         worried_financial_situation_dict,
#     ),
# },
# # What about welfare programs? Should federal spending on welfare programs be increased, decreased, or kept the same?
# "welfare_spending": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if federal spending on welfare programs be increased, decreased, or kept the same,\n\n"
#             f" I would say federal spending on welfare programs should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What about welfare programs? Should federal spending welfare programs be increased, decreased, or kept the same?\n\n"
#             f"A: Federal spending on welfare programs should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on welfare programs be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should federal spending on welfare programs be increased, decreased, or kept the same?\n\n"
#             f"ANSWER:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
#             f"A:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
#             f"P1: I think it should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on welfare programs should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Federal spending on welfare programs should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} I think federal spending on welfare programs should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} I think federal spending on welfare programs should be"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person want federal spending on welfare programs to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person want federal spending on welfare programs to be increased, decreased, or kept the same?\n\n"
#             f"ANSWERS:\n1) Yes\n2)"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should federal spending on welfare programs be increased, decreased, or kept the same?\nAnswer 11:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on welfare programs be increased, decreased, or kept the same?\nAnswer 12:"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on welfare programs be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should federal spending on welfare programs be increased, decreased, or kept the same?''\n\n"
#             f" -> '''"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 1:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
#             f"A: Increased\n"
#             f"B: Decreased\n"
#             f"C: Kept the same\n"
#             f"Answer 2:"
#         ),
#         {
#             "Increased": ["A", "Increased"],
#             "Decreased": ["B", "Decreased"],
#             "Kept the same": ["C", "Kept"],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
#             f"Answer 1 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
#             f"Answer 2 (Increased, Decreased, Kept the same):"
#         ),
#         dealing_with_crime_dict,
#     ),
# },
# # Which comes closest to your view? You can just tell me the number of your choice.
# "gender_view": {
#     "finish_sentence": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"When asked if gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized,\n\n"
#             f" I would say gay couples should be"
#         ),
#         gender_view_dict,
#     ),
#     "delimiter_qa": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Q: What about gay relationship recognition? Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
#             f"A: Gay relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "0_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
#             f"ANSWER: They should be"
#         ),
#         gender_view_dict,
#     ),
#     "1_shot_task_context_question": (
#         lambda row: (
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2_shot()}\n\n"
#             f"QUESTION: Do you like sandwiches?\n\n"
#             f"ANSWER: Yes\n\n"
#             f"TASK: Consider the below demographic information and answer the following question.\n\n"
#             f"CONTEXT: {make_backstory2(row)}\n\n"
#             f"QUESTION: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
#             f"ANSWER: They should be"
#         ),
#         gender_view_dict,
#     ),
#     "explicit_instructions": (
#         lambda row: (
#             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
#             f"{make_backstory3(row)}\n\n"
#             f"Q: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
#             f"A:"
#         ),
#         gender_view_dict,
#     ),
#     "implicit_instructions": (
#         lambda row: (
#             f"P1: {make_backstory1(row)}\n"
#             f"P2: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
#             f"P1: I think they should be"
#         ),
#         gender_view_dict,
#     ),
#     "enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Between fully, somewhat, or not legally recognized, gay couple relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "non_enumerated_response": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding legal recognition, gay couple relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "0_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1(row)} Regarding legal recognition, I think gay couple relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "1_shot_first_person_backstory": (
#         lambda row: (
#             f"{make_backstory1_shot()} I think sandwiches are good.\n\n"
#             f"{make_backstory1(row)} Regarding legal recognition, I think gay couple relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "0_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person say gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
#             f"ANSWERS:\n1) This person would say gay relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "1_shot_chapter_quiz": (
#         lambda row: (
#             f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
#             f"1) According to the above backstory, would this person likely want a sandwich?\n"
#             f"2) According to the above backstory, would this person say gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
#             f"ANSWERS:\n1) Yes\n2) This person would say gay relationships should be"
#         ),
#         gender_view_dict,
#     ),
#     "0_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\nAnswer 11: They should be"
#         ),
#         gender_view_dict,
#     ),
#     "1_shot_survey_response": (
#         lambda row: (
#             f"{make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
#             f"Answer 11: Yes\n\nQuestion 12: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\nAnswer 12: They should be"
#         ),
#         gender_view_dict,
#     ),
#     "0_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?''\n\n"
#             f" -> '''They should be"
#         ),
#         gender_view_dict,
#     ),
#     "1_shot_heavy_delimited": (
#         lambda row: (
#             f"Given the backstory and question, please answer the question appropriately.\n\n"
#             f"'''Backstory -- {make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
#             f" -> '''Yes'''\n\n"
#             f"'''Backstory -- {make_backstory1(row)}''', '''Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?''\n\n"
#             f" -> '''"
#         ),
#         gender_view_dict,
#     ),
#     "0_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
#             f"A: fully recognized\n"
#             f"B: somewhat recognized\n"
#             f"C: not recognized\n"
#             f"Answer 1:"
#         ),
#         {
#             "Gay and lesbian couples should be allowed to legally marry": [
#                 "fully",
#                 "A",
#             ],
#             "Gay and lesbian couples should be allowed to form civil unions but not legally marry": [
#                 "somewhat",
#                 "B",
#             ],
#             "There should be no legal recognition of gay or lesbian couples' relationship": [
#                 "not",
#                 "C",
#             ],
#         },
#     ),
#     "1_shot_multiple_choice": (
#         lambda row: (
#             f"Background info: {make_backstory2(row)}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"A: Yes\n"
#             f"B: No\n"
#             f"C: Indifferent\n"
#             f"Answer 1: Yes\n\n"
#             f"Question 2: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
#             f"A: fully recognized\n"
#             f"B: somewhat recognized\n"
#             f"C: not recognized\n"
#             f"Answer 2:"
#         ),
#         {
#             "Gay and lesbian couples should be allowed to legally marry": [
#                 "fully",
#                 "A",
#             ],
#             "Gay and lesbian couples should be allowed to form civil unions but not legally marry": [
#                 "somewhat",
#                 "B",
#             ],
#             "There should be no legal recognition of gay or lesbian couples' relationship": [
#                 "not",
#                 "C",
#             ],
#         },
#     ),
#     "0_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1(row)}\n\n"
#             f"Question 1: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
#             f"Answer 1 (fully, somewhat, not):"
#         ),
#         gender_view_dict,
#     ),
#     "1_shot_direct_mapping": (
#         lambda row: (
#             f"{make_backstory1_shot()}\n\n"
#             f"Question 1: Do you want a sandwich?\n"
#             f"Answer 1 (Yes, No, Indifferent): Yes"
#             f"{make_backstory1(row)}\n\n"
#             f"Question 2: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
#             f"Answer 2 (fully, somewhat, not):"
#         ),
#         gender_view_dict,
#     ),
# },
# factory.sample_templates(factory.survey_obj.df, dvs=["social_security_spending"], playground=True)


# OLD KYLE TEMPLATES
