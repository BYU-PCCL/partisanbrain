"""@author: Kyle Rogers"""

import numpy as np
from ..dataset_factory import DatasetFactory
from ..surveys.anes_survey import AnesSurvey
from pdb import set_trace as breakpoint
from .. import constants as k


class AnesFactory(DatasetFactory):
    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj, sample_seed=sample_seed, n=n)

    def modify_data(self, df):
        age_dict = {-9: np.nan}

        # 'Inapplicable', 'Democratic party', "None or 'independent'", 'Republican party'
        party_dict = {
            # "Inapplicable": "an independent",
            # For now we'll just drop inapplicables, but we don't want to do
            # that long term
            "Inapplicable": np.nan,
            "Democratic party": "a Democrat",
            "Republican party": "a Republican",
            "None or 'independent'": "an independent",
        }

        # 'High school graduate - High school diploma or equivalent (e.g: GED)', 'Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)', "Bachelor's degree (e.g. BA, AB, BS)", "Master's degree (e.g. MA, MS, MEng, MEd, MSW, MBA)", 'Associate degree in college - academic', 'Some college but no degree', 'Less than high school credential', 'Associate degree in college - occupational/vocational'
        education_dict = {
            "High school graduate - High school diploma or equivalent (e.g: GED)": "graduated from high school",
            "Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)": "went to graduate school",
            "Bachelor's degree (e.g. BA, AB, BS)": "got a bachelor's degree",
            "Master's degree (e.g. MA, MS, MEng, MEd, MSW, MBA)": "got a master's degree",
            "Associate degree in college - academic": "got an associate's degree",
            "Some college but no degree": "went to some college but didn't graduate",
            "Less than high school credential": "didn't graduate from high school",
            "Associate degree in college - occupational/vocational": "got an associate's degree",
        }

        # 'Slightly conservative', 'Liberal', 'Conservative', 'Moderate; middle of the road', 'Slightly liberal', 'Extremely liberal', 'Extremely conservative'
        ideology_dict = {
            "Slightly conservative": "slightly conservative",
            "Liberal": "liberal",
            "Conservative": "conservative",
            "Moderate; middle of the road": "moderate",
            "Slightly liberal": "slightly liberal",
            "Extremely liberal": "extremely liberal",
            "Extremely conservative": "extremely conservative",
        }

        # -"Refused", -"Interview breakoff", "Under $9,999", "$10,000-14,999", "$15,000-19,999", "$20,000-24,999", "$25,000-29,999", "$30,000-34,999", "$35,000-39,999", "$40,000-44,999", "$45,000-49,999", "$50,000-59,999", "$60,000-64,999", "$65,000-69,999", "$70,000-74,999", "$75,000-79,999", "$80,000-89,999", "$90,000-99,999", "$100,000-109,999", "$110,000-124,999", "$125,000-149,999", "$150,000-174,999", "$175,000-249,999", "$250,00 or more",
        income_dict = {
            "Refused": np.nan,
            "Interview breakoff": np.nan,
            "Under $9,999": "under $9,999",
            "$10,000-14,999": "between $10,000-$14,999",
            "$15,000-19,999": "between $15,000-$19,999",
            "$20,000-24,999": "between $20,000-$24,999",
            "$25,000-29,999": "between $25,000-$29,999",
            "$30,000-34,999": "between $30,000-$34,999",
            "$35,000-39,999": "between $35,000-$39,999",
            "$40,000-44,999": "between $40,000-$44,999",
            "$45,000-49,999": "between $45,000-$49,999",
            "$50,000-59,999": "between $50,000-$59,999",
            "$60,000-64,999": "between $60,000-$64,999",
            "$65,000-69,999": "between $65,000-$69,999",
            "$70,000-74,999": "between $70,000-$74,999",
            "$75,000-79,999": "between $75,000-$79,999",
            "$80,000-89,999": "between $80,000-$89,999",
            "$90,000-99,999": "between $90,000-$99,999",
            "$100,000-109,999": "between $100,000-$109,999",
            "$110,000-124,999": "between $110,000-$124,999",
            "$125,000-149,999": "between $125,000-$149,999",
            "$150,000-174,999": "between $150,000-$174,999",
            "$175,000-249,999": "between $175,000-$249,999",
            "$250,00 or more": "over $250,000",
        }

        # 'Undifferentiated Protestant', 'Not religious', 'Mainline Protestant', 'Roman Catholic', 'Other religion', 'Other Christian', 'Evangelical Protestant', 'Jewish'
        religion_dict = {
            "Undifferentiated Protestant": "Religiously speaking, I am a Protestant",
            "Not religious": "I am not religious",
            "Mainline Protestant": "Religiously speaking, I am a Protestant",
            "Roman Catholic": "Religiously speaking, I am Roman Catholic",
            "Other religion": "I am religious and I am not Christian",
            "Other Christian": "Religiously speaking, I am Christian",
            "Evangelical Protestant": "Religiously speaking, I am an Evangelical Protestant",
            "Jewish": "Religiously speaking, I am Jewish",
        }

        # 'White, non-Hispanic'|'Black, non-Hispanic'|'Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone'|'Hispanic'|'Multiple races, non-Hispanic'|'Native American/Alaska Native or other race, non-Hispanic alone'
        race_ethnicity_dict = {
            "White, non-Hispanic": "white",
            "Black, non-Hispanic": "Black",
            "Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone": "Asian-American/Pacific Islander",
            "Hispanic": "Hispanic",
            "Multiple races, non-Hispanic": "multiracial",
            "Native American/Alaska Native or other race, non-Hispanic alone": "Native American",
        }

        # 'Married: spouse present', 'Never married', 'Divorced', 'Married: spouse absent {VOL - video/phone only}', 'Widowed', 'Separated'
        marital_status_dict = {
            "Married: spouse present": "am married",
            "Never married": "am single and have never been married",
            "Divorced": "am divorced",
            "Married: spouse absent {VOL - video/phone only}": "am married but don't live with my spouse",
            "Widowed": "am widowed",
            "Separated": "am separated from my spouse",
        }

        # 'Donald Trump', 'Hillary Clinton', 'Refused', 'Inapplicable', 'Other \\{SPECIFY\\}'
        vote_2016_dict = {
            "Donald Trump": "Donald Trump",
            "Hillary Clinton": "Hillary Clinton",
            "Refused": np.nan,
            "Inapplicable": np.nan,
            "Other \\{SPECIFY\\}": np.nan,
        }

        social_security_dict = {
            'Increased': 'increased',
            'Decreased': 'decreased',
            'Kept the same': 'kept the same',
            'Refused': np.nan,
            'Don\'t know': np.nan,
        }

        # Make a processed version of the
        # "age", "gender", "party", "education", "ideology", "income", "religion", "race_ethnicity", "region", "marital_status",
        # columns
        df["age_processed"] = df["age"].replace(age_dict)
        df["gender_processed"] = df["gender"]
        df["party_processed"] = df["party"].map(party_dict)
        df["education_processed"] = df["education"].map(education_dict)
        df["ideology_processed"] = df["ideology"].map(ideology_dict)
        df["income_processed"] = df["income"].map(income_dict)
        df["religion_processed"] = df["religion"].map(religion_dict)
        df["region_processed"] = df["region"]
        df["race_ethnicity_processed"] = df["race_ethnicity"].map(race_ethnicity_dict)
        df["marital_status_processed"] = df["marital_status"].map(marital_status_dict)

        # process dvs
        df["vote_2016_processed"] = df["vote_2016"].map(vote_2016_dict)
        df["social_security_spending_processed"] = df["social_security_spending"].map(social_security_dict)

        # Only drop invalid demographics, as we'll drop invalid DV values later
        processed_cols = [f"{col}_processed" for col in k.DEMOGRAPHIC_COLNAMES]
        df.dropna(subset=processed_cols, inplace=True)

        return df

    def make_backstory1(self, row):
        # create a generic backstory using values in the demographics
        return (
            f"I am {int(row['age_processed'])} years old. I am {row['gender_processed'].lower()}. "
            f"Politically speaking, I am {row['party_processed']}. "
            f"Ideologically, I am {row['ideology_processed'].lower()}. I {row['education_processed'].lower()}. "
            f"My salary is {row['income_processed']}. "
            f"{row['religion_processed']}. "
            f"I am {row['race_ethnicity_processed']}. I am from the {row['region_processed']}. "
            f"I {row['marital_status_processed']}."
        )

    def make_backstory2(self, row):
        return (
            f"Age: {int(row['age_processed'])}, Gender: {row['gender']}, Political Affiliation: {row['party']}, "
            f"Education: {row['education']}, Ideology: {row['ideology']}, Total Income: {row['income']}, Religion: {row['religion']}, "
            f"Race/Ethnicity: {row['race_ethnicity']}, Region: {row['region']}, Marital Status: {row['marital_status']}"
        )

    def make_backstory3(self, row):
        return (
            f"Q: What is your age?\nA: {int(row['age_processed'])}\n\nQ: What is your gender?\nA: {row['gender']}\n\n"
            f"Q: What is your political affiliation?\nA: {row['party']}\n\nQ: What is your education?\nA: {row['education']}\n\n"
            f"Q: What is your ideology?\nA: {row['ideology']}\n\nQ: What is your income?\nA: {row['income']}\n\n"
            f"Q: What is your religion?\nA: {row['religion']}\n\n"
            f"Q: What is your race/ethnicity?\nA: {row['race_ethnicity']}\n\nQ: What region of the country are you from?\nA: {row['region']}\n\n"
            f"Q: What is your marital status?\nA: {row['marital_status']}"
        )

    def make_backstory4(self, row):
        return (
            f"Question 1: What is your age?\n\nAnswer 1: {row['age']}\n\nQuestion 2: What is your gender?\n\nAnswer 2: {row['gender']}\n\n"
            f"Question 3: What is your political affiliation?\n\nAnswer 3: {row['party']}\n\nQuestion 4: What is your education?\n\nAnswer 4: {row['education']}\n\n"
            f"Question 5: What is your ideology?\n\nAnswer 5: {row['ideology']}\n\nQuestion 6: What is your income?\n\nAnswer 6: {row['income']}\n\n"
            f"Question 7: What is your religion?\n\nAnswer 7: {row['religion']}\n\n"
            f"Question 8: What is your race/ethnicity?\n\nAnswer 8: {row['race_ethnicity']}\n\nQuestion 9: What region of the country are you from?\n\nAnswer 9: {row['region']}\n\n"
            f"Question 10: What is your marital status?\n\nAnswer 10: {row['marital_status']}"
        )

    def make_backstory5(self, row):
        return (
            f"Person 1: What is your age?\nPerson 2: {row['age']}\n\nPerson 1: What is your gender?\nPerson 2: {row['gender']}\n\n"
            f"Person 1: What is your political affiliation?\nPerson 2: {row['party']}\n\nPerson 1: What is your education?\nPerson 2: {row['education']}\n\n"
            f"Person 1: What is your ideology?\nPerson 2: {row['ideology']}\n\nPerson 1: What is your income?\nPerson 2: {row['income']}\n\n"
            f"Person 1: What is your religion?\nPerson 2: {row['religion']}\n\n"
            f"Person 1: What is your race/ethnicity?\nPerson 2: {row['race_ethnicity']}\n\nPerson 1: What region of the country are you from?\nPerson 2: {row['region']}\n\n"
            f"Person 1: What is your marital status?\nPerson 2: {row['marital_status']}"
        )

    def make_backstory6(self, row):
        return (
            f"Q: What is your age? (in years)\n"
            f"A: {int(row['age_processed'])}\n\n"
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

    def get_templates(self):
        protecting_environment_spending_dict = None
        rising_temp_action_dict = {
            "Should be doing more": "more",
            "Should be doing less": "less",
            "Is currently doing the right amount": ["the", "about"],
        }
        dealing_with_crime_dict = None
        trump_handling_economy_dict = {
            "Approve": "approves",
            "Disapprove": "disapproves",
        }
        govt_waste_money_dict = {
            "Don't waste very much": ["none", "none of it"],
            "Waste a lot": ["a", "all", "a lot"],
            "Waste some": ["some", "some of it"],
        }
        worried_financial_situation_dict = {
            "Moderately worried": ["worried"],
            "Extremely worried": ["extremely"],
            "Very worried": ["very"],
            "A little worried": ["a little"],
            "Not at all worried": ["not"],
        }
        gender_view_dict = {
            "Gay and lesbian couples should be allowed to legally marry": "fully",
            "Gay and lesbian couples should be allowed to form civil unions but not legally marry": "somewhat",
            "There should be no legal recognition of gay or lesbian couples' relationship": "not",
        }

        # Social Security Spending
        social_security_tokens = {
            'increased': ['increased'],
            'decreased': ['decreased'],
            'kept the same': ['kept the same'],
        }

        social_security_mc_tokens = {
            'increased': ['A', 'increased'],
            'decreased': ['B', 'decreased'],
            'kept the same': ['C', 'kept the same'],
        }

        # Vote 2016
        vote_2016_dict = {
            "Hillary Clinton": ["Hillary", "Clinton"],
            "Donald Trump": ["Donald", "Trump"],
        }

        vote_2016_mc_dict = {
            "Hillary Clinton": ["A", "Hillary", "Clinton"],
            "Donald Trump": ["B", "Donald", "Trump"],
        }
        # add token sets as dicts see example (map Kept the same to keep, kept, etc) -- for protecting_environment_spending

        return {
            "vote_2016": {
                "surveyq_exact": (
                    lambda row: self.make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Four years ago, in 2016, Hillary Clinton ran "
                        "on the Democratic ticket against Donald Trump for the "
                        "Republicans. Which one did you vote for?\nAnswer 11:"
                    ),
                    vote_2016_dict,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        self.make_backstory4(row)
                        + "\n\nQuestion 11: Do you remember the presidential "
                        "election of 2016? Did you vote for Hillary Clinton "
                        "or Donald Trump?\nAnswer 11:"
                    ),
                    vote_2016_dict,
                ),
                "fpbs": (
                    lambda row: self.make_backstory1(row)
                    + (" In the 2016 presidential election, I voted for"),
                    vote_2016_dict,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer who that person voted for.\n\n"
                        + self.make_backstory2(row)
                        + ", 2016 Presidential Election "
                        "Vote:"
                    ),
                    vote_2016_dict,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots("vote_2016", "task", 3, sep="\n\n")
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer who that person voted for.\n\n"
                        + self.make_backstory2(row)
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
                        "\n\n" + self.make_backstory3(row) + "\n\nQ: Who did you "
                        "vote for in the 2016 presidential election?\nA:"
                    ),
                    vote_2016_dict,
                ),
                "conversation": (
                    lambda row: self.make_backstory5(row) + "\n\nPerson 1: Who did "
                    "you vote for in the 2016 presidential election?\nPerson 2:",
                    vote_2016_dict,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + self.make_backstory4(row)
                        + "\n\nQuestion 11: Who did "
                        "you vote for in the 2016 presidential election?\nA: "
                        "Hillary Clinton\nB: Donald Trump\n\nAnswer 11:"
                    ),
                    vote_2016_mc_dict,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots("vote_2016", "mc", 2, "\n\n")
                    + "SURVEY_RESPONSE\n\n"
                    + self.make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Who did "
                        "you vote for in the 2016 presidential election?\nA: "
                        "Hillary Clinton\nB: Donald Trump\n\nAnswer 11:"
                    ),
                    vote_2016_dict,
                ),
                "explicit_enumeration": (
                    lambda row: self.make_backstory6(row)
                    + (
                        "\n\nQ: Who did you"
                        f"vote for in the 2016 presidential election? (Donald "
                        f"Trump, Hillary Clinton)\nA:"
                    ),
                    vote_2016_dict,
                ),
            },
            # What about Social Security? Should federal spending on Social Security be increased, decreased, or kept the same?
            "social_security_spending": {
                "surveyq_exact": (
                    lambda row: self.make_backstory4(row)
                    + (
                        "\n\nQuestion 11: What about Social Security? "
                        "Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "surveyq_rephrase": (
                    lambda row: (
                        self.make_backstory4(row)
                        + "\n\nQuestion 11: Do you think federal spending on Social Security should be increased, decreased, or kept the same?\nAnswer 11:"
                    ),
                    social_security_tokens,
                ),
                "fpbs": (
                    lambda row: self.make_backstory1(row)
                    + (" If asked if social security spending should be increased, decreased, or kept the same, I would say that it should be"),
                    social_security_tokens,
                ),
                "task": (
                    lambda row: (
                        "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + self.make_backstory2(row)
                        + ", Opinion on Social Security spending (increased, decreased, or kept the same):"
                    ),
                    social_security_tokens,
                ),
                "task_3shot": (
                    lambda row: (
                        self.get_shots("social_security_spending", "task", 3, "\n\n-----------------------\n\n")
                        + "TASK: The following is a description of a "
                        "voter in the 2016 presidential election. Please read "
                        "it and infer their opinion on federal Social Security spending.\n\n"
                        + self.make_backstory2(row)
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
                        "\n\n" + self.make_backstory3(row) + "\n\nQ: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?\nA:"
                    ),
                    social_security_tokens,
                ),
                "conversation": (
                    lambda row: self.make_backstory5(row) + "\n\nPerson 1: Should federal spending "
                    "on Social Security be increased, decreased, or kept the same?\nPerson 2:",
                    social_security_tokens,
                ),
                "mc": (
                    lambda row: (
                        "SURVEY_RESPONSE\n\n"
                        + self.make_backstory4(row)
                        + "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_2shot": (
                    lambda row: self.get_shots("social_security_spending", "mc", 2, "\n\n-----------------------\n\n")
                    + "SURVEY_RESPONSE\n\n"
                    + self.make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                "mc_5shot": (
                    lambda row: self.get_shots("social_security_spending", "mc", 7, "\n\n-----------------------\n\n")
                    + "SURVEY_RESPONSE\n\n"
                    + self.make_backstory4(row)
                    + (
                        "\n\nQuestion 11: Should federal spending "
                        "on Social Security be increased, decreased, or kept the same?"
                        "\nA: Increased\nB: Decreased\nC: Kept the same\n\nAnswer 11:"
                    ),
                    social_security_mc_tokens,
                ),
                # "explicit_enumeration": (
                #     lambda row: self.make_backstory6(row)
                #     + (
                #         "\n\nQ: Should federal spending "
                #         f"on Social Security be increased, decreased, or kept the same? (Increased, Decreased, Kept the same)\nA:"
                #     ),
                #     social_security_tokens,
                # ),
            },
            # "protecting_environment_spending": {
            #     "finish_sentence": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if federal spending on protecting the environment should be increased, decreased or stay the same,\n\n"
            #             f" I would say federal spending on protecting the environment should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What about protecting the environment? Should federal spending on protecting the environment be increased, decreased, or kept the same?\n\n"
            #             f"A: Federal spending on protecting the environment should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you consider environment protection to be important to you?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
            #             f"A:"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should federal spending on protecting the environment should be increased, decreased or stay the same?\n"
            #             f"P1: I think it should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on protecting the environment should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Federal spending on protecting the environment should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think federal spending on protecting the environment should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think environmental protection is important.\n\n"
            #             f"{self.make_backstory1(row)} I think federal spending on protecting the environment should be"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person want federal spending on protecting the environment to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely support green reforms?\n"
            #             f"2) According to the above backstory, would this person want federal spending on protecting the environment to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should federal spending on protecting the environment be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you consider environment protection to be important to you?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on protecting the environment be increased, decreased, or kept the same?\nAnswer 12:"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on protecting the environment be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you consider environment protection to be important to you?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on protecting the environment be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
            #             f"Answer 1 (Increased, Decreased, Kept the same):"
            #         ),
            #         protecting_environment_spending_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you consider environment protection to be important to you?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked should the federal government be doing more, less, or the same about rising temperatures,\n\n"
            #             f" I would say federal government should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: Do you think the federal government should be doing more about rising temperatures, should be doing less, or should be doing the same?\n\n"
            #             f"A: The federal government should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should the federal government be doing more, less, or the same about rising temperatures?\n\n"
            #             f"ANSWER:"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you consider global warming to be important to you?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should the federal government be doing more, less, or the same about rising temperatures?\n\n"
            #             f"ANSWER:"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should the federal government be doing more, less, or the same about rising temperatures?\n"
            #             f"A: It should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should the federal government be doing more, less, or the same about rising temperatures?\n"
            #             f"P1: I think it should be doing"
            #         ),
            #         {},
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between more, less, or the same, regarding rising temperatures the federal government should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding rising temperatures the federal government should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding rising temperatures, I think the federal government should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think global warming is an important issue.\n\n"
            #             f"{self.make_backstory1(row)} Regarding rising temperatures, I think the federal government should be doing"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person want the federal government to do more, less, or the same about rising temperatures?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely think gloabal warming is important?\n"
            #             f"2) According to the above backstory, would this person want the federal government to do more, less, or the same about rising temperatures?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should the federal government to do more, less, or the same about rising temperatures?\nAnswer 11:"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you think global warming is important?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should the federal government to do more, less, or the same about rising temperatures?\nAnswer 12:"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should the federal government to do more, less, or the same about rising temperatures?''\n\n"
            #             f" -> '''It should do"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you consider global warming to be important?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should the federal government to do more, less, or the same about rising temperatures?''\n\n"
            #             f" -> '''It should do"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should the federal government to do more, less, or the same about rising temperatures?\n"
            #             f"Answer 1 (More, Less, The same):"
            #         ),
            #         rising_temp_action_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you consider environment protection to be important to you?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if federal spending on dealing with crime should be increased, decreased or stay the same,\n\n"
            #             f" I would say federal spending on dealing with crime should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What about dealing with crime? Should federal spending on dealing with crime be increased, decreased, or kept the same?\n\n"
            #             f"A: Federal spending on dealing with crime should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you consider public safety to be important to you?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
            #             f"A:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
            #             f"P1: I think it should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on dealing with crime should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Federal spending on dealing with crime should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think federal spending on dealing with crime should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think public safety is important.\n\n"
            #             f"{self.make_backstory1(row)} I think federal spending on dealing with crime should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person want federal spending on dealing with crime to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person want federal spending on dealing with crime to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should federal spending on dealing with crime be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on dealing with crime be increased, decreased, or kept the same?\nAnswer 12:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on dealing with crime be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on dealing with crime be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
            #             f"Answer 1 (Increased, Decreased, Kept the same):"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if I approve or disapprove of the way Donald Trump is handling the economy,\n\n"
            #             f" I would say I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: Do you approve or disapprove of the way Donald Trump is handling the economy?\n\n"
            #             f"A: I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Does this person approve or disapprove of the way Donald Trump is handling the economy?\n\n"
            #             f"ANSWER: This person"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you want a sandwich?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Does this person approve or disapprove of the way Donald Trump is handling the economy?\n\n"
            #             f"ANSWER: This person"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Does this person approve or disapprove of the way Donald Trump is handling the economy?\n"
            #             f"A: This person"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
            #             f"P1: I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding the way Donald Trump is handling the economy, between approve or disapprove, I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f'{self.make_backstory1(row)} If someone asked me "Do you approve or disapprove of the way Donald Trump is handling the economy?", I would say "I'
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} My friend asked if I approve or disapprove of the way Donald Trump is handling the economy, I said I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} My friend asked if I approve or disapprove of the way Donald Trump is handling the economy, I said I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, regarding the way Donald Trump is handling the economy, would this person approve or disapprove?\n\n"
            #             f"ANSWERS:\n1) This person would"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, According to the above backstory, regarding the way Donald Trump is handling the economy, would this person approve or disapprove?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you approve or disapprove of the way Donald Trump is handling the economy?\nAnswer 11: I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Do you approve or disapprove of the way Donald Trump is handling the economy?\nAnswer 12: I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Do you approve or disapprove of the way Donald Trump is handling the economy?''\n\n"
            #             f" -> '''I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Do you approve or disapprove of the way Donald Trump is handling the economy?''\n\n"
            #             f" -> '''I"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
            #             f"Question 1: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
            #             f"A: Approve\n"
            #             f"B: Disapprove\n"
            #             f"Answer 1:"
            #         ),
            #         {"Approve": ["A"], "Disapprove": ["B"]},
            #     ),
            #     "1_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Do you approve or disapprove of the way Donald Trump is handling the economy?\n"
            #             f"Answer 1 (Approve, Disapprove):"
            #         ),
            #         trump_handling_economy_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if people in government waste a lot of the money we pay in taxes, waste some of it, or none of it,\n\n"
            #             f" I would say people in the government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
            #             f"A: I think people in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
            #             f"ANSWER: People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
            #             f"ANSWER: People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
            #             f"A: People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
            #             f"P1: I think people in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between a lot, some, or none, people in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding tax money, people in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding tax money, I think people in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} Regarding tax money, I think people in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person feel that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
            #             f"ANSWERS:\n1) This person would say they waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person feel that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n\n"
            #             f"ANSWERS:\n1) Yes\n2) This person would say they waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
            #             f"Answer 11: People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
            #             f"Answer 12: People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?''\n\n"
            #             f" -> '''People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?''\n\n"
            #             f" -> '''People in government waste"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Do people in government waste a lot of the money we pay in taxes, waste some of it, or none of it?\n"
            #             f"Answer 1 (A lot, Some, None):"
            #         ),
            #         govt_waste_money_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if federal spending on Social Security be increased, decreased, or kept the same,\n\n"
            #             f" I would say federal spending on Social Security should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What about Social Security? Should federal spending on Social Security be increased, decreased, or kept the same?\n\n"
            #             f"A: Federal spending on Social Security should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on Social Security be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on Social Security be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
            #             f"A:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
            #             f"P1: I think it should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on Social Security should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Federal spending on Social Security should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think federal spending on Social Security should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} I think federal spending on Social Security should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person want federal spending on Social Security to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person want federal spending on Social Security to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on Social Security be increased, decreased, or kept the same?\nAnswer 12:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on Social Security be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on Social Security be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should federal spending on Social Security be increased, decreased, or kept the same?\n"
            #             f"Answer 1 (Increased, Decreased, Kept the same):"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if federal spending on aid to the poor be increased, decreased, or kept the same,\n\n"
            #             f" I would say federal spending on aid to the poor should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What about aid to the poor? Should federal spending on aid to the poor be increased, decreased, or kept the same?\n\n"
            #             f"A: Federal spending on aid to the poor should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
            #             f"A:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
            #             f"P1: I think it should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on aid to the poor should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Federal spending on aid to the poor should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think federal spending on aid to the poor should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} I think federal spending on aid to the poor should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person want federal spending on aid to the poor to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person want federal spending on aid to the poor to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should federal spending on aid to the poor be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on aid to the poor be increased, decreased, or kept the same?\nAnswer 12:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on aid to the poor be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on aid to the poor be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should federal spending on aid to the poor be increased, decreased, or kept the same?\n"
            #             f"Answer 1 (Increased, Decreased, Kept the same):"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if the state of the United States economy is very good, good, neither good nor bad, bad, or very bad,\n\n"
            #             f" I would say the economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n\n"
            #             f"A: The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n\n"
            #             f"ANSWER: The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n\n"
            #             f"ANSWER: The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n"
            #             f"A: The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good, good, neither good nor bad, bad, or very bad?\n"
            #             f"P1: I think the economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between very good, good, neither good nor bad, bad or very bad, the United States economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} The United States economy these days is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think the United States economy these days is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} I think the United States economy these days is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person think the United States economy is very good, good, neither good nor bad, bad or very bad?\n\n"
            #             f"ANSWERS:\n1) This person would think the economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person think the United States economy is very good, good, neither good nor bad, bad or very bad?\n\n"
            #             f"ANSWERS:\n1) Yes\n2) This person would think the economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\nAnswer 11: The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\nAnswer 12: The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?''\n\n"
            #             f" -> '''The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?''\n\n"
            #             f" -> '''The economy is"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?\n"
            #             f"Answer 1 (Very good, Good, Neither good nor bad, Bad, Very bad):"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if over the past year the nation's economy has gotten better, stayed about the same, or gotten worse,\n\n"
            #             f" I would say the economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
            #             f"A: The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
            #             f"ANSWER: The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
            #             f"ANSWER: The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
            #             f"A: The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
            #             f"P1: I would say the economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between gotten better, gotten worse, or stayed the same, over the past year the nation's economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Over the past year the nation's economy has gotten"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think over the past year the nation's economy has gotten"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} I think over the past year the nation's economy has gotten"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
            #             f"Answer 11: The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
            #             f"Answer 12: The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?''\n\n"
            #             f" -> '''The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?''\n\n"
            #             f" -> '''The economy has"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?\n"
            #             f"Answer 1 (Gotten better, Gotten worse, Stayed the same):"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked, so far as you and your family are concerned, how worried are you about your current financial situation,\n\n"
            #             f" I would say I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: So far as you and your family are concerned, how worried are you about your current financial situation?\n\n"
            #             f"A: I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n\n"
            #             f"ANSWER: I would say I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n\n"
            #             f"ANSWER: I would say I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
            #             f"A: I would say I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
            #             f"P1: I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding my family's financial situation, between extremely worried, very worried, moderately worried, a little worried, or not at all worried, I am"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding my family's financial situation, I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding my family's financial situation, I think I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} Regarding my family's financial situation, I think I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person feel extremely worried, very worried, moderately worried, a little worried, or not at all worried about his family's financial situation?\n\n"
            #             f"ANSWERS:\n1) This person would be"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) would this person feel extremely worried, very worried, moderately worried, a little worried, or not at all worried about his family's financial situation?\n\n"
            #             f"ANSWERS:\n1) Yes\n2) This person would be"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\nAnswer 11: I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\nAnswer 12: I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?''\n\n"
            #             f" -> '''I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?''\n\n"
            #             f" -> '''I am"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: So far as you and your family are concerned, would you say you are extremely worried, very worried, moderately worried, a little worried or not at all worried about your current financial situation?\n"
            #             f"Answer 1 (Extremely worried, Very worried, Moderately worried, A little worried, Not at all worried):"
            #         ),
            #         worried_financial_situation_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if federal spending on welfare programs be increased, decreased, or kept the same,\n\n"
            #             f" I would say federal spending on welfare programs should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What about welfare programs? Should federal spending welfare programs be increased, decreased, or kept the same?\n\n"
            #             f"A: Federal spending on welfare programs should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on welfare programs be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should federal spending on welfare programs be increased, decreased, or kept the same?\n\n"
            #             f"ANSWER:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
            #             f"A:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
            #             f"P1: I think it should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on welfare programs should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Federal spending on welfare programs should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} I think federal spending on welfare programs should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} I think federal spending on welfare programs should be"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person want federal spending on welfare programs to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person want federal spending on welfare programs to be increased, decreased, or kept the same?\n\n"
            #             f"ANSWERS:\n1) Yes\n2)"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should federal spending on welfare programs be increased, decreased, or kept the same?\nAnswer 11:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should federal spending on welfare programs be increased, decreased, or kept the same?\nAnswer 12:"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on welfare programs be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should federal spending on welfare programs be increased, decreased, or kept the same?''\n\n"
            #             f" -> '''"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should federal spending on welfare programs be increased, decreased, or kept the same?\n"
            #             f"Answer 1 (Increased, Decreased, Kept the same):"
            #         ),
            #         dealing_with_crime_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"When asked if gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized,\n\n"
            #             f" I would say gay couples should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "delimiter_qa": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Q: What about gay relationship recognition? Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
            #             f"A: Gay relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "0_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
            #             f"ANSWER: They should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "1_shot_task_context_question": (
            #         lambda row: (
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2_shot()}\n\n"
            #             f"QUESTION: Do you like sandwiches?\n\n"
            #             f"ANSWER: Yes\n\n"
            #             f"TASK: Consider the below demographic information and answer the following question.\n\n"
            #             f"CONTEXT: {self.make_backstory2(row)}\n\n"
            #             f"QUESTION: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
            #             f"ANSWER: They should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "explicit_instructions": (
            #         lambda row: (
            #             f"The ANES is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
            #             f"{self.make_backstory3(row)}\n\n"
            #             f"Q: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
            #             f"A:"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "implicit_instructions": (
            #         lambda row: (
            #             f"P1: {self.make_backstory1(row)}\n"
            #             f"P2: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
            #             f"P1: I think they should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Between fully, somewhat, or not legally recognized, gay couple relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "non_enumerated_response": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding legal recognition, gay couple relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "0_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1(row)} Regarding legal recognition, I think gay couple relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "1_shot_first_person_backstory": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()} I think sandwiches are good.\n\n"
            #             f"{self.make_backstory1(row)} Regarding legal recognition, I think gay couple relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "0_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person say gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
            #             f"ANSWERS:\n1) This person would say gay relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "1_shot_chapter_quiz": (
            #         lambda row: (
            #             f"CHAPTER QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTIONS:\n"
            #             f"1) According to the above backstory, would this person likely want a sandwich?\n"
            #             f"2) According to the above backstory, would this person say gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n\n"
            #             f"ANSWERS:\n1) Yes\n2) This person would say gay relationships should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "0_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\nAnswer 11: They should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "1_shot_survey_response": (
            #         lambda row: (
            #             f"{self.make_backstory4(row)}\n\nQuestion 11: Do you want a sandwich?\n"
            #             f"Answer 11: Yes\n\nQuestion 12: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\nAnswer 12: They should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "0_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?''\n\n"
            #             f" -> '''They should be"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "1_shot_heavy_delimited": (
            #         lambda row: (
            #             f"Given the backstory and question, please answer the question appropriately.\n\n"
            #             f"'''Backstory -- {self.make_backstory1_shot()}''', '''Do you want a sandwich?''\n\n"
            #             f" -> '''Yes'''\n\n"
            #             f"'''Backstory -- {self.make_backstory1(row)}''', '''Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?''\n\n"
            #             f" -> '''"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "0_shot_multiple_choice": (
            #         lambda row: (
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"Background info: {self.make_backstory2(row)}\n\n"
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
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 1: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
            #             f"Answer 1 (fully, somewhat, not):"
            #         ),
            #         gender_view_dict,
            #     ),
            #     "1_shot_direct_mapping": (
            #         lambda row: (
            #             f"{self.make_backstory1_shot()}\n\n"
            #             f"Question 1: Do you want a sandwich?\n"
            #             f"Answer 1 (Yes, No, Indifferent): Yes"
            #             f"{self.make_backstory1(row)}\n\n"
            #             f"Question 2: Should gay couples should be fully legally recognized, somewhat legally recognized, or not legally recognized?\n"
            #             f"Answer 2 (fully, somewhat, not):"
            #         ),
            #         gender_view_dict,
            #     ),
            # },
        }


if __name__ == "__main__":
    factory = AnesFactory(AnesSurvey(force_recreate=True), n=500)
    factory.sample_templates(factory.survey_obj.df, dvs=["vote_2016", "social_security_spending"], playground=True)
    # factory.sample_templates(factory.survey_obj.df, dvs=["social_security_spending"], playground=True)


# OLD KYLE TEMPLATES
