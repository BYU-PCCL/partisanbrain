"""@author: Kyle Rogers, Chris Rytting, Taylor Sorensen"""

from ..survey import Survey
from ..constants import DEMOGRAPHIC_COLNAMES, SURVEY_DATA_PATH

import os
import pandas as pd
import requests
import zipfile
import numpy as np
from pdb import set_trace as breakpoint

SURVEY_URL = "https://electionstudies.org/anes_timeseries_2020_csv_20210719/"


class AnesSurvey(Survey):
    def __init__(self, force_recreate=False):
        super().__init__(force_recreate=force_recreate)

    def download_data(self):
        survey_data_csv_path = f"{SURVEY_DATA_PATH}/anes_survey/raw.csv"
        survey_data_zip_path = f"{SURVEY_DATA_PATH}/anes_survey/raw.csv"
        if not os.path.exists(survey_data_csv_path):
            # download the data from the link: https://electionstudies.org/anes_timeseries_2020_csv_20210719/
            fzip = requests.get(SURVEY_URL)
            # save the zip file
            with open(survey_data_zip_path, "wb") as f:
                f.write(fzip.content)

            # unzip the file
            with zipfile.ZipFile(survey_data_zip_path, "r") as zip_ref:
                zip_ref.extractall(f"{SURVEY_DATA_PATH}/anes_survey")
            # rename the csv file
            os.rename(
                f"{SURVEY_DATA_PATH}/anes_survey/anes_timeseries_2020_csv_20210719.csv",
                f"{SURVEY_DATA_PATH}/anes_survey/raw.csv",
            )

        df = pd.read_csv(survey_data_csv_path)
        return df

    def modify_data(self, df):
        # Rename demographic columns.
        # breakpoint()
        mod_df = df.rename(
            columns={
                "V201507x": "age",
                # Bad example of gender
                # "V201004": "gender",
                "V201600": "gender",
                "V201018": "party",
                "V201510": "education",
                "V201200": "ideology",
                "V201617x": "income",
                "V201458x": "religion",
                "V201549x": "race_ethnicity",
                "V203003": "region",
                "V201508": "marital_status",
            }
        )

        # Drop rows which don't contain full demographics.
        mod_df = mod_df.dropna(subset=DEMOGRAPHIC_COLNAMES)

        # Rename DV columns.
        # note: econ0, intl_affairs0, intl_affairs1 have an alternate question and corresponding code.
        mod_df = mod_df.rename(
            columns={
                "V201321": "protecting_environment_spending",
                "V201401": "rising_temp_action",
                "V201309": "dealing_with_crime_spending",
                "V201130": "trump_handling_economy",
                "V201235": "govt_waste_money",
                "V201300": "social_security_spending",
                "V201318": "aid_poor_spending",
                "V201324": "state_of_economy",
                "V201325": "economy_change",
                "V201594": "worry_financial_situation",
                # "V201312": "welfare_spending",
                "V201416": "gay_marriage",
                "V201133": "trump_handling_relations",
                "V201139": "trump_hanlding_immigration",
                "V201350": "willing_military_force",
                "V201619": "restless_sleep",
                "V201620": "have_health_insurance",
                "V201006": "attn_to_politics",
                "V201223": "feel_voting_is_duty",
                "V201234": "govt_run_by_who",
                "V201103": "vote_2016",
            }
        )

        # Drop all columns that are not DV questions or demographic columns.
        column_lst = DEMOGRAPHIC_COLNAMES + [
            "protecting_environment_spending",
            "rising_temp_action",
            "dealing_with_crime_spending",
            "trump_handling_economy",
            "govt_waste_money",
            "social_security_spending",
            "aid_poor_spending",
            "state_of_economy",
            "economy_change",
            "worry_financial_situation",
            # "welfare_spending",
            "gay_marriage",
            "trump_handling_relations",
            "trump_hanlding_immigration",
            "willing_military_force",
            "restless_sleep",
            "have_health_insurance",
            "attn_to_politics",
            "feel_voting_is_duty",
            "govt_run_by_who",
            "vote_2016",
        ]
        mod_df = mod_df[column_lst]

        # Clean data by renaming codes to natural langauge responses using codebook.
        # note taken from old survey code - income and age are not coded.
        demographics_dict = {
            # "age": {
            #     {k:str(k) for k in range(80)}.update({80: "80 or older"}),
            # },
            "gender": {
                1: "Male",
                2: "Female",
            },
            "party": {
                -9: "Refused",
                -8: "Don't know",
                -1: "Inapplicable",
                1: "Democratic party",
                2: "Republican party",
                4: "None or 'independent'",
            },
            "education": {
                1: "Less than high school credential",
                2: "High school graduate - High school diploma or equivalent (e.g: GED)",
                3: "Some college but no degree",
                4: "Associate degree in college - occupational/vocational",
                5: "Associate degree in college - academic",
                6: "Bachelor's degree (e.g. BA, AB, BS)",
                7: "Master's degree (e.g. MA, MS, MEng, MEd, MSW, MBA)",
                8: "Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)",
            },
            "ideology": {
                1: "Extremely liberal",
                2: "Liberal",
                3: "Slightly liberal",
                4: "Moderate; middle of the road",
                5: "Slightly conservative",
                6: "Conservative",
                7: "Extremely conservative",
            },
            "income": {
                -9: "Refused",
                -5: "Interview breakoff",
                1: "Under $9,999",
                2: "$10,000-14,999",
                3: "$15,000-19,999",
                4: "$20,000-24,999",
                5: "$25,000-29,999",
                6: "$30,000-34,999",
                7: "$35,000-39,999",
                8: "$40,000-44,999",
                9: "$45,000-49,999",
                10: "$50,000-59,999",
                11: "$60,000-64,999",
                12: "$65,000-69,999",
                13: "$70,000-74,999",
                14: "$75,000-79,999",
                15: "$80,000-89,999",
                16: "$90,000-99,999",
                17: "$100,000-109,999",
                18: "$110,000-124,999",
                19: "$125,000-149,999",
                20: "$150,000-174,999",
                21: "$175,000-249,999",
                22: "$250,00 or more",
            },
            "religion": {
                1: "Mainline Protestant",
                2: "Evangelical Protestant",
                3: "Black Protestant",
                4: "Undifferentiated Protestant",
                5: "Roman Catholic",
                6: "Other Christian",
                7: "Jewish",
                8: "Other religion",
                9: "Not religious",
            },
            "race_ethnicity": {
                1: "White, non-Hispanic",
                2: "Black, non-Hispanic",
                3: "Hispanic",
                4: "Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone",
                5: "Native American/Alaska Native or other race, non-Hispanic alone",
                6: "Multiple races, non-Hispanic",
            },
            "region": {
                1: "Northeast",
                2: "Midwest",
                3: "South",
                4: "West",
            },
            "marital_status": {
                -9: "Refused",
                -8: "Don't know",
                1: "Married: spouse present",
                2: "Married: spouse absent {VOL - video/phone only}",
                3: "Widowed",
                4: "Divorced",
                5: "Separated",
                6: "Never married",
            },
        }

        answers_dict = {
            "protecting_environment_spending": {
                -9: "Refused",
                -8: "Don't know",
                1: "Increased",
                2: "Decreased",
                3: "Kept the same",
            },
            "rising_temp_action": {
                -9: "Refused",
                -8: "Don't know",
                1: "Should be doing more",
                2: "Should be doing less",
                3: "Is currently doing the right amount",
            },
            "dealing_with_crime_spending": {
                -9: "Refused",
                -8: "Don't know",
                1: "Increased",
                2: "Decreased",
                3: "Kept the same",
            },
            "trump_handling_economy": {
                -9: "Refused",
                -8: "Don't know",
                1: "Approve",
                2: "Disapprove",
            },
            "govt_waste_money": {
                -9: "Refused",
                -8: "Don't know",
                1: "Waste a lot",
                2: "Waste some",
                3: "Don't waste very much",
            },
            "social_security_spending": {
                -9: "Refused",
                -8: "Don't know",
                1: "Increased",
                2: "Decreased",
                3: "Kept the same",
            },
            "aid_poor_spending": {
                -9: "Refused",
                -8: "Don't know",
                1: "Increased",
                2: "Decreased",
                3: "Kept the same",
            },
            "state_of_economy": {
                -9: "Refused",
                -8: "Don't know",
                1: "Very good",
                2: "Good",
                3: "Neither good nor bad",
                4: "Bad",
                5: "Very bad",
            },
            "economy_change": {
                -9: "Refused",
                -8: "Don't know",
                1: "Gotten better",
                2: "Stayed about the same",
                3: "Gotten worse",
            },
            "worry_financial_situation": {
                -9: "Refused",
                -8: "Don't know",
                1: "Extremely worried",
                2: "Very worried",
                3: "Moderately worried",
                4: "A little worried",
                5: "Not at all worried",
            },
            # "welfare_spending": {
            #     -9: "Refused",
            #     -8: "Don't know",
            #     1: "Increased",
            #     2: "Decreased",
            #     3: "Kept the same",
            # },
            "gay_marriage": {
                -9: "Refused",
                -8: "Don't know",
                1: "Gay and lesbian couples should be allowed to legally marry",
                2: "Gay and lesbian couples should be allowed to form civil unions but not legally marry",
                3: "There should be no legal recognition of gay or lesbian couples' relationship",
            },
            "trump_handling_relations": {
                -9: "Refused",
                -8: "Don't know",
                1: "Approve",
                2: "Disapprove",
            },
            "trump_hanlding_immigration": {
                -9: "Refused",
                -8: "Don't know",
                1: "Approve",
                2: "Disapprove",
            },
            "willing_military_force": {
                -9: "Refused",
                -8: "Don't know",
                1: "Extremely willing",
                2: "Very willing",
                3: "Moderately willing",
                4: "A little willing",
                5: "Not at all willing",
            },
            "restless_sleep": {
                -9: "Refused",
                -5: "Interview breakoff (sufficient parital IW)",
                1: "All the time",
                2: "Often",
                3: "Sometimes",
                4: "Rarely",
                5: "Never",
            },
            "have_health_insurance": {
                -9: "Refused",
                -5: "Interview breakoff (sufficient parital IW)",
                1: "Yes",
                2: "No",
            },
            "attn_to_politics": {
                -9: "Refused",
                1: "Very much interested",
                2: "Somewhat interested",
                3: "Not much interested",
            },
            "feel_voting_is_duty": {
                -9: "Refused",
                -1: "Inapplicable",
                1: "Very strongly",
                2: "Moderately strongly",
                3: "A little strongly",
            },
            "govt_run_by_who": {
                -9: "Refused",
                -8: "Don't know",
                1: "Run by a few big interests",
                2: "For the benefit of all people",
            },
            "vote_2016": {
                -9: "Refused",
                -8: "Don't know",
                -1: "Inapplicable",
                1: "Hillary Clinton",
                2: "Donald Trump",
                5: "Other \{SPECIFY\}",
            },
        }

        # use above dicts to decode mod_df
        for col in mod_df.columns:
            if col in answers_dict:
                mod_df[col] = mod_df[col].map(answers_dict[col])
            elif col in demographics_dict:
                mod_df[col] = mod_df[col].map(demographics_dict[col])
            else:
                # will pass on income and age demographic series bc not modification needed
                pass

        return mod_df

    def get_dv_questions(self):
        return {
            "protecting_environment_spending": "What about protecting the environment? Should federal spending on protecting the environment be increased, decreased, or kept the same?",
            "rising_temp_action": "Do you think the federal government should be doing more about rising temperatures, should be doing less, or is it currently doing the right amount?",
            "dealing_with_crime_spending": "What about dealing with crime? Should federal spending on dealing with crime be increased, decreased, or kept the same?",
            "trump_handling_economy": "Do you approve or disapprove of the way Donald Trump is handling the economy?",
            "govt_waste_money": "Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or don't waste very much of it?",
            "social_security_spending": "What about Social Security? Should federal spending on Social Security be increased, decreased, or kept the same?",
            "aid_poor_spending": "What about aid to the poor? Should federal spending on aid to the poor be increased, decreased, or kept the same?",
            "state_of_economy": "What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?",
            "economy_change": "Now thinking about the economy in the country as a whole, would you say that over the past year the nation's economy has gotten better, stayed about the same, or gotten worse?",
            "worry_financial_situation": "So far as you and your family are concerned, how worried are you about your current financial situation?",
            # "welfare_spending": "What about welfare programs? Should federal spending on welfare programs be increased, decreased, or kept the same?",
            "gay_marriage": "Which comes closest to your view? You can just tell me the number of your choice.",
            "trump_handling_relations": "Do you approve or disapprove of the way Donald Trump is handling relations with foreign countries?",
            "trump_hanlding_immigration": "Do you approve or disapprove of the way Donald Trump is handling immigration?",
            "willing_military_force": "How willing should the United States be to use military force to solve international problems?",
            "restless_sleep": "In the past week, how often has your sleep been restless?",
            "have_health_insurance": "Do you presently have any kind of health insurance?",
            "attn_to_politics": "Some people don't pay much attention to political campaigns. How about you? Would you say that you have been very much interested, somewhat interested or not much interested in the political campaigns so far this year?",
            "feel_voting_is_duty": "How strongly do you feel that voting is a duty?",
            "govt_run_by_who": "Would you say the government is pretty much run by a few big interests looking out for themselves or that it is run for the benefit of all the people?",
            "vote_2016": "Four years ago, in 2016, Hillary Clinton ran on the Democratic ticket against Donald Trump for the Republicans. Do you remember for sure whether or not you voted in that election? Which one did you vote for?",
        }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = AnesSurvey()
