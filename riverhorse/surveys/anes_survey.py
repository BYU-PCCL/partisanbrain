from ..survey import Survey, UserInterventionNeededError
from ..constants import DEMOGRAPHIC_COLNAMES

import os
import pandas as pd
import requests
import zipfile

SURVEY_URL = "https://electionstudies.org/anes_timeseries_2020_csv_20210719/"

class AnesSurvey(Survey):

    def __init__(self):
        super().__init__()

    def download_data(self):
        if not os.path.exists("survey_data/anes_survey/raw.csv"):
            # download the data from the link: https://electionstudies.org/anes_timeseries_2020_csv_20210719/
            fzip = requests.get(SURVEY_URL)
            # save the zip file
            with open("survey_data/anes_survey/raw.zip", "wb") as f:
                f.write(fzip.content)
            f.close()

            # unzip the file
            with zipfile.ZipFile("survey_data/anes_survey/raw.zip", 'r') as zip_ref:
                zip_ref.extractall("survey_data/anes_survey")
            # rename the csv file
            os.rename("survey_data/anes_survey/anes_timeseries_2020_csv_20210719.csv", "survey_data/anes_survey/raw.csv")

        return pd.read_csv("survey_data/anes_survey/raw.csv")


    def modify_data(self, df):
        # Rename demographic columns.
        mod_df = df.rename(columns={"V201507x" : "age",
                                    "V202637" : "gender",
                                    "V201018": "party",
                                    "V201510": "education",
                                    "V201200" : "ideology",
                                    "V201607" : "income",
                                    "V201458x" : "religion",
                                    "V201549x" : "race_ethnicity",
                                    "V203003" : "region",
                                    "V201508" : "marital_status"})

        # Drop rows which don't contain full demographics.
        mod_df = mod_df.dropna(subset=DEMOGRAPHIC_COLNAMES)

        # Rename DV columns.
        # note: econ0, intl_affairs0, intl_affairs1 have an alternate question and corresponding code.
        mod_df = mod_df.rename(columns={"V201321" : "climate0",
                                        "V201401" : "climate1",
                                        "V201309" : "crime0",
                                        "V201130" : "econ0",
                                        "V201235" : "econ1",
                                        "V201300" : "econ2",
                                        "V201318" : "econ3",
                                        "V201324" : "econ4",
                                        "V201325" : "econ5",
                                        "V201594" : "econ6",
                                        "V201312" : "econ7",
                                        "V201416" : "gender0",
                                        "V201133" : "intl_affairs0",
                                        "V201139" : "intl_affairs1",
                                        "V201350" : "intl_affairs2",
                                        "V201619" : "mental_hlth0",
                                        "V201620" : "physical_hlth0",
                                        "V201006" : "politics0",
                                        "V201223" : "politics1",
                                        "V201234" : "politics2"})

        # Drop all columns that are not DV questions or demographic columns.
        column_lst = DEMOGRAPHIC_COLNAMES + ["climate0", "climate1", "crime0",
                                             "econ0", "econ1", "econ2", "econ3",
                                             "econ4", "econ5", "econ6", "econ7",
                                             "gender0", "intl_affairs0", "intl_affairs1",
                                             "intl_affairs2", "mental_hlth0", "physical_hlth0",
                                             "politics0", "politics1", "politics2"]
        mod_df = mod_df[column_lst]

        # Clean data by renaming codes to natural langauge responses using codebook.
        # note taken from old survey code - income and age are not coded.
        demographics_dict = {
            # "age": {
            #     {k:str(k) for k in range(80)}.update({80: "80 or older"}),
            # },
            "gender" : {
                1 : "Male",
                2 : "Female",
            },
            "party" : {
                -9: "Refused",
                -8: "Don\'t know",
                -1: "Inapplicable",
                1: "Democratic party",
                2: "Republican party",
                4: "None or \'independent\'",
            },
            "education" : {
                1: "Less than high school credential",
                2: "High school graduate - High school diploma or equivalent (e.g: GED)",
                3: "Some college but no degree",
                4: "Associate degree in college - occupational/vocational",
                5: "Associate degree in college - academic",
                6: "Bachelor\'s degree (e.g. BA, AB, BS)",
                7: "Master\'s degree (e.g. MA, MS, MEng, MEd, MSW, MBA)",
                8: "Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)",
            },
            "ideology" : {
                1: "Extremely liberal",
                2: "Liberal",
                3: "Slightly liberal",
                4: "Moderate; middle of the road",
                5: "Slightly conservative",
                6: "Conservative",
                7: "Extremely conservative",
            },
            "religion" : {
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
            "race_ethnicity" : {
                1: "White, non-Hispanic",
                2: "Black, non-Hispanic",
                3: "Hispanic",
                4: "Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone",
                5: "Native American/Alaska Native or other race, non-Hispanic alone",
                6: "Multiple races, non-Hispanic", 
            },
            "reigon" : {
                1: "Northeast",
                2: "Midwest",
                3: "South",
                4: "West",
            },
            "marital_status" : {
                -9: "Refused",
                -8: "Don\'t know",
                1: "Married: spouse present",
                2: "Married: spouse absent {VOL - video/phone only}",
                3: "Widowed",
                4: "Divorced",
                5: "Separated",
                6: "Never married",
            },
        }
        
        answers_dict = {
            "climate0" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Increased",
                2 : "Decreased",
                3 : "Kept the same",
            },
            "climate1" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Should be doing more",
                2 : "Should be doing less",
                3 : "Is currently doing the right amount",
            },
            "crime0" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Increased",
                2 : "Decreased",
                3 : "Kept the same",
            },
            "econ0" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Approve",
                2 : "Disapprove",
            },
            "econ1" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Waste a lot",
                2 : "Waste some",
                3 : "Don\'t waste very much",
            },
            "econ2" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Increased",
                2 : "Decreased",
                3 : "Kept the same", 
            },
            "econ3" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Increased",
                2 : "Decreased",
                3 : "Kept the same", 
            },
            "econ4" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Very good",
                2 : "Good",
                3 : "Neither good nor bad",
                4 : "Bad",
                5 : "Very bad",
            },
            "econ5" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Gotten better",
                2 : "Stayed about the same",
                3 : "Gotten worse",
            },
            "econ6" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Extremely worried",
                2 : "Very worried",
                3 : "Moderately worried",
                4 : "A little worried",
                5 : "Not at all worried",
            },
            "econ7" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Increased",
                2 : "Decreased",
                3 : "Kept the same",
            },
            "gender0" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Gay and lesbian couples should be allowed to legally marry",
                2 : "Gay and lesbian couples should be allowed to form civil unions but not legally marry",
                3 : "There should be no legal recognition of gay or lesbian couples\' relationship",
            },
            "intl_affairs0" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Approve",
                2 : "Disapprove",
            },
            "intl_affairs1" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Approve",
                2 : "Disapprove",
            },
            "intl_affairs2" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Extremely willing",
                2 : "Very willing",
                3 : "Moderately willing",
                4 : "A little willing",
                5 : "Not at all willing",
            },
            "mental_hlth0" : {
                -9 : "Refused",
                -5 : "Interview breakoff (sufficient parital IW)",
                1 : "All the time",
                2 : "Often",
                3 : "Sometimes",
                4 : "Rarely",
                5 : "Never",
            },
            "physical_hlth0" : {
                -9 : "Refused",
                -5 : "Interview breakoff (sufficient parital IW)",
                1 : "Yes",
                2 : "No",
            },
            "politics0" : {
                -9 : "Refused",
                1 : "Very much interested",
                2 : "Somewhat interested",
                3 : "Not much interested",    
            },
            "politics1" : {
                -9 : "Refused",
                -1 : "Inapplicable",
                1 : "Very strongly",
                2 : "Moderately strongly",
                3 : "A little strongly",
            },
            "politics2" : {
                -9 : "Refused",
                -8 : "Don\'t know",
                1 : "Run by a few big interests",
                2 : "For the benefit of all people",
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
            "climate0" : "What about protecting the environment? Should federal spending on protecting the environment be increased, decreased, or kept the same?",
            "climate1" : "Do you think the federal government should be doing more about rising temperatures, should be doing less, or is it currently doing the right amount?",
            "crime0" : "What about dealing with crime? Should federal spending on dealing with crime be increased, decreased, or kept the same?",
            "econ0" : "Do you approve or disapprove of the way Donald Trump is handling the economy?",
            "econ1" : "Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or don\'t waste very much of it?",
            "econ2" : "What about Social Security? Should federal spending on Social Security be increased, decreased, or kept the same?",
            "econ3" : "What about aid to the poor? Should federal spending on aid to the poor be increased, decreased, or kept the same?",
            "econ4" : "What do you think about the state of the economy these days in the United States? Would you say the state of the economy is very good,good, neither good nor bad, bad, or very bad?",
            "econ5" : "Now thinking about the economy in the country as a whole, would you say that over the past year the nation\'s economy has gotten better, stayed about the same, or gotten worse?",
            "econ6" : "So far as you and your family are concerned, how worried are you about your current financial situation?",
            "econ7" : "What about welfare programs? Should federal spending on welfare programs be increased, decreased, or kept the same?",
            "gender0" : "Which comes closest to your view? You can just tell me the number of your choice.",
            "intl_affairs0" : "Do you approve or disapprove of the way Donald Trump is handling relations with foreign countries?",
            "intl_affairs1" : "Do you approve or disapprove of the way Donald Trump is handling immigration?",
            "intl_affairs2" : "How willing should the United States be to use military force to solve international problems?",
            "mental_hlth0" : "In the past week, how often has your sleep been restless?",
            "physical_hlth0" : "Do you presently have any kind of health insurance?",
            "politics0" : "Some people don\'t pay much attention to political campaigns. How about you? Would you say that you have been very much interested, somewhat interested or not much interested in the political campaigns so far this year?",
            "politics1" : "How strongly do you feel that voting is a duty?",
            "politics2" : "Would you say the government is pretty much run by a few big interests looking out for themselves or that it is run for the benefit of all the people?"
        }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = AnesSurvey()
