"""@author: Sophie Gee: PCCL Princess"""

from ..survey import Survey
from ..constants import DEMOGRAPHIC_COLNAMES, SURVEY_DATA_PATH

import os
import pandas as pd
import requests
import zipfile
import numpy as np
from pdb import set_trace as breakpoint

SURVEY_URL = "https://electionstudies.org/anes_timeseries_2020_csv_20210719/"


class AnesSurveySophie(Survey):
    def __init__(self, force_recreate=False):
        super().__init__(force_recreate=force_recreate)

    def download_data(self):
        survey_data_csv_path = f"{SURVEY_DATA_PATH}/anes_survey_sophie/raw.csv"
        survey_data_zip_path = f"{SURVEY_DATA_PATH}/anes_survey_sophie/raw.csv"
        if not os.path.exists(survey_data_csv_path):
            # download the data from the link: https://electionstudies.org/anes_timeseries_2020_csv_20210719/
            fzip = requests.get(SURVEY_URL)
            # save the zip file
            with open(survey_data_zip_path, "wb") as f:
                f.write(fzip.content)

            # unzip the file
            with zipfile.ZipFile(survey_data_zip_path, "r") as zip_ref:
                zip_ref.extractall(f"{SURVEY_DATA_PATH}/anes_survey_sophie")
            # rename the csv file
            os.rename(
                f"{SURVEY_DATA_PATH}/anes_survey_sophie/anes_timeseries_2020_csv_20210719.csv",
                f"{SURVEY_DATA_PATH}/anes_survey_sophie/raw.csv",
            )

        df = pd.read_csv(survey_data_csv_path)
        return df

    def modify_data(self, df):
        # Rename demographic columns.
        # breakpoint()
        mod_df = df.rename(
            columns={
                "V200001": "id",
                "V201507x": "age",
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
        mod_df = mod_df.rename(
            columns={
                "V201103": "vote_2016",
            }
        )

        # Drop all columns that are not DV questions or demographic columns.
        column_lst = DEMOGRAPHIC_COLNAMES + [
            "vote_2016",
        ]
        mod_df = mod_df[column_lst]
        mod_df["vote_2016_raw"] = mod_df["vote_2016"]
        mod_df["question"] = "Four years ago, in 2016, Hillary Clinton ran on the Democratic ticket against Donald Trump for the Republicans. Do you remember for sure whether or not you voted in that election? Which one did you vote for?"
        # Clean data by renaming codes to natural langauge responses using codebook.
        # note taken from old survey code - income and age are not coded.
        demographics_dict = {
            "gender": {
                1: "Male",
                2: "Female",
            },
            "party": {
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
                1: "Married: spouse present",
                2: "Married: spouse absent {VOL - video/phone only}",
                3: "Widowed",
                4: "Divorced",
                5: "Separated",
                6: "Never married",
            },
        }

        answers_dict = {
            "vote_2016": {
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
            mod_df = mod_df.dropna()
        return mod_df

    def get_dv_questions(self):
        return {
            "vote_2016": "Four years ago, in 2016, Hillary Clinton ran on the Democratic ticket against Donald Trump for the Republicans. Do you remember for sure whether or not you voted in that election? Which one did you vote for?",
        }
def for_chris(row):
    # create a generic backstory using values in the demographics
    return (
        f"Age: {int(row['age'])} \n"
        f"Gender: {row['gender']} \n"
        f"Political party: {row['party']} \n"
        f"Ideology: {row['ideology']} \n"
        f"Education: {row['education']} \n"
        f"Income: {row['income']} \n"
        f"Religion: {row['religion']} \n"
        f"Race/ Ethnicity: {row['race_ethnicity']} \n"
        f"Region: {row['region']} \n"
        f"Marital status {row['marital_status']}: \n"
        f"Question: Four years ago, in 2016, Hillary Clinton ran on the Democratic ticket against Donald Trump for the Republicans. Do you remember for sure whether or not you voted in that election? Which one did you vote for? \n"
        f"1. Hillary Clinton \n"
        f"2. Donald Trump \n"
        f"5. Other (specify) \n"
        f"Answer:"
    )

if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = AnesSurveySophie()
    df = s.download_data()
    df_mod = s.modify_data(df)
    # print(for_chris(df_mod.iloc[7]))
    # print(df_mod.iloc[7]["vote_2016"])
    s.modify_data(df).to_csv("ANES_for_josh.csv")