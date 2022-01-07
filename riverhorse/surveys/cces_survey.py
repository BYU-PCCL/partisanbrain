from parent_dir import Survey, UserInterventionNeededError
import os
import pandas as pd


class CcesSurvey(Survey):
    def __init__(self):
        super().__init__()

    def download_data(self):
        directory = "survey_data/cces/"
        filename = "CES20_Common_OUTPUT_vv.csv"
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
            raise UserInterventionNeededError(
                "Download the data from this link https://dataverse.harvard.edu/file.xhtml?fileId=4949558&version=3.0"
            )

    def modify_data(self, df):
        demo_dict = {
            "birthyr": "age",
            "gender": "gender",
            "CL_party": "party",
            "educ": "education",
            "CC20_340a": "ideology",
            "faminc_new": "income",
            "religpew": "religion",
            "race": "race_ethnicity",
            "region_post": "region",
            "marstat": "marital_status",
        }
        demo_cols = list(demo_dict.values())

        mod_df = df.rename(columns=demo_dict)
        mod_df = mod_df.dropna(subset=demo_cols)

        # Turn age into an actual age
        survey_year = 2020
        mod_df["age"] = survey_year - mod_df["age"]

        dvs_dict = {
            "CC20_302": "nations_economy",
            "CC20_303": "income_change",
            "CC20_327grid": "health_care_policies",
            "CC20_330grid": "gun_regulation",
            "CC20_338grid": "trade_tariffs",
            "CC20_350grid": "support_congress",
            "union": "labor_union",
            "trans": "change_gender",
            # "Page 71": "religion",
        }

        mod_df = mod_df.rename(columns=dvs_dict)

        cols = list(demo_dict.values()) + list(dvs_dict.values())
        print("region" in mod_df.columns)

        mod_df = mod_df[cols]

        return mod_df

    def get_dv_questions(self):
        return {
            "nations_economy": "Would you say that OVER THE PAST YEAR the nation's economy has ...",
            "income_change": "OVER THE PAST YEAR, has your household's annual income…?",
            "health_care_policies": "Thinking now about health care policy, would you support or oppose each of the following proposals?",
            "gun_regulation": "On the issue of gun regulation, do you support or oppose each of the following proposals?",
            "trade_tariffs": "On the issue of trade, do you support or oppose the following proposed tariffs?",
            "support_congress": "Over the past two years, Congress voted on many issues. Do you support each of the following proposals?",
            "labor_union": "Are you a member of a labor union?",
            "change_gender": "Have you ever undergone any part of a process (including any thought or action) to change your gender / perceived gender from the one you were assigned at birth? This may include steps such as changing the type of clothes you wear, name you are known by or undergoing surgery.",
        }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = CcesSurvey()
