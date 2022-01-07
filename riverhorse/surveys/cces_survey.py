from parent_dir import Survey, UserInterventionNeededError
import os
import pandas as pd


class CcesSurvey(Survey):
    def __init__(self, force_recreate=False):
        super().__init__(force_recreate=force_recreate)

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
            "CC20_333a": "co2_emissions",
            "CC20_333b": "renewable_fuels",
            "CC20_333c": "clean_air",
            "CC20_305_9": "crime_victim",
            "CC20_307": "police_feel",
            "CC20_334b": "body_cameras",
            "CC20_334c": "increase_police",
            "CC20_334d": "decrease_police",
            "CC20_302": "nations_economy",
            "CC20_303": "income_change",
            "union": "labor_union",
            "trans": "gender_change",
            "sexuality": "sexuality",
            "CC20_331a": "illegal_immigrants",
            "CC20_331b": "border_patrols",
            "CC20_331c": "withhold_police_funds",
            "CC20_331d": "reduce_immigration",
            "CC20_441g": "whites_understand_blacks",
            "CC20_441b": "slavery_influence",
            "CC20_441e": "resent_white_denial",
        }

        mod_df = mod_df.rename(columns=dvs_dict)

        cols = list(demo_dict.values()) + list(dvs_dict.values())

        mod_df = mod_df[cols]

        return mod_df

    def get_dv_questions(self):
        return {
            "co2_emissions": "Give the Environmental Protection Agency power to regulate Carbon Dioxide emissions",
            "renewable_fuels": "Require that each state use a minimum amount of renewable fuels (wind, solar, and hydroelectric) in the generation of electricity even if electricity prices increase a little",
            "clean_air": "Strengthen the Environmental Protection Agency enforcement of the Clean Air Act and Clean Water Act even if it costs U.S. jobs",
            "crime_victim": "Over the past year have you been the victim of a crime?",
            "police_feel": "Do the police make you feel...",
            "body_cameras": "Require police officers to wear body cameras that record all of their activities while on duty.",
            "increase_police": "Increase the number of police on the street by 10 percent, even if it means fewer funds for other public services.",
            "decrease_police": "Decrease the number of police on the street by 10 percent, and increase funding for other public services",
            "nations_economy": "Would you say that OVER THE PAST YEAR the nation's economy has ...",
            "income_change": "OVER THE PAST YEAR, has your household's annual income…?",
            "labor_union": "Are you a member of a labor union?",
            "gender_change": "Have you ever undergone any part of a process (including any thought or action) to change your gender / perceived gender from the one you were assigned at birth? This may include steps such as changing the type of clothes you wear, name you are known by or undergoing surgery.",
            "sexuality": "Which of the following best describes your sexuality?",
            "illegal_immigrants": "Grant legal status to all illegal immigrants who have held jobs and paid taxes for at least 3 years, and not been convicted of any felony crimes.",
            "border_patrols": "Increase the number of border patrols on the US-Mexican border.",
            "withhold_police_funds": "Withhold federal funds from any local police department that does not report to the federal government anyone they identify as an illegal immigrant.",
            "reduce_immigration": "Reduce legal immigration by 50 percent over the next 10 years by eliminating the visa lottery and ending family-based migration.",
            "whites_understand_blacks": "Whites do not go to great lengths to understand the problems African Americans face.",
            "slavery_influence": "Generations of slavery and discrimination have created conditions that make it difficult for blacks to work their way out of the lower class.",
            "resent_white_denial": "I resent when Whites deny the existence of racial discrimination.",
        }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = CcesSurvey()
