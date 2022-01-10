# Author: Alex Shaw

from ..survey import Survey, UserInterventionNeededError
import os
import pandas as pd


demo_drop_dict = {
    "race_ethnicity": [7],
    "ideology": [8],
    "income": [97],
    "religion": [12],
    "party": [4, 9, 13],
}

dv_drop_dict = {
    "whites_understand_blacks": [8, 9],
    "slavery_influence": [8, 9],
    "resent_white_denial": [8, 9],
    "nations_economy": [6],
    "gender_change": [3],
    "sexuality": [5, 6],
}


class CcesSurvey(Survey):
    def __init__(self, force_recreate=False):
        super().__init__(force_recreate=force_recreate)

    def _map_answers(self, df):
        mod_df = df

        identity = lambda x: x
        survey_year = 2020
        demo_map = {
            "age": identity,
            "gender": {
                1: "Male",
                2: "Female",
            },
            "party": {
                1: "Conservative Party",
                2: "Constitution Party",
                3: "Democratic Party",
                5: "Green Party",
                6: "Independent",
                7: "Libertarian Party",
                8: "No Party Affiliation",
                10: "Reform Party",
                11: "Republican Party",
                12: "Socialist Party",
                14: "Working Families Party",
            },
            "education": {
                1: "Did not graduate from high school",
                2: "High school graduate",
                3: "Some college, but no degree (yet)",
                4: "2-year college degree",
                5: "4-year college degree",
                6: "Postgraduate degree (MA, MBA, MD, JD, PhD, etc.)",
            },
            "ideology": {
                1: "Very Liberal",
                2: "Liberal",
                3: "Somewhat Liberal",
                4: "Middle of the Road",
                5: "Somewhat Conservative",
                6: "Conservative",
                7: "Very Conservative",
            },
            "income": {
                1: "Less than $10,000",
                2: "$10,000 - $19,999",
                3: "$20,000 - $29,999",
                4: "$30,000 - $39,999",
                5: "$40,000 - $49,999",
                6: "$50,000 - $59,999",
                7: "$60,000 - $69,999",
                8: "$70,000 - $79,999",
                9: "$80,000 - $99,999",
                10: "$100,000 - $119,999",
                11: "$120,000 - $149,999",
                12: "$150,000 - $199,999",
                13: "$200,000 - $249,999",
                14: "$250,000 - $349,999",
                15: "$350,000 - $499,999",
                16: "$500,000 or more",
            },
            "religion": {
                1: "Protestant",
                2: "Roman Catholic",
                3: "Mormon",
                4: "Eastern or Greek Orthodox",
                5: "Jewish",
                6: "Muslim",
                7: "Buddhist",
                8: "Hindu",
                9: "Atheist",
                10: "Agnostic",
                11: "Nothing in particular",
            },
            "race_ethnicity": {
                1: "White",
                2: "Black or African-American",
                3: "Hispanic or Latino",
                4: "Asian or Asian-American",
                5: "Native American",
                6: "Two or more races",
                8: "Middle Eastern",
            },
            "region": {
                1: "Northeast",
                2: "Midwest",
                3: "South",
                4: "West",
            },
            "marital_status": {
                1: "Married",
                2: "Separated",
                3: "Divorced",
                4: "Widowed",
                5: "Never married",
                6: "Domestic/civil partnership",
            },
        }

        for col, mapping in demo_map.items():
            mod_df[col] = mod_df[col].map(mapping)

        support_oppose_dict = {
            1: "Support",
            2: "Oppose",
        }
        agreement_dict = {
            1: "Strongly agree",
            2: "Somewhat agree",
            3: "Neither agree nor disagree",
            4: "Somewhat disagree",
            5: "Strongly disagree",
        }
        yes_no_dict = {
            1: "yes",
            2: "no",
        }
        dv_map = {
            "co2_emissions": support_oppose_dict,
            "renewable_fuels": support_oppose_dict,
            "clean_air": support_oppose_dict,
            "crime_victim": yes_no_dict,
            "police_feel": {
                1: "Mostly safe",
                2: "Somewhat safe",
                3: "Somewhat unsafe",
                4: "Mostly unsafe",
            },
            "body_cameras": support_oppose_dict,
            "increase_police": support_oppose_dict,
            "decrease_police": support_oppose_dict,
            "nations_economy": {
                1: "Gotten much better",
                2: "Gotten somewhat better",
                3: "Stayed about the same",
                4: "Gotten somewhat worse",
                5: "Gotten much worse",
            },
            "income_change": {
                1: "Increased a lot",
                2: "Increased somewhat",
                3: "Stayed about the same",
                4: "Decreased somewhat",
                5: "Decreased a lot",
            },
            "labor_union": {
                1: "Yes, I am currently a member of a labor union",
                2: "I formerly was a member of a labor union",
                3: "I am not now, nor have I been, a member of a labor union",
            },
            "gender_change": yes_no_dict,
            "sexuality": {
                1: "Heterosexual/straight",
                2: "Lesbian/gay woman",
                3: "Gay man",
                4: "Bisexual",
            },
            "illegal_immigrants": support_oppose_dict,
            "border_patrols": support_oppose_dict,
            "withhold_police_funds": support_oppose_dict,
            "reduce_immigration": support_oppose_dict,
            "whites_understand_blacks": agreement_dict,
            "slavery_influence": agreement_dict,
            "resent_white_denial": agreement_dict,
        }

        for col, mapping in dv_map.items():
            mod_df[col] = mod_df[col].map(mapping)

        return mod_df

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
            "region": "region",
            "marstat": "marital_status",
        }
        demo_cols = list(demo_dict.values())

        mod_df = df.rename(columns=demo_dict)
        mod_df = mod_df.dropna(subset=demo_cols)

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

        mod_df = self._map_answers(mod_df)

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
