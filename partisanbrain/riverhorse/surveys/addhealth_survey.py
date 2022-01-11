# Author: Alex Shaw

from ..survey import Survey, UserInterventionNeededError
import pandas as pd
import os


class AddhealthSurvey(Survey):
    def __init__(self, force_recreate=False):
        super().__init__(force_recreate=force_recreate)

    def download_data(self):
        directory = "survey_data/addhealth/"
        filename = "21600-0022-Data.sav"
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            return pd.read_spss(filepath)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
            raise UserInterventionNeededError(
                "Download the data from this link (requires a login) https://www.icpsr.umich.edu/web/ICPSR/studies/21600"
            )

    def modify_data(self, df):
        dem_dict = {
            "H4OD1Y": "age",
            "BIO_SEX4": "gender",
            "H4ED2": "education",
            "H4DA28": "ideology",
            "H4EC1": "income",
            "H4RE1": "religion",
            "H4IR4": "race_ethnicity",
            "H4TR1": "marital_status",
        }

        dem_codes = [code for code in dem_dict.keys()]

        # Drop any rows that don't have all the demographics
        mod_df = df.dropna(subset=dem_codes)

        # Rename the demographics
        mod_df = mod_df.rename(columns=dem_dict)

        # Get the DVs now TODO two of these aren't in the df
        dvs_dict = {
            "H4DS20": "shot_or_stabbed",
            "H4CJ1": "arrested", # yn
            "H4DS11": "physical_fight", # times
            "H4CJ10": "convicted_of_charges", # No, 
            "H4DS5": "sell_drugs", # times
            "H4HS9": "counseling", # yn
            "H4MH19": "sadness_family", # occasion
            "H4PE6": "worrying", # agree
            "H4SE2": "suicide",
            "H4PE23": "optimism", # agree
            "H4MH24": "happiness", # occasion
            "H4GH8": "fast_food", # this is a weird one
            "H4DA1": "hours_of_tv", # weird one too
            "H4DA5": "individual_sports", # times7
            "H4TO1": "smoked_cigarette", # yn
            "H4MA3": "physical_child_abuse", # abuse
            "H4TO34": "age_of_first_drink", # big range of numbers
            "H4ID8": "car_accidents", # yn
            "H4TO33": "drinking", # yn
            "H4RE10": "prayer_in_private",
        }

        # Rename the DVs
        mod_df = mod_df.rename(columns=dvs_dict)

        # Drop everything else
        cols = list(dem_dict.values()) + list(dvs_dict.values())
        mod_df = mod_df[cols]

        return mod_df

    def get_dv_questions(self):
        return {
            "shot_or_stabbed": "Which of the following things happened in the past 12 months: You shot or stabbed someone?",
            "arrested": "The next questions are about arrests and convictions. Have you ever been arrested?",
            "physical_fight": "In the past 12 months, how often did you get into a serious physical fight?",
            "convicted_of_charges": "Have you ever been convicted of or pled guilty to any charges other than a minor traffic violation?",
            "sell_drugs": "In the past 12 months, how often did you sell marijuana or other drugs?",
            "counseling": "In the past 12 months, have you received psychological or emotional counseling?",
            "sadness_family": "(During the past seven days:) You could not shake off the blues, even with help from your family and your friends.",
            "worrying": "How much do you agree with each statement about you as you generally are now, not as you wish to be in the future? I worry about things.",
            "suicide": "During the past 12 months, how many times have you actually attempted suicide?",
            "optimism": "How much do you agree with each statement about you as you generally are now, not as you wish to be in the future? Overall, I expect more good things to happen to me than bad.",
            "happiness": "(During the past seven days:) You felt happy.",
            "fast_food": "How many times in the past seven days did you eat food from a fast food restaurant, such as McDonald's, Burger King, Wendy's, Arby's, Pizza Hut, Taco Bell, or Kentucky Fried Chicken or a local fast food restaurant?",
            "hours_of_tv": "In the past seven days, how many hours did you watch television or videos, including VHS, DVDs or music videos?",
            "individual_sports": "In the past seven days, how many times did you participate in individual sports such as running, wrestling, swimming, cross-country skiing, cycle racing, or martial arts?",
            "smoked_cigarette": "The next questions are about your experiences with tobacco, alcohol, and drugs. Remember, your answers will not be linked to you. Have you ever smoked an entire cigarette?",
            "physical_child_abuse": "Before your 18th birthday, how often did a parent or adult caregiver hit you with a fist, kick you, or throw you down on the floor, into a wall, or down stairs?",
            "age_of_first_drink": "How old were you when you first had an alcoholic drink? By drink, we mean a glass of wine, a can or bottle of beer, a wine cooler, a shot glass of liquor, or a mixed drink, not just sips or tastes from someone else's drink.",
            "car_accidents": "In the past 12 months, were you involved in a motor vehicle accident?",
            "drinking": "Have you had a drink of beer, wine, or liquor more than two or three times? Do not include sips or tastes from someone else's drink.",
            "prayer_in_private": "How often do you pray privately, that is, when you're alone in places other than a church, synagogue, temple, mosque, or religious assembly?",
        }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = AddhealthSurvey()
