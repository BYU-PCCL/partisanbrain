from parent_dir import Survey, UserInterventionNeededError
import pandas as pd
import os

class AddhealthSurvey(Survey):

    def __init__(self):
        super().__init__()

    def download_data(self):
        directory = 'survey_data/addhealth/'
        filename = '21600-0032-Data.sav'
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return pd.read_spss(filepath)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
            raise UserInterventionNeededError("Download the data from this link (requires a login) https://www.icpsr.umich.edu/web/ICPSR/studies/21600")

    def modify_data(self, df):
        dem_dict = {
            "age": "H5OD1Y",
            "gender": "H5OD2A",
            "education": "H5OD11",
            "ideology": "H5SS9",
            "income": "H5EC1",
            "religion": "H5RE1",
            "race_ethnicity": ['H5OD4A', 'H5OD4B', 'H5OD4C', 'H5OD4D', 'H5OD4E', 'H5OD4F', 'H5OD4G'],
            "region": "W5REGION",
            "marital_status": "H5HR1",
        }

        dem_cols = sum([col if isinstance(col, list) else [col] for col in dem_dict.values()], [])

        # Drop any rows that don't have all the demographics
        mod_df = df.dropna(subset=dem_cols)

        # Drop rows with more than one ethnicity
        race_cols = dem_dict['race_ethnicity']
        mask = (mod_df[race_cols].sum(axis=1) < 2)
        mod_df = mod_df[mask]

        # Reverse one-hot-encode the ethnicities
        mod_df['race_ethnicity'] = sum([i*mod_df[code].values for i, code in enumerate(race_cols, 1)])
        no_ethnicity_mask = (mod_df['race_ethnicity'] != 0)
        mod_df = mod_df[no_ethnicity_mask]

        # Rename the demographics
        rename_dict = {code: dem for dem, code in dem_dict.items() if not isinstance(code, list)}
        mod_df = mod_df.rename(columns=rename_dict)

        # Turn the birth year into age TODO double check that respondants took survey in 2018
        survey_year = 2018
        mod_df['age'] = survey_year - mod_df['age']

        # Get the DVs now TODO two of these aren't in the df
        dvs_dict = {
            'H5ID13':'psych_emo_council',
            'H5ID21':'fast_food',
            'H5ID23':'watch_tv',
            'H5ID25':'roller_blade',
            'H5ID24':'bicycle',
            'H5ID27':'individual_sports',
            'H5ID28':'golf',
            'H5TO1':'health',
            'H5TO5':'has_smoked',
            'H5TO17':'quit_alcohol',
            'H5EL6P':'was_obese',
            'H5SS0A':'sad_boy_vibes',
        #     'H5CJ1D',
            'H5SS3C':'open_up_to_family',
            'H5WP1':'mother_in_jail',
            'H5WP15':'father_in_jail',
            'H5RE4':'prayer_in_private',
        #     'H5CJ1F',
            'H5FT3':'track_health',
            'H5MN8':'suicidal',
            'H5CJ3':'was_arrested',
            'H5MN7':'unfair_police_questioning',
            # 'H5CJ1E':'pulled_a_knife_or_gun',
        }

        # Rename the DVs
        mod_df = mod_df.rename(columns=dvs_dict)

        # Drop everything else
        cols = list(dvs_dict.values()) + list(dem_dict.keys())
        mod_df = mod_df[cols]

        return mod_df

    def get_dv_questions(self):
        return {
            'psych_emo_council':'In the past 12 months, have you received psychological or emotional counseling?',
            'fast_food':"In the past 7 days, how many times did you eat food from a fast food restaurant, such as McDonald's, Burger King, Wendy's, Arby's, Pizza Hut, Taco Bell, or Kentucky Fried Chicken or a local fast food restaurant?",
            'watch_tv':"In the past 7 days, how many hours did you watch television, movies or videos, including DVDs or music videos?",
            'roller_blade':"In the past 7 days, how many times did you roller blade, roller skate, downhill ski, snowboard, play racquet sports, or do aerobics?",
            'bicycle':"In the past 7 days, how many times did you bicycle, skateboard, dance, hike, hunt, or do yard work?",
            'individual_sports':"In the past 7 days, how many times did you participate in individual sports such as running, wrestling, swimming, cross-country skiing, cycle racing, martial arts, or in strenuous team sports such as football, soccer, basketball, lacrosse, rugby, field hockey, or ice hockey?",
            'golf':"In the past 7 days, how many times did you play golf, go fishing or bowling, or play softball or baseball?",
            'health':"Health",
            'has_smoked':"Have you ever smoked or used tobacco?",
            'quit_alcohol':"Has there ever been a period of time when you wanted to quit or cut down on your use of alcohol, or thought you should quit?",
            'was_obese':"When you were growing up, before age 16, did a doctor, nurse, or other health care provider tell you or your parents/adult caregivers that you have or had any of the following? Obesity",
            'sad_boy_vibes':"The next questions are about your feelings. How often was each of the following things true during the past 7 days? Q0A. During the past 7 days, I felt that I could not shake off the blues, even with help from my family and friends.",
            #     'H5CJ1D':"Indicate whether or not you did any of these things in the past 12 months. Q1D. Get into a serious physical fight",
                'open_up_to_family':"For each of the following individuals or groups of people indicate whether or not you can open up to them if you need to talk about your worries. Q3C. Other family members",
            'mother_in_jail':"This section asks questions about your parents and people who may have acted like parents to you. Q1. Has your biological mother ever spent time in jail or prison?",
                'father_in_jail':"Has your biological father ever spent time in jail or prison?",
            'prayer_in_private':"How often do you pray privately, that is, when you are alone in places other than a church, synagogue, temple, mosque, or religious assembly?",
            #     'H5CJ1F':"Indicate whether or not you did any of these things in the past 12 months. Q1F. Shoot or stab someone",
            'track_health':"Do you use any smartphone apps to track or manage your health?",
            'suicidal':"These next questions are about suicide. They refer to the past 12 months. Q8. During the past 12 months, have you ever seriously thought about committing suicide?",
            'was_arrested':"Have you ever been arrested?",
            'unfair_police_questioning':"Have you ever been unfairly stopped, searched, or questioned by the police?",
            # 'pulled_a_knife_or_gun':'Indicate whether or not you did any of these things in the past 12 months. Q1E. Pulled a knife or gun on someone',
        }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = AddhealthSurvey()
