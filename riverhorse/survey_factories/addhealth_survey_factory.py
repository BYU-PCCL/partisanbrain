from parent_dir import DatasetFactory
from datetime import date
from collections import defaultdict
from survey_classes import AddhealthSurvey
import pandas as pd
import numpy as np


class AddhealthFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    # Bacstory 1 - "I am" statements, how would you answer?
    # Backstory 2 - Imagine you were this person; how would you imagine they answer?
    # Backstory 3 - 
    def make_backstory1(self, row):
        '''
        list style backstory dropping nans
        '''
        dictionary = {
            'age': {
                '-1': '',
                'nan': '',
                'default': f'''I am {row['age']} years old. ''',
            },
            'gender': {
                np.nan: '',
                'nan': '',
                'default': f'''I am {row['gender']}. ''',
            },
            'party': {
                np.nan: '',
                'nan': '',
                'None/Independent': 'I am an independent. ',
                'default': f'''I am a {row['party']}. ''',
            },
            'ideology': {
                np.nan: '',
                'nan': '',
                'default': f'''Ideologically, I am {str(row['ideo']).strip().lower()}. ''',
            },
            'education': {
                np.nan: '',
                'nan': '',
                # 'trade school': 'I attended trade school. ',
                # 'professional school degree': 'I have a professional school degree. ',
                # "Bachelor's degree": "I have a Bachelor's degree. ",
                'Some high school': "I attended some high school. ",
                # 'less than high school': "I didn't finish high school. ",
                'High school graduate': "I graduated high school. ",
                # "Master's degree": "I have a Master's degree. ",
                # 'associate degree': "I have an associate's degree. ",
                # 'Completed vocational/technical training (after high scho' : 
                    # "I completed vocational or technical training after high school.",
                'Some college': "I attended some college. ",
                # 'some college but no degree': "I attended some college. ",
                '8th grade or less' : "I completed up to 8th grade or less. ",
                'Some post baccalaureate professional education' : 
                    "I have some post baccalaureate professional education. ",
                "Completed college (bachelor's degree)" : "I have a Bachelor's degree. "
                # no default, should be exhaustive
            },
            #income
            # i make between blank and blank per year
            'religion': {
                np.nan: '',
                'nan': '',
                'default': f'''I am {row['religion']}. ''',
                'Undifferentiated Protestant': 'Religiously, I am protestant. ',
                'Undifferentiated Protstant': 'Religiously, I am protestant. ',
                'Other Christian': 'Religiously, I am Christian. ',
                'default': f'Religiously, I identify as {row["religion"]}. ',
            },
            'race': {
                np.nan: '',
                'nan': '',
                'default': f'''I am {row['race']}. ''',

                'default': f'Racially, I identify as {row["race"]}. ',
            },
            #region
            'marital': {
                np.nan: '',
                'nan': '',
                'never married': 'I have never married. ',
                'default': f'''I am {row['marital']}. ''',
            },
            
            
            
            
        }
        backstory = ''
        for key in dictionary.keys():
            val = row[key]
            if val in dictionary[key]:
                backstory += dictionary[key][val]
            else:
                backstory += dictionary[key]['default']
        return backstory

    
    def modify_data(self, df):
        # file = open("data.txt","w+")
        # file.write(df)
        # file.close()
        # for x in df:
        #     print(y)

        # rename rows?
        mod_df_dict = defaultdict(list)
        for _, row in df.iterrows():
            mod_df_dict['age'].append(self.get_age(row['age']))
            mod_df_dict['gender'].append(row['gender'][4:])
            mod_df_dict['education'].append(row['education'][4:])
            mod_df_dict['ideology'].append(row['ideology'][4:])
            mod_df_dict['income'].append(row['income'][4:])
            mod_df_dict['religion'].append(row['religion'][4:])
            mod_df_dict['race_ethnicity'].append(row['race_ethnicity'][4:])
            mod_df_dict['marital_status'].append(row['marital_status'][4:])

            mod_df_dict['shot_or_stabbed'].append(row['shot_or_stabbed'][4:])
            mod_df_dict['arrested'].append(row['arrested'][4:])
            mod_df_dict['physical_fight'].append(row['physical_fight'][4:])
            mod_df_dict['convicted_of_charges'].append(row['convicted_of_charges'][4:])
            mod_df_dict['sell_drugs'].append(row['sell_drugs'][4:])
            mod_df_dict['counseling'].append(row['counseling'][4:])
            mod_df_dict['sadness_family'].append(row['sadness_family'][4:])
            mod_df_dict['worrying'].append(row['worrying'][4:])
            mod_df_dict['suicide'].append(row['suicide'][4:])
            mod_df_dict['optimism'].append(row['optimism'][4:])
            mod_df_dict['happiness'].append(row['happiness'][4:])
            mod_df_dict['fast_food'].append(row['fast_food'][4:])
            mod_df_dict['hours_of_tv'].append(row['hours_of_tv'][4:])
            mod_df_dict['individual_sports'].append(row['individual_sports'][4:])
            mod_df_dict['smoked_cigarette'].append(row['smoked_cigarette'][4:])
            mod_df_dict['physical_child_abuse'].append(row['physical_child_abuse'][4:])
            mod_df_dict['age_of_first_drink'].append(row['age_of_first_drink'][4:])
            mod_df_dict['car_accidents'].append(row['car_accidents'][4:])
            mod_df_dict['drinking'].append(row['drinking'][4:])
            mod_df_dict['prayer_in_private'].append(row['prayer_in_private'][4:])
            
        new_df = pd.DataFrame(mod_df_dict, index=df.index)

            


        print(df)
        pass

    def get_age(self, year):
        return date.today().year - year

    def get_tokens_yn(self):
        return {
            'No': ['(0) No', 'No'], 
            'Yes': ['(1) Yes', 'Yes']}

    def get_templates(self):
        templates = {
            "shot_or_stabbed": {
                "personal1": (lambda row: (f"{self.make_backstory1}\n Have you shot or stabbed someone in the past 12 months?"), 
                self.get_tokens_yn),
                "personal2": (lambda row: (f"{self.make_backstory1}\n Is it true that you have shot or stabbed someone in the past 12 months?"), 
                self.get_tokens_yn),
                "personal3": (lambda row: (f"{self.make_backstory1}\n In the past 12 months, have you shot or stabbed someone?"), 
                self.get_tokens_yn),
                "personal4": (lambda row: (f"{self.make_backstory1}\n In the past year, has someone been shot or stabbed by you?"), 
                self.get_tokens_yn),
                "personal5": (lambda row: (f"{self.make_backstory1}\n Have you shot or stabbed at least one person in the past year?"), 
                self.get_tokens_yn),
                "personal3": (lambda row: (f"{self.make_backstory1}\n In the past 12 months, have you shot or stabbed someone?"), 
                self.get_tokens_yn),
                
                # More templates here
            },
        }
        return templates


if __name__ == "__main__":
    factory = AddhealthFactory(AddhealthSurvey())
