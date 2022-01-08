from parent_dir import DatasetFactory
# import pandas as pd
import numpy as np
from survey_classes import CcesSurvey


class CcesFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        print("aaaaa")
        print(df)
        print(df.unique())
        pass

    def get_templates(self):
        templates = {
            "epa_power": {
                "i_believe": (lambda row: (f"{self.make_backstory1}\n I believe the Environmental Protection Agency should hold power to regulate Carbon Dioxide emissions, yes or no"), {"yes": "yes",
                                                               "no": "no"}),
                
                # More templates here
            },
        }
        return templates

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
                'trade school': 'I attended trade school. ',
                'professional school degree': 'I have a professional school degree. ',
                "Bachelor's degree": "I have a Bachelor's degree. ",
                'less than high school': "I didn't finish high school. ",
                'high school graduate': "I graduated high school. ",
                "Master's degree": "I have a Master's degree. ",
                'associate degree': "I have an associate's degree. ",
                'some college but no degree': "I attended some college. ",
                # no default, should be exhaustive
            },
            #income
            # i make between blank and blank per year
            'religion': {
                np.nan: '',
                'nan': '',
                'default': f'''I am {row['gender']}. ''',
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


if __name__ == "__main__":
    survey = CcesSurvey()
    factory = CcesFactory(survey)
