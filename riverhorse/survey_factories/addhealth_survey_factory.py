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
            'ideology': {
                np.nan: '',
                'nan': '',
                'default': f'''Ideologically, I am {str(row['ideo']).strip().lower()}. ''',
            },
            'education': {
                np.nan: '',
                'nan': '',
                '8th grade or less' : "I completed up to 8th grade or less. ",
                'Some high school': "I attended some high school. ",
                'High school graduate': "I graduated high school. ",
                'Some vocational/technical training (after high school)': "I have completed some vocational/technical training",
                'Completed vocational/technical training (after high scho' : 
                    "I completed vocational or technical training after high school.",
                'Some college': "I attended some college. ",
                "Completed college (bachelor's degree)" : "I have a Bachelor's degree. ",
                "Some graduate school" : "I have attended some graduate school",
                "Completed a master's degree " : "I have a Master's degree",
                "Some graduate training beyond a master's degree" : "I have attended some graduate training beyond a master's degree",
                "Completed a doctoral degree " : "I have completed a doctoral degree",
                "Some post baccalaureate professional education" : "I have completed some post baccalaureate professional education",
                "Completed post baccalaureate professional education" : "I have completed post baccalaureate professional education"
                # no default, should be exhaustive
            },
            #income
            'income': {
                np.nan: '',
                'nan': '',
                "Don't know": '',
                'refused': '',
                'default': f'''My household income is {row['income']}. ''',
                # '$5,000 to $9,999' : ''
                # '$10,000 to $14,999' : 
                # '$15,000 to $19,999'
                # '$20,000 to $24,999'
                # '$25,000 to $29,999'
                # '$30,000 to $39,999'
                # '$40,000 to $49,999'
                # '$50,000 to $74,999'
                # '$75,000 to $99,999'
                # '$100,000 to $149,999'
                # '$150,000 or more'
                
            },
            'religion': {
                np.nan: '',
                'nan': '',
                'none/atheist/agnostic' : "I am either atheist, agnostic, or non-religious",
                'Protestant (Such as Assembly of God, Baptist, etc.)': 'Religiously, I identify as protestant',
                'Other Christian': 'Religiously, I am Christian. ',
                'Other' : "I am religious",
                # 'default': f'Religiously, I identify as {row["religion"]}. ',
                'default': f'''I am {row['religion']}. ''',
            },
            'race': {
                np.nan: '',
                'nan': '',
                'default': f'''I am {row['race']}. ''',
                # 'default': f'Racially, I identify as {row["race"]}. ',
            },
            #region
            'marital': {
                np.nan: '',
                'nan': '',
                # 'never married': 'I have never married. ',
                'default': f'''I have married {row['marital']}. ''',
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

            # mod_df_dict['shot_or_stabbed'].append(row['shot_or_stabbed'][4:])
            # mod_df_dict['arrested'].append(row['arrested'][4:])
            # mod_df_dict['physical_fight'].append(row['physical_fight'][4:])
            # mod_df_dict['convicted_of_charges'].append(row['convicted_of_charges'][4:])
            # mod_df_dict['sell_drugs'].append(row['sell_drugs'][4:])
            # mod_df_dict['counseling'].append(row['counseling'][4:])
            # mod_df_dict['sadness_family'].append(row['sadness_family'][4:])
            # mod_df_dict['worrying'].append(row['worrying'][4:])
            # mod_df_dict['suicide'].append(row['suicide'][4:])
            # mod_df_dict['optimism'].append(row['optimism'][4:])
            # mod_df_dict['happiness'].append(row['happiness'][4:])
            # mod_df_dict['fast_food'].append(row['fast_food'][4:])
            # mod_df_dict['hours_of_tv'].append(row['hours_of_tv'][4:])
            # mod_df_dict['individual_sports'].append(row['individual_sports'][4:])
            # mod_df_dict['smoked_cigarette'].append(row['smoked_cigarette'][4:])
            # mod_df_dict['physical_child_abuse'].append(row['physical_child_abuse'][4:])
            # mod_df_dict['age_of_first_drink'].append(row['age_of_first_drink'][4:])
            # mod_df_dict['car_accidents'].append(row['car_accidents'][4:])
            # mod_df_dict['drinking'].append(row['drinking'][4:])
            # mod_df_dict['prayer_in_private'].append(row['prayer_in_private'][4:])
            
        new_df = pd.DataFrame(mod_df_dict, index=df.index)

            


        print(df)
        pass

    def get_age(self, year):
        return date.today().year - year

    def get_tokens_yn(self):
        return {
            'No': 'No', 
            'Yes': 'Yes'}

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
