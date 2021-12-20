from collections import defaultdict
from dataset import Dataset
import argparse
import pandas as pd
import numpy as np

class AnesDataset(Dataset):
    """
    This class assumes that the data has been decoded using the codebook

    To add templates, insert more entries into the dictionary in the _get_templates method
    To add dv phrasings for existing or new dvs, insert more entries into the dictionary in the _make_dv_question method
    For each new dv added, add a new entry to the _get_tokens method
    
    Note: this class is set up to handle a (dataset, dv) tuple as input to the future pipeline
    (i.e. to run an experiment for a survey, this class is set up to handle only one dv)
    """

    def __init__(self, dv, in_fname=None, opening_func=None, out_fname=None, n=None):
        super().__init__(dv, in_fname=in_fname, opening_func=opening_func, out_fname=out_fname, n=n)

    def _modify_raw_data(self, df):
        # if the data hasn't already been decoded use the codebook values to decode
        # ...
        # else
        if self._dv not in df.columns:
            raise KeyError(f"{self._dv} not found in survey dvs")

        mod_df_dict = defaultdict(list)
        for _, row in df.iterrows():
            mod_df_dict['age'].append(row['age'])
            mod_df_dict["gender"].append(row["gender"])
            mod_df_dict['education'].append(row['education'])
            mod_df_dict['party'].append(row['party'])
            mod_df_dict["ideo"].append(row["ideo"])
            mod_df_dict['religion'].append(row['religion'])
            mod_df_dict["race"].append(row["race"])
            mod_df_dict['region'].append(row['region'])
            mod_df_dict["marital"].append(row["marital"])
            mod_df_dict["ground_truth"].append(row[self._dv])
        new_df = pd.DataFrame(mod_df_dict, index=df.index)
        new_df['age'] = new_df['age'].fillna(-1)
        new_df['age'] = new_df['age'].astype('int', errors='ignore').astype('str', errors='ignore')
        new_df['party'] = new_df['party'].astype(str)
        return new_df
        
    def _get_templates(self):
        """
        Returns a dictionary of templates for a given dv
        """
        templates = {
            "fpbs0" : (lambda row: f"{self._make_fpbs(row)}\n{self._make_dv_question(self._dv)[0]}", self._get_tokens()[self._dv]),
        }
        return templates

    def _get_tokens(self):
        return {
            "2016_presidential_vote" : {
                'Hillary Clinton': ['Clinton', 'Hillary'], 
                'Donald Trump': ['Trump', 'Donald']},
        }

    def _make_qa(self, row):
        """
        make question and answer backstory
        Returns a string of the question and answer
        """
        dictionary = {
            'age': {
                '-1': '',
                'nan': '',
                'default': f'''Q: What is your age?\nA: {row['age']} years old\n\n''',
            },
            'gender': {
                np.nan: '',
                'nan': '',
                'default': f'''Q: What is your gender?\nA: {row['gender']}\n\n''',
            },
            'marital': {
                np.nan: '',
                'nan': '',
                'default': f'''Q: What is your marital status?\nA: {row['marital']}\n\n''',
            },
            'religion': {
                np.nan: '',
                'nan': '',
                'default': f'Q: What is your religion?\nA: {row["religion"]}\n\n',
            },
            'race': {
                np.nan: '',
                'nan': '',
                'default': f'Q: What is your race/ethnicity?\nA: {row["race"]}\n\n',
            },
            'education': {
                np.nan: '',
                'nan': '',
                'default': f'''Q: What is your education level?\nA: {row['education']}\n\n''',
            },
            'party': {
                np.nan: '',
                'nan': '',
                'default': f'''Q: What is your political party?\nA: {row['party']}\n\n''',
            },
            'ideo': {
                np.nan: '',
                'nan': '',
                'default': f'''Q: What is your ideology?\nA: {str(row['ideo']).strip().lower()}\n\n''',
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

    def _make_fpbs(self, row):
        '''
        make first person backstory
        Returns a string of the first person backstory
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
            'marital': {
                np.nan: '',
                'nan': '',
                'default': f'''I am {row['gender']}. ''',
                'never married': 'I have never married. ',
                'default': f'''I am {row['marital']}. ''',
            },
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
                'default': f'''I am {row['gender']}. ''',

                'default': f'Racially, I identify as {row["race"]}. ',
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
            'party': {
                np.nan: '',
                'nan': '',
                'None/Independent': 'I am an independent. ',
                'default': f'''I am a {row['party']}. ''',
            },
            'ideo': {
                np.nan: '',
                'nan': '',
                'default': f'''Ideologically, I am {str(row['ideo']).strip().lower()}. ''',
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

    def _make_dv_question(self, dv_col_name):
        """
        Add templates for dv phrasing here
        Returns a list of dv phrasings
        """
        dv_questions = {
            "2016_presidential_vote" : [
                "In 2016, between Trump and Clinton, I voted for",
                "Considering the above biographic statements, which \
                 presidential candidate did the person most likely vote \
                 for in the 2016 US election?\nThe person probably voted for",
                "Between Clinton and Trump, the candidate I voted for in the 2016 presidential election was",
                "Q: In 2016, between Trump and Clinton, who did you vote for?\n\nA:",
                "Q: Who did you vote for in the 2016 election? Trump or Clinton?\n\nA:",
            ],
        }
        try:
            return dv_questions[dv_col_name]
        except KeyError:
            raise KeyError(f"{dv_col_name} not found in dv_questions")


if __name__ == "__main__":
    """
    args: dv (\'2016_presidential_vote\', ), n, in_fname, out_fname
    Build dataset from anes survey 
    Saves a pkl file with templates to out_fname
    Prints out a few examples of the templates
    """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dv', type=str, default='2016_presidential_vote')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--in_fname', type=str)
    parser.add_argument('--out_fname', type=str)
    args = parser.parse_args()

    # build dataset
    anes = AnesDataset(dv=args.dv, in_fname=args.in_fname, out_fname=args.out_fname)

    print(anes._get_n_prompts(10))