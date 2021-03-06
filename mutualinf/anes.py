from collections import defaultdict
from infra_modules import Dataset

import pandas as pd
import numpy as np

SHOTS = [
"""Analyze the following biographic information and determine which candidate the person voted for in the 2016 election.
\"I am 58 years old. I am male. I am married. Religiously, I identify as Roman Catholic. Racially, I identify as White. I have a Bachelor's degree. I am a Republican. Ideologically, I am conservative.\"
Answer: The person voted for Trump.
""",
"""1) Given the following demographic information, which candidate did this person likely vote for in the 2016 election?
\"I am 18 years old. I am female. I am single. Religiously, I identify as Atheist. Racially, I identify as Black. I am a Democrat. Ideologically, I am liberal.\"
"""
]

class AnesDataset(Dataset):

    def __init__(self, sample_seed=0, n=None):
        self.token_set_dict = None
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
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
            mod_df_dict["ground_truth"].append(row["2016_presidential_vote"])
        new_df = pd.DataFrame(mod_df_dict, index=df.index)
        new_df['age'] = new_df['age'].fillna(-1)
        new_df['age'] = new_df['age'].astype('int', errors='ignore').astype('str', errors='ignore')
        new_df['party'] = new_df['party'].astype(str)
        return new_df

    def _get_templates(self):
        templates = {
            "template_1a": (lambda row: f"Analyze the following biographic information and determine which candidate the person voted for in the 2016 election.\n"
            f"{self.make_backstory1(row)}\nAnswer: This person voted for", self._get_tokens()),

            "template_1b": (lambda row: f"{self.make_backstory1(row)}In 2016, between Trump and Clinton, I voted for", self._get_tokens()),

            "template_1c": (lambda row: f"{self.make_backstory1(row)}\n"
            f"Considering the above biographic statements, which presidential candidate did the person most likely vote for in the 2016 US election?\n"
            f"The person probably voted for", self._get_tokens()),

            "template_1d": (lambda row: f"{self.make_backstory1(row)}The candidate I voted for in the 2016 presidential election was", self._get_tokens()),

            "template_1e": (lambda row: f"{self.make_backstory1(row)}Between Clinton and Trump, the candidate I voted for in the 2016 presidential election was", self._get_tokens()),

            "template_2a": (lambda row: f"After reading the following quote, infer which candidate the person voted for in the 2016 US election.\n"
            f"{self.make_backstory2(row)}\nWho did this person vote for?\nHe/she voted for", self._get_tokens()),

            "template_2b": (lambda row: f"{self.make_backstory2(row)}In 2016, between Trump and Clinton, I voted for", self._get_tokens()),

            "template_2c": (lambda row: f"{self.make_backstory2(row)}In 2016 the candidate I voted for was", self._get_tokens()),

            "template_2d": (lambda row: f"After reading the quote, GPT3 will predict which candidate the person voted for in the 2016 election.\n"
            f"{self.make_backstory2(row)}\n\nAnswer: The person selected", self._get_tokens()),

            "template_2e": (lambda row: f"{self.make_backstory2(row)}Between Clinton and Trump, the candidate I voted for in the 2016 presidential election was", self._get_tokens()),

            "template_3a": (lambda row: SHOTS[0] + f"\n\nAnalyze the following biographic information and determine which candidate the person voted for in the 2016 election.\n"
            f"{self.make_backstory1(row)}\nAnswer: The person voted for", self._get_tokens()),

            "template_3b": (lambda row: f"{self.make_backstory3(row)}Q: In 2016, between Trump and Clinton, who did you vote for?\n\nA:", self._get_tokens()),

            "template_3c": (lambda row: f"CHAPTER QUIZ:\n\n1) Given the following demographic information, which candidate did this person likely vote for in the 2016 election?\n\n"
            f"{self.make_backstory1(row)}\n\n"
            f"#########################\n\nANSWER KEY:\n"
            f"1) The person voted for", self._get_tokens()),

            "template_3d": (lambda row: f"{self.make_backstory3(row)}Q: Who did you vote for in the 2016 election? Trump or Clinton?\n\nA:", self._get_tokens()),

            "template_3e": (lambda row: f"{self.make_backstory3(row)}Q: Who did you vote for in the 2016 election?\n\nA:", self._get_tokens()),

            "template_4a": (lambda row: f"{self.make_backstory4(row)}Q: In 2016, between Clinton and Trump, who did you vote for?\n\nA:", self._get_tokens()),

            "template_4b": (lambda row: f"CHAPTER QUIZ:\n\n" + SHOTS[1] + "\n2) Given the following demographic information, which candidate did this person likely vote for in the 2016 election?\n"
            f"{self.make_backstory1(row)}\n\n"
            f"#########################\n\nANSWER KEY:\n"
            f"1) The person voted for Clinton.\n2) The person voted for", self._get_tokens()),

            "template_4c": (lambda row: f"{self.make_backstory4(row)}Q: Who did you vote for in the 2016 election? Clinton or Trump?\n\nA: I voted for", self._get_tokens()),

            "template_4d": (lambda row: f"Passage: {self.make_backstory1(row)}\n\n"
            f"Comprehension Questions:\n\n"
            f"1) In the 2016 election, the person described in the passage likely voted for who?\n"
            f"The person likely voted for", self._get_tokens()),

            "template_4e": (lambda row: f"P1: {self.make_backstory1(row)}\n"
            f"P2: So who did you vote for in the 2016 election?\n"
            f"P1: I voted for", self._get_tokens()),
        }
        return templates

    def _get_tokens(self):
        return {
            'Hillary Clinton': ['Clinton', 'Hillary'], 
            'Donald Trump': ['Trump', 'Donald']}

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

    def make_backstory2(self, row):
        '''
        list style backstory including nans
        '''
        dictionary = {
            'age': {
                '-1': 'I prefer not to say my age. ',
                'default': f'''I am {row['age']} years old. ''',
            },
            'gender': {
                np.nan: 'I prefer not to say my gender. ',
                'nan': 'I prefer not to say my gender. ',
                'default': f'''I am {row['gender']}. ''',
            },
            'marital': {
                np.nan: 'I prefer not to say my marital status. ',
                'nan': 'I prefer not to say my marital status. ',
                'never married': 'I have never married. ',
                'default': f'''I am {row['marital']}. ''',
            },
            'religion': {
                np.nan: 'I prefer not to say my religion. ',
                'nan': 'I prefer not to say my religion. ',
                'Undiferentiated Protstant': 'Religiously, I am protestant. ',
                'Undifferentiated Protstant': 'Religiously, I am protestant. ',
                'Other Christian': 'Religiously, I am Christian. ',
                'default': f'Religiously, I identify as {row["religion"]}. ',
            },
            'race': {
                np.nan: 'I prefer not to say my race. ',
                'nan': 'I prefer not to say my race. ',
                'default': f'Racially, I identify as {row["race"]}. ',
            },
            'education': {
                np.nan: 'I prefer not to say my education level. ',
                'nan': 'I prefer not to say my education level. ',
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
                np.nan: 'I prefer not to say my political party. ',
                'nan': 'I prefer not to say my political party. ',
                'None/Independent': 'I am an independent. ',
                'default': f'''I am a {row['party']}. ''',
            },
            'ideo': {
                np.nan: 'I prefer not to say my ideology. ',
                'nan': 'I prefer not to say my ideology. ',
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

    def make_backstory3(self, row):
        '''
        Q&A style not including nans
        '''
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

    def make_backstory4(self, row):
        '''
        Q&A style including nans
        '''
        dictionary = {
            'age': {
                '-1': f'''Q: What is your age?\nA: I prefer not to say my age\n\n''',
                'nan': '',
                'default': f'''Q: What is your age?\nA: {row['age']} years old\n\n''',
            },
            'gender': {
                np.nan: f'''Q: What is your gender?\nA: I prefer not to say my gender\n\n''',
                'nan': '',
                'default': f'''Q: What is your gender?\nA: {row['gender']}\n\n''',
            },
            'marital': {
                np.nan: f'''Q: What is your marital status?\nA: I prefer not to say my marital status\n\n''',
                'nan': '',
                'default': f'''Q: What is your marital status?\nA: {row['marital']}\n\n''',
            },
            'religion': {
                np.nan: f'''Q: What is your religion?\nA: I prefer not to say my religion\n\n''',
                'nan': '',
                'default': f'Q: What is your religion?\nA: {row["religion"]}\n\n',
            },
            'race': {
                np.nan: f'''Q: What is your race/ethnicity?\nA: I prefer not to say my race/ethnicity\n\n''',
                'nan': '',
                'default': f'Q: What is your race/ethnicity?\nA: {row["race"]}\n\n',
            },
            'education': {
                np.nan: f'''Q: What is your education level?\nA: I prefer not to say my education level\n\n''',
                'nan': '',
                'default': f'''Q: What is your education level?\nA: {row['education']}\n\n''',
            },
            'party': {
                np.nan: f'''Q: What is your political party?\nA: I prefer not to say my political party\n\n''',
                'nan': '',
                'default': f'''Q: What is your political party?\nA: {row['party']}\n\n''',
            },
            'ideo': {
                np.nan: f'''Q: What is your ideology?\nA: I prefer not to say my ideology\n\n''',
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


if __name__ == "__main__":
    # Data should be at data/example/raw.csv
    ad = AnesDataset()
