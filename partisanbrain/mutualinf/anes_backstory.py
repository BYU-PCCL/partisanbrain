import pandas as pd
import numpy as np

def make_backstory1(row):
    '''
    Columns: age', 'gender', 'party', 'education', 'ideo', 'religion', 'race', 'region', 'marital', '2016_presidential_vote'
    '''
    dictionary = {
        'age': {
            np.nan: '',
            'default': f'''I am {int(row['age'])} years old. ''',
        },
        'gender': {
            np.nan: '',
            'default': f'''I am {row['gender']}. ''',
        },
        'marital': {
            'never married': 'I have never married. ',
            'default': f'''I am {row['marital']}. ''',
        },
        'religion': {
            'Undiferentiated Protstant': 'Religiously, I am protestant. ',
            'Other Christian': 'Religiously, I am Christian. ',
            'default': f'Religiously, I identify as {row["religion"]}. ',
        },
        'race': {
            'default': f'Racially, I identify as {row["race"]}. ',
        },
        'education': {
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
            'None/Independent': 'I am an independent. ',
            'default': f'''I am a {row['party']}. ''',
        },
        'ideo': {
            'default': f'''Ideologically, I am {row['ideo'].strip().lower()}. ''',
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

def make_backstory2(row):
    '''
    Columns: age', 'gender', 'party', 'education', 'ideo', 'religion', 'race', 'region', 'marital', '2016_presidential_vote'
    '''
    dictionary = {
        'party': {
            np.nan: '',
            'None/Independent': 'I am an independent. ',
            'default': f'''I am a {row['party']}. ''',
        },
        'ideo': {
            'default': f'''Ideologically, I am {row['ideo'].strip().lower()}. ''',
        },
        'age': {
            np.nan: '',
            'default': f'''I am {int(row['age'])} years old. ''',
        },
        'gender': {
            np.nan: '',
            'default': f'''I am {row['gender']}. ''',
        },
        'marital': {
            'never married': 'I have never married. ',
            'default': f'''I am {row['marital']}. ''',
        },
        'religion': {
            'Undiferentiated Protstant': 'Religiously, I am protestant. ',
            'Other Christian': 'Religiously, I am Christian. ',
            'default': f'Religiously, I identify as {row["religion"]}. ',
        },
        'race': {
            'default': f'Racially, I identify as {row["race"]}. ',
        },
        'education': {
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
    }
    backstory = ''
    for key in dictionary.keys():
        val = row[key]
        if val in dictionary[key]:
            backstory += dictionary[key][val]
        else:
            backstory += dictionary[key]['default']
    return backstory

def make_backstory3(row):
    '''
    Columns: age', 'gender', 'party', 'education', 'ideo', 'religion', 'race', 'region', 'marital', '2016_presidential_vote'
    '''
    dictionary = {
        'age': {
            'default': f'''Q: What is your age?\nA: {int(row['age'])} years old\n\n''',
        },
        'gender': {
            'default': f'''Q: What is your gender?\nA: {row['gender']}\n\n''',
        },
        'marital': {
            'default': f'''Q: What is your marital status?\nA: {row['marital']}\n\n''',
        },
        'religion': {
            'default': f'Q: What is your religion?\nA: {row["religion"]}\n\n',
        },
        'race': {
            'default': f'Q: What is your race/ethnicity?\nA: {row["race"]}\n\n',
        },
        'education': {
            'default': f'''Q: What is your education level?\nA: {row['education']}\n\n''',
        },
        'party': {
            'default': f'''Q: What is your political party?\nA: {row['party']}\n\n''',
        },
        'ideo': {
            'default': f'''Q: What is your ideology?\nA: {row['ideo'].strip().lower()}\n\n''',
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


if __name__ == '__main__':
    # load data/formatted_anes.csv
    df = pd.read_csv('data/formatted_anes.csv')
    # apply make_backstory1 rowwise
    df = df.dropna()
    df['backstory'] = df.apply(make_backstory3, axis=1)
    print(df.backstory)
    print(df.backstory.iloc[0])