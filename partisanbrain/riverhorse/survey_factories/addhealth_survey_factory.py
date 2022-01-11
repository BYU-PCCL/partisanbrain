import re

from numpy.core.numeric import outer
from ..dataset_factory import DatasetFactory
from ..surveys.addhealth_survey import AddhealthSurvey
from datetime import date
from collections import defaultdict
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

    GENDER_NEUTRAL_NAME = "Sam"

    questions = {
        'age': "What year were you born? ",
        'gender': "What gender are you? ",
        'ideology': "In terms of politics, do you consider yourself very conservative, " 
            + "conservative, middle-of-the-road, liberal, or very liberal? ",
        'education': "What is the highest level of education that you have achieved to date? ",
        'income': "What is your household income? ",
        'religion': "What is your present religion? ",
        'race_ethnicity': "What race are you? ",
        'marital': "How many persons have you ever married? (including a current spouse) ",
    }

    answers = {
        'age': ["1974",
                "1975",
                "1976",
                "1977",
                "1978",
                "1979",
                "1980",
                "1981",
                "1982",
                "1983", ],
        'gender': ["Male", "Female"],
        'ideology': ["Conservative", "Middle-of-the-road", "Liberal"],
        'education': ['8th grade or less',
                'Some high school',
                'High school graduate',
                'Some vocational/technical training (after high school)',
                'Completed vocational/technical training (after high scho',
                'Some college',
                "Completed college (bachelor's degree)",
                "Some graduate school",
                "Completed a master's degree ",
                "Some graduate training beyond a master's degree",
                "Completed a doctoral degree ",
                "Some post baccalaureate professional education",
                "Completed post baccalaureate professional education"],
        'income': ['$5,000 to $9,999',
                '$10,000 to $14,999',
                '$15,000 to $19,999',
                '$20,000 to $24,999',
                '$25,000 to $29,999',
                '$30,000 to $39,999',
                '$40,000 to $49,999',
                '$50,000 to $74,999',
                '$75,000 to $99,999',
                '$100,000 to $149,999',
                '$150,000 or more'],
        'religion': ["none/atheist/agnostic",
                "Protestant (such as Assembly of God, Baptist, Lutheran, Methodist, Presbyterian, etc.)"
                "Catholic",
                "Other Christian",
                "Jewish",
                "Buddhist",
                "Hindu",
                "Muslim",
                "Other",
                # "Refused",
                # "Don't know"
                ],
        'race_ethnicity': ["White",
                "Black or African American",
                "American Indian or Alaska Native",
                "Asian or Pacific Islander",
                # "Missing"
                ],
        'marital': ["0 persons",
                "1 person",
                "2 persons",
                "3 persons",
                "4 persons",
                # "refused",
                # "Don't know"
                ],
    }

    def get_dictionary(self, row):
        dictionary = {
            'age': {
                '-1': '',
                'default': f'''I was born in the year {row['age']}. ''',
            },
            'gender': {
                'default': f'''I am {row['gender']}. ''',
            },
            'ideology': {
                'Refused' : np.nan,
                "Dont' know" : np.nan,
                'default': f'''Ideologically, I am {str(row['ideo']).strip().lower()}. ''',
            },
            'education': {
                "Don't know": np.nan,
                'Refused': np.nan,
                '8th grade or less' : "I completed up to 8th grade or less. ",
                'Some high school': "I attended some high school. ",
                'High school graduate': "I graduated high school. ",
                'Some vocational/technical training (after high school)': 
                    "I have completed some vocational/technical training",
                'Completed vocational/technical training (after high scho' : 
                    "I completed vocational or technical training after high school.",
                'Some college': "I attended some college. ",
                "Completed college (bachelor's degree)" : "I have a Bachelor's degree. ",
                "Some graduate school" : "I have attended some graduate school",
                "Completed a master's degree " : "I have a Master's degree",
                "Some graduate training beyond a master's degree" : 
                    "I have attended some graduate training beyond a master's degree",
                "Completed a doctoral degree " : 
                    "I have completed a doctoral degree",
                "Some post baccalaureate professional education" : 
                    "I have completed some post baccalaureate professional education",
                "Completed post baccalaureate professional education" : 
                    "I have completed post baccalaureate professional education"
                # no default, should be exhaustive
            },
            'income': {
                "Don't know": np.nan,
                'Refused': np.nan,
                'default': f'''My household income is {row['income']}. ''',
                
            },
            'religion': {
                "Don't know": np.nan,
                'Refused': np.nan,
                'none/atheist/agnostic' : "I am either atheist, agnostic, or non-religious",
                'Protestant (Such as Assembly of God, Baptist, etc.)': 'Religiously, I identify as protestant',
                'Other Christian': 'Religiously, I am Christian. ',
                'Other' : "I am religious",
                # 'default': f'Religiously, I identify as {row["religion"]}. ',
                'default': f'''I am {row['religion']}. ''',
            },
            'race_ethnicity': {
                "Missing": np.nan,
                'default': f'''I am {row['race_ethnicity']}. ''',
                # 'default': f'Racially, I identify as {row["race"]}. ',
            },
            #region
            'marital': {
                "Don't know": np.nan,
                'Refused': np.nan,
                # 'never married': 'I have never married. ',
                'default': f'''I have married {row['marital']}. ''',
            },
        }
        return dictionary

    def get_dictionary_third_person(self, row, name):
        dictionary = {
            'age': {
                '-1': '',
                'default': f'''{name} was born in the year {row['age']}. ''',
            },
            'gender': {
                'default': f'''{name} is {row['gender']}. ''',
            },
            'ideology': {
                'Refused' : np.nan,
                "Dont' know" : np.nan,
                'default': f'''Ideologically, {name} is {str(row['ideo']).strip().lower()}. ''',
            },
            'education': {
                "Don't know": np.nan,
                'Refused': np.nan,
                '8th grade or less' : f"{name} completed up to 8th grade or less. ",
                'Some high school': f"{name} attended some high school. ",
                'High school graduate': f"{name} graduated high school. ",
                'Some vocational/technical training (after high school)': 
                    f"{name} has completed some vocational/technical training",
                'Completed vocational/technical training (after high scho' : 
                    f"{name} completed vocational or technical training after high school.",
                'Some college': f"{name} attended some college. ",
                "Completed college (bachelor's degree)" : f"{name} has a Bachelor's degree. ",
                "Some graduate school" : f"{name} has attended some graduate school",
                "Completed a master's degree " : f"{name} has a Master's degree",
                "Some graduate training beyond a master's degree" : 
                    f"{name} has attended some graduate training beyond a master's degree",
                "Completed a doctoral degree " : 
                    f"{name} has completed a doctoral degree",
                "Some post baccalaureate professional education" : 
                    f"{name} has completed some post baccalaureate professional education",
                "Completed post baccalaureate professional education" : 
                    f"{name} has completed post baccalaureate professional education"
                # no default, should be exhaustive
            },
            'income': {
                "Don't know": np.nan,
                'Refused': np.nan,
                'default': f'''{name}'s household income is {row['income']}. ''',
            },
            'religion': {
                "Don't know": np.nan,
                'Refused': np.nan,
                'none/atheist/agnostic' : f"{name} is either atheist, agnostic, or non-religious",
                'Protestant (Such as Assembly of God, Baptist, etc.)': f'Religiously, {name} identifies as protestant',
                'Other Christian': f'Religiously, {name} is Christian. ',
                'Other' : f"{name} is religious",
                # 'default': f'Religiously, I identify as {row["religion"]}. ',
                'default': f'''{name} is {row['religion']}. ''',
            },
            'race_ethnicity': {
                "Missing": np.nan,
                'default': f'''{name} is {row['race_ethnicity']}. ''',
                # 'default': f'Racially, I identify as {row["race"]}. ',
            },
            #region
            'marital': {
                "Don't know": np.nan,
                'Refused': np.nan,
                # 'never married': 'I have never married. ',
                'default': f'''{name} has married {row['marital']}. ''',
            },
        }
        return dictionary
    
    # 01 First Person
    def mb_first_person(self, row):
        '''
        list style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        backstory = ''
        for key in dictionary.keys():
            val = row[key]
            if val in dictionary[key]:
                backstory += dictionary[key][val]
            else:
                backstory += dictionary[key]['default']
        return backstory

    # 02 Third Person
    def mb_third_person(self, row):
        '''
        list style backstory dropping nans
        '''
        dictionary = self.get_dictionary_third_person(row, self.GENDER_NEUTRAL_NAME)
        backstory = ''
        for key in dictionary.keys():
            val = row[key]
            if val in dictionary[key]:
                backstory += dictionary[key][val]
            else:
                backstory += dictionary[key]['default']
        return backstory

    # 03 QA
    def mb_qa(self, row):
        '''
        qa style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        backstory = ''
        for key in dictionary.keys():
            val = row[key]
            backstory += "Q: " + questions[key] + "\nA: "
            if val in dictionary[key]:
                backstory += dictionary[key][val] + "\n"
            else:
                backstory += dictionary[key]['default'] + "\n"
        backstory += "Q: "
        return backstory

    # 04 Explicit
    def mb_qa_exp(self, row):
        '''
        qa style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        answers = self.answers
        backstory = ''
        for key in dictionary.keys():
            val = row[key]
            backstory += "Q: " + questions[key][:len(questions[key])-2] + " ("
            comma = ""
            for ans in answers[key]:
                backstory += comma + ans
                comma = ", "
            backstory += ") ?\nA: "
            if val in dictionary[key]:
                backstory += dictionary[key][val] + "\n"
            else:
                backstory += dictionary[key]['default'] + "\n"
        backstory += "Q: "
        return backstory

    # 05 First PersonConversation (P1, P2)
    def mb_convo(self, row):
        '''
        conversation style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        backstory = ''
        for key in dictionary.keys():
            val = row[key]
            backstory += "P1: " + questions[key] + "\nP2: "
            if val in dictionary[key]:
                backstory += dictionary[key][val] + "\n"
            else:
                backstory += dictionary[key]['default'] + "\n"
        backstory += "P1: "
        return backstory

    # 06 First Person Answer Key
    def mb_ans_key(self, row):
        '''
        Answer key style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        backstory = "PASSAGE:\n"
        for key in dictionary.keys():
            val = row[key]
            if val in dictionary[key]:
                backstory += dictionary[key][val]
            else:
                backstory += dictionary[key]['default']
        backstory += "\n\n QUESTIONS: \n"
        x = 1
        for key in dictionary.keys():
            backstory += x + ") " + questions[key] + "\n"
            x += 1
        return backstory

    def ans_key_answers(self, row):
        dictionary = self.get_dictionary(row)
        x = 1
        answer_key = "\n\nANSWER KEY:\n\n" + x + ") "
        
        for key in dictionary.keys():
            x += 1
            val = row[key]
            answer_key += val + "\n" + x + ") "
            
    # 07 Survey Response
    def mb_survey(self, row):
        '''
        survey style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        answers = self.answers
        backstory = ''
        x = 1
        for key in dictionary.keys():
            val = row[key]
            backstory += "Question " + x + ": " + questions[key] + " ("
            comma = ""
            for ans in answers[key]:
                backstory += comma + ans
                comma = ", "
            backstory += ")\n Answer " + x + ": "
            if val in dictionary[key]:
                backstory += dictionary[key][val] + "\n"
            else:
                backstory += dictionary[key]['default'] + "\n"
            x += 1
        
        backstory += "Question " + x + ": " # should be len(dictionary.keys()), right?
        return backstory

    # better way to do this?
    def get_answer_num(self, row):
        return len(self.get_dictionary(row).keys())

    # 08 Multiple Choice Response
    def mb_mult(self, row):
        '''
        mutliple choice style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        answers = self.answers
        backstory = ''
        x = 1
        for key in dictionary.keys():
            val = row[key]
            backstory += "Question " + x + ": " + questions[key] + "\n"
            x = 0
            backstory += self.format_mult(answers[key]) # includes "Correct Answer: "
            backstory += chr(65 + answers[key].index(val))
            x += 1
        
        backstory += "Question " + x + ": " # should be len(dictionary.keys()), right?
        return backstory
    
    def format_mult(self,answers):
        output = ""
        for ans in answers:
                output += chr(65 + answers.index(ans)) + ": " + ans + "\n"
        output += "Correct Answer: "
        return output

    def get_mult_yn(self):
        return self.format_mult(["Yes", "No"])
    # For yes/no, do format_mult(["Yes, "No"])

    

    
    def modify_data(self, df):
        # file = open("data.txt","w+")
        # file.write(df)
        # file.close()
        # for x in df:
        #     print(y)

        # rename rows?
        mod_df_dict = defaultdict(list)
        for _, row in df.iterrows():
            # mod_df_dict['age'].append(self.get_age(row['age'])) should we have age or year of birth?
            mod_df_dict['age'].append(row['age'])
            mod_df_dict['gender'].append(row['gender'][4:])
            mod_df_dict['education'].append(row['education'][4:])
            mod_df_dict['ideology'].append(self.get_ideo(row['ideology'][4:]))
            mod_df_dict['income'].append(row['income'][4:])
            mod_df_dict['religion'].append(row['religion'][4:])
            mod_df_dict['race_ethnicity'].append(row['race_ethnicity'][4:])
            mod_df_dict['marital_status'].append(row['marital_status'][4:])

            if False:
                maybe = {
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
                }
            
        new_df = pd.DataFrame(mod_df_dict, index=df.index)

        return new_df

    def get_age(self, year):
        return date.today().year - year

    def get_ideo(self, ideo):
        if ideo == "Very Conservative":
            return "Conservative"
        if ideo == "Very Liberal":
            return "Liberal"
        if ideo == np.nan: 
            return 'Refused'
        return ideo
        

    def get_tokens_yn(self):
        return {
            'No': 'No', 
            'Yes': 'Yes'}

    def get_tokens_mult(self):
        return {
            'A': 'Yes',
            'B': 'No', }

    def get_tokens_2_times(self):
        return {
            "Never", 
            "1 or 2 times", 
            "3 or 4 times", 
            "5 or more times"}

    def get_tokens_charges(self):
        return {
            "No", 
            "Once", 
            "More than once"}

    def get_tokens_occasion(self):
        return {
            "Never or rarely", 
            "Sometimes", 
            "A lot of the time",
            "Most of the time or all of the time"}

    def get_tokens_agree(self):
        return {
            # "Strongly agree" 
            "Agree", 
            "Neither agree nor disagree",
            "Disagree",
            # "Strongly disagree"
            "Most of the time or all of the time"}

    def get_tokens_times(self):
        return {
            "None",
            "Once",
            "Twice",
            "3 or 4 times"
            "5 or 6 times"
        }

    def get_tokens_times7(self):
        return {
            "1 time",
            "2 times",
            "3 times",
            "4 times",
            "5 times",
            "6 times",
            "7 or more times",
        }

    def get_tokens_abuse(self):
        return {
            "one time",
            "two times",
            "three times",
            "six to ten times",
            "more than ten times",
            "this has never happened",
        }

    def get_tokens_prayer(self):
        return {
            "Never",
            "Once a month",
            "A few times a month",
            "Once a week",
            "A few times a day",
            "Once a day",
            "More than once a day",
        }

    

    def get_templates(self):
        templates = {
            "shot_or_stabbed": {
                "first_person1": (lambda row: (f"{self.mb_first_person(row)}\n Have I shot or stabbed someone in the past 12 months? My answer is"), 
                self.get_tokens_yn()),
                "first_person2": (lambda row: (f"{self.mb_first_person(row)}\n Is it true that I have shot or stabbed someone in the past 12 months? Answer yes or no:"), 
                self.get_tokens_yn()),
                "first_person3": (lambda row: (f"{self.mb_first_person(row)}\n In the past year, has someone been shot or stabbed by me? Answer yes or no:"), 
                self.get_tokens_yn()),

                # "third_person1": (lambda row: (f"{self.mb_third_person(row)}\n In the past 12 months, has {self.GENDER_NEUTRAL_NAME} shot or stabbed someone?"), 
                # self.get_tokens_yn()),
                "third_person2": (lambda row: (f"{self.mb_third_person(row)}\n Has {self.GENDER_NEUTRAL_NAME} shot or stabbed at least one person in the past year?"),  
                self.get_tokens_yn()),
                "third_person3": (lambda row: (f"{self.mb_third_person(row)}\n In the past year, has someone been shot or stabbed by {self.GENDER_NEUTRAL_NAME}?"), 
                self.get_tokens_yn()),
                
                "qa1": (lambda row: (f"{self.mb_qa(row)}\n Have you shot or stabbed someone in the past 12 months? \nA:"), 
                self.get_tokens_yn()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have shot or stabbed someone in the past 12 months? \nA:"), 
                self.get_tokens_yn()),
                "qa3": (lambda row: (f"{self.mb_qa(row)}\n Has someone been shot or stabbed by you in the past year? \nA:"), 
                self.get_tokens_yn()),

                "qa1": (lambda row: (f"{self.mb_qa_exp(row)}\n Have you shot or stabbed someone in the past 12 months (Yes, No) ? \nA:"), 
                self.get_tokens_yn()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have shot or stabbed someone in the past 12 months (Yes, No) ? \nA:"), 
                self.get_tokens_yn()),
                # "qa3": (lambda row: (f"{self.mb_qa(row)}\n Has someone been shot or stabbed by you in the past year (Yes, No) ? \nA:"), 
                # self.get_tokens_yn()),
                
                "conv1": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have shot or stabbed someone in the past 12 months? P2:"), 
                self.get_tokens_yn()),
                "conv2": (lambda row: (f"{self.mb_qa(row)}\n I heard that you have shot or stabbed someone in the past 12 months. Is that true? P2:"), 
                self.get_tokens_yn()),
                # "conv3": (lambda row: (f"{self.mb_qa(row)}\n Did you shoot or stab someone in the past 12 months? P2:"), 
                # self.get_tokens_yn()),

                "anskey1": (lambda row: (f"{self.mb_ans_key(row)}\n In the past year, has someone been shot or stabbed by you? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey2": (lambda row: (f"{self.mb_ans_key(row)}\n Was someone shot or stabbed by you in the past year? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey3": (lambda row: (f"{self.mb_ans_key(row)}\n If these are true about you, have you shot or stabbed someone in the past 12 months? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),

                "survey1": (lambda row: (f"{self.mb_survey(row)}\n In the past year, have you shot or stabbed someone? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),
                "survey2": (lambda row: (f"{self.mb_survey(row)}\n Has anyone been shot or stabbed by you in the past 12 months? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),

                "mult1": (lambda row: (f"{self.mb_mult(row)}\n You have shot or stabbed someone within 12 months from today. Is this true? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
                "mult2": (lambda row: (f"{self.mb_mult(row)}\n Can you honestly say that in the past 12 months, you have shot or stabbed someone? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
                "mult3": (lambda row: (f"{self.mb_mult(row)}\n In the past 12 months, have you shot or stabbed someone? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
            },
            "arrested": {
                "first_person1": (lambda row: (f"{self.mb_first_person(row)}\n Have you ever been arrested?"), 
                self.get_tokens_yn()),
                "first_person2": (lambda row: (f"{self.mb_first_person(row)}\n Is it true that you have been arrested?"), 
                self.get_tokens_yn()),
                "first_person3": (lambda row: (f"{self.mb_first_person(row)}\n I heard that you have been arrested. Is that true?"), 
                self.get_tokens_yn()),

                # "third_person1": (lambda row: (f"{self.mb_third_person(row)}\n , has {self.GENDER_NEUTRAL_NAME} shot or stabbed someone?"), 
                # self.get_tokens_yn()),
                "third_person2": (lambda row: (f"{self.mb_third_person(row)}\n Has {self.GENDER_NEUTRAL_NAME} been arrested?"),  
                self.get_tokens_yn()),
                "third_person3": (lambda row: (f"{self.mb_third_person(row)}\n Is it likely that {self.GENDER_NEUTRAL_NAME} has been arrested before? Answer yes or no:"), 
                self.get_tokens_yn()),
                
                "qa1": (lambda row: (f"{self.mb_qa(row)}\n Have you been arrested before? \nA:"), 
                self.get_tokens_yn()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have been arrested? \nA:"), 
                self.get_tokens_yn()),
                "qa3": (lambda row: (f"{self.mb_qa(row)}\n Can you honestly say that you have been arrested? \nA:"), 
                self.get_tokens_yn()),

                "qa1": (lambda row: (f"{self.mb_qa_exp(row)}\n Have you ever been arrested (Yes, No) ? \nA:"), 
                self.get_tokens_yn()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it likely that you have been arrested before (Yes, No) ? \nA:"), 
                self.get_tokens_yn()),
                # "qa3": (lambda row: (f"{self.mb_qa(row)}\n have you been arrested  (Yes, No) ? \nA:"), 
                # self.get_tokens_yn()),
                
                "conv1": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have been arrested before? P2:"), 
                self.get_tokens_yn()),
                "conv2": (lambda row: (f"{self.mb_qa(row)}\n I heard that you have been arrested before. Is that true? P2:"), 
                self.get_tokens_yn()),
                # "conv3": (lambda row: (f"{self.mb_qa(row)}\n Did you shoot or stab someone ? P2:"), 
                # self.get_tokens_yn()),

                "anskey1": (lambda row: (f"{self.mb_ans_key(row)}\n Have you ever been arrested? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey2": (lambda row: (f"{self.mb_ans_key(row)}\n Have you ever been arrested by someone? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey3": (lambda row: (f"{self.mb_ans_key(row)}\n If these are true about you, have you been arrested? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),

                "survey1": (lambda row: (f"{self.mb_survey(row)}\n Is it true that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),
                "survey2": (lambda row: (f"{self.mb_survey(row)}\n Is it likely that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),

                "mult1": (lambda row: (f"{self.mb_mult(row)}\n You have been arrested before. Is this true? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
                "mult2": (lambda row: (f"{self.mb_mult(row)}\n Can you honestly say that you have been arrested? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
                "mult3": (lambda row: (f"{self.mb_mult(row)}\n Have you ever been arrested? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
            },
            "physical_fight": {
                "first_person1": (lambda row: (f"{self.mb_first_person(row)}\n Have you ever been arrested?"), 
                self.get_tokens_yn()),
                "first_person2": (lambda row: (f"{self.mb_first_person(row)}\n Is it true that you have been arrested?"), 
                self.get_tokens_yn()),
                "first_person3": (lambda row: (f"{self.mb_first_person(row)}\n I heard that you have been arrested. Is that true?"), 
                self.get_tokens_yn()),

                # "third_person1": (lambda row: (f"{self.mb_third_person(row)}\n , has {self.GENDER_NEUTRAL_NAME} shot or stabbed someone?"), 
                # self.get_tokens_yn()),
                "third_person2": (lambda row: (f"{self.mb_third_person(row)}\n Has {self.GENDER_NEUTRAL_NAME} been arrested?"),  
                self.get_tokens_yn()),
                "third_person3": (lambda row: (f"{self.mb_third_person(row)}\n Is it likely that {self.GENDER_NEUTRAL_NAME} has been arrested before? Answer yes or no:"), 
                self.get_tokens_yn()),
                
                "qa1": (lambda row: (f"{self.mb_qa(row)}\n Have you been arrested before? \nA:"), 
                self.get_tokens_yn()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have been arrested? \nA:"), 
                self.get_tokens_yn()),
                "qa3": (lambda row: (f"{self.mb_qa(row)}\n Can you honestly say that you have been arrested? \nA:"), 
                self.get_tokens_yn()),

                "qa1": (lambda row: (f"{self.mb_qa_exp(row)}\n Have you ever been arrested (Yes, No) ? \nA:"), 
                self.get_tokens_yn()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it likely that you have been arrested before (Yes, No) ? \nA:"), 
                self.get_tokens_yn()),
                # "qa3": (lambda row: (f"{self.mb_qa(row)}\n have you been arrested  (Yes, No) ? \nA:"), 
                # self.get_tokens_yn()),
                
                "conv1": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have been arrested before? P2:"), 
                self.get_tokens_yn()),
                "conv2": (lambda row: (f"{self.mb_qa(row)}\n I heard that you have been arrested before. Is that true? P2:"), 
                self.get_tokens_yn()),
                # "conv3": (lambda row: (f"{self.mb_qa(row)}\n Did you shoot or stab someone ? P2:"), 
                # self.get_tokens_yn()),

                "anskey1": (lambda row: (f"{self.mb_ans_key(row)}\n Have you ever been arrested? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey2": (lambda row: (f"{self.mb_ans_key(row)}\n Have you ever been arrested by someone? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey3": (lambda row: (f"{self.mb_ans_key(row)}\n If these are true about you, have you been arrested? {self.ans_key_answers(row)}"), 
                self.get_tokens_yn()),

                "survey1": (lambda row: (f"{self.mb_survey(row)}\n Is it true that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),
                "survey2": (lambda row: (f"{self.mb_survey(row)}\n Is it likely that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),

                "mult1": (lambda row: (f"{self.mb_mult(row)}\n You have been arrested before. Is this true? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
                "mult2": (lambda row: (f"{self.mb_mult(row)}\n Can you honestly say that you have been arrested? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
                "mult3": (lambda row: (f"{self.mb_mult(row)}\n Have you ever been arrested? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult()),
            },
            # Counseling"In the past 12 months, have you received psychological or emotional counseling?",
        }
        if False:
                    print("")
                    # "qa1": (lambda row: (f"{self.mb_qa(row)}\n Have you shot or stabbed someone in the past 12 months? A: "), 
                    # self.get_tokens_yn()),
                    # "qa2": (lambda row: (f"{self.mb_qa(row)}\n Is it true that you have shot or stabbed someone in the past 12 months? A:"), 
                    # self.get_tokens_yn()),
                    # "qa3": (lambda row: (f"{self.mb_qa(row)}\n In the past 12 months, have you shot or stabbed someone? A:"), 
                    # self.get_tokens_yn()),
                    # "qa4": (lambda row: (f"{self.mb_qa(row)}\n In the past year, has someone been shot or stabbed by you? A:"), 
                    # self.get_tokens_yn()),
                    # "qa5": (lambda row: (f"{self.mb_qa(row)}\n Have you shot or stabbed at least one person in the past year? A:"), 
                    # self.get_tokens_yn()),
        return templates


if __name__ == "__main__":
    factory = AddhealthFactory(AddhealthSurvey())