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
        'ideology': "In terms of politics, do you consider yourself " 
            + "conservative, middle-of-the-road, or liberal? ",
        'education': "What is the highest level of education that you have achieved to date? ",
        'income': "What is your household income? ",
        'religion': "What is your present religion? ",
        'race_ethnicity': "What race are you? ",
        'marital_status': "How many times have you been married? ",
    }

    answers_backstory = {
        'age': [1974,
                1975,
                1976,
                1977,
                1978,
                1979,
                1980,
                1981,
                1982,
                1983, ],
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
        'religion': ["Non/athiest/agnostic",
                "Protestant (such as Assembly of God, Baptist, Lutheran, Methodist, Presbyterian, etc.)",
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
        'marital_status': [
                0,
                1,
                2,
                3,
                4,
                # "0 persons",
                # "1 person",
                # "2 persons",
                # "3 persons",
                # "4 persons",
                # "refused",
                # "Don't know"
                ],
    }

    answers_dv = {
        "shot_or_stabbed" : ["Yes, No"],
        "arrested" : ["Yes, No"],
        "physical_fight" : ["Never", 
            "1 or 2 times", 
            "3 or 4 times", 
            "5 or more times"],        
        "convicted_of_charges" : ["No", 
            "Once", 
            "More than once"], 
        "sell_drugs" : ["Never", 
            "1 or 2 times", 
            "3 or 4 times", 
            "5 or more times"],
        "counseling" : ["Yes, No"],
        "sadness_family" : ["Never or rarely", 
            "Sometimes", 
            "A lot of the time",
            "Most of the time or all of the time"],
        "worrying" : ["Agree", 
            "Neither agree nor disagree",
            "Disagree",
            # "Strongly disagree"
            "Most of the time or all of the time"],
        "suicide" : ["None",
            "Once",
            "Twice",
            "3 or 4 times"
            "5 or 6 times"],
        "optimism" : ["Agree", 
            "Neither agree nor disagree",
            "Disagree",
            # "Strongly disagree"
            "Most of the time or all of the time"],
        "happiness" : ["Never or rarely", 
            "Sometimes", 
            "A lot of the time",
            "Most of the time or all of the time"],
        # "fast_food", # this is a weird one
        # "hours_of_tv", # weird one too
        "individual_sports" : ["1 time",
            "2 times",
            "3 times",
            "4 times",
            "5 times",
            "6 times",
            "7 or more times",],
        "smoked_cigarette" : ["Yes, No"],
        "physical_child_abuse" : ["One time",
            "Two times",
            "Three times",
            "Six to ten times",
            "More than ten times",
            "This has never happened",],
        # "age_of_first_drink", # big range of numbers
        "car_accidents" : ["Yes, No"],
        "drinking" : ["Yes, No"],
        "prayer_in_private" : ["Never",
            "Once a month",
            "A few times a month",
            "Once a week",
            "A few times a day",
            "Once a day",
            "More than once a day",],
    }

    def answers_or(self, answers):
        # answers = self.answers_dv[key]
        output = ""
        x = 0
        for ans in answers:
            comma = "" if x == 0 else ", or" if x == len(answers) - 1  else  ", "
            output += comma + ans
        return output

    def answers_paren(self, answers):
        # answers = self.answers_dv[key]
        output = "("
        comma = ""
        for ans in answers:
            output += comma + ans
            comma = ", "
        output += ")"
        return output



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
                'default': f'''Ideologically, I am {str(row['ideology']).strip().lower()}. ''',
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
                'default': f'''My household income is around {row['income']}. ''',
                
            },
            'religion': {
                "Don't know": np.nan,
                'Refused': np.nan,
                'Non/athiest/agnostic' : "I am either atheist, agnostic, or non-religious",
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
            'marital_status': {
                # This should just be 1/2/3/4 persons, but the data is a number. So be it I guess
                "Don't know": np.nan,
                'Refused': np.nan,
                # 'never married': 'I have never married. ',
                1 : f'''I have been married once. ''',
                'default': f'''I have been married {row['marital_status']} times. ''',
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
                'default': f'''Ideologically, {name} is {str(row['ideology']).strip().lower()}. ''',
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
                'default': f'''{name}'s household income is around {row['income']}. ''',
            },
            'religion': {
                "Don't know": np.nan,
                'Refused': np.nan,
                'Non/athiest/agnostic' : f"{name} is either atheist, agnostic, or non-religious",
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
            'marital_status': {
                "Don't know": np.nan,
                'Refused': np.nan,
                # 'never married': 'I have never married. ',
                'default': f'''{name} has married {row['marital_status']}. ''',
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
        answers = self.answers_backstory
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
            backstory += str(x) + ") " + questions[key] + "\n"
            x += 1
        backstory += str(x) + ") "
        return backstory

    def mb_ans_key_answers(self, row):
        dictionary = self.get_dictionary(row)
        x = 1
        answer_key = "\n\nANSWER KEY:\n\n" + str(x) + ") "
        
        for key in dictionary.keys():
            x += 1
            val = row[key]
            answer_key += val + "\n" + str(x) + ") "
        return answer_key
            
    # 07 Survey Response
    def mb_survey(self, row):
        '''
        survey style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        answers = self.answers_backstory
        backstory = ''
        x = 1
        for key in dictionary.keys():
            val = row[key]
            backstory += "Question " + str(x) + ": " + questions[key] + " ("
            comma = ""
            for ans in answers[key]:
                backstory += comma + ans
                comma = ", "
            backstory += ")\nAnswer " + str(x) + ": "
            if val in dictionary[key]:
                backstory += dictionary[key][val] + "\n"
            else:
                backstory += dictionary[key]['default'] + "\n"
            x += 1
        
        backstory += "Question " + str(x) + ": " # should be len(dictionary.keys()), right?
        return backstory

    # better way to do this?
    def get_answer_num(self, row):
        return len(self.get_dictionary(row)) + 1 

    # 08 Multiple Choice Response
    def mb_mult(self, row):
        '''
        mutliple choice style backstory dropping nans
        '''
        dictionary = self.get_dictionary(row)
        questions = self.questions
        answers_backstory = self.answers_backstory
        backstory = ''
        x = 1
        for key in dictionary.keys():
            val = row[key]
            backstory += "Question " + str(x) + ": " + questions[key] + "\n"
            backstory += self.format_mult(answers_backstory[key]) # includes "Correct Answer: "
            backstory += chr(65 + answers_backstory[key].index(val)) + "\n\n"
            x += 1
        
        backstory += "Question " + str(x) + ": " 
        return backstory
    
    def format_mult(self,answers_backstory):
        output = ""
        for ans in answers_backstory:
                output += chr(65 + answers_backstory.index(ans)) + ": " + str(ans) + "\n"
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
            mod_df_dict['age'].append(int(row['age']))
            mod_df_dict['gender'].append(str(row['gender'])[4:])
            mod_df_dict['education'].append(str(row['education'])[4:])
            mod_df_dict['ideology'].append(self.get_ideo(row['ideology'])[4:])
            mod_df_dict['income'].append(str(row['income'])[4:])
            mod_df_dict['religion'].append(str(row['religion'])[4:])
            mod_df_dict['race_ethnicity'].append(str(row['race_ethnicity'])[4:])
            mod_df_dict['marital_status'].append(int(row['marital_status']))

            # if False:
            #     maybe = {
            mod_df_dict['shot_or_stabbed'].append(str(row['shot_or_stabbed'])[4:])
            mod_df_dict['arrested'].append(str(row['arrested'])[4:])
            mod_df_dict['physical_fight'].append(str(row['physical_fight'])[4:])
            mod_df_dict['convicted_of_charges'].append(str(row['convicted_of_charges'])[4:]) #????
            mod_df_dict['sell_drugs'].append(str(row['sell_drugs'])[4:])
            mod_df_dict['counseling'].append(str(row['counseling'])[4:])
            mod_df_dict['sadness_family'].append(str(row['sadness_family'])[4:])
            mod_df_dict['worrying'].append(str(row['worrying'])[4:])
            mod_df_dict['suicide'].append(str(row['suicide'])[4:])
            mod_df_dict['optimism'].append(str(row['optimism'])[4:])
            mod_df_dict['happiness'].append(str(row['happiness'])[4:])
            mod_df_dict['fast_food'].append(str(row['fast_food']))
            mod_df_dict['hours_of_tv'].append(str(row['hours_of_tv']))
            mod_df_dict['individual_sports'].append(str(row['individual_sports']))
            mod_df_dict['smoked_cigarette'].append(str(row['smoked_cigarette'])[4:])
            mod_df_dict['physical_child_abuse'].append(str(row['physical_child_abuse'])[4:])
            mod_df_dict['age_of_first_drink'].append(str(row['age_of_first_drink']))
            mod_df_dict['car_accidents'].append(str(row['car_accidents'])[4:])
            mod_df_dict['drinking'].append(str(row['drinking'])[4:])
            mod_df_dict['prayer_in_private'].append(str(row['prayer_in_private'])[4:])
                # }
            
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
            'Yes': 'Yes', 
            'No': 'No'}

    def get_tokens_mult(self, answers):
        tokens = {}
        counter = 0
        for ans in answers:
            tokens[chr(65 + counter)] = ans
        return tokens
        # return {
        #     'A': 'Yes',
        #     'B': 'No', }

    def get_tokens_times(self):
        return {
            "Never" : "Never", 
            "1 or 2 times" : "1 or 2 times", 
            "3 or 4 times" : "3 or 4 times", 
            "5 or more times" : "5 or more times"}

    def get_tokens_charges(self):
        return {
            "No" : "No", 
            "Once" : "Once", 
            "More than once" : "More than once"}

    def get_tokens_occasion(self):
        return {
            "Never or rarely" : "Never or rarely", 
            "Sometimes" : "Sometimes", 
            "A lot of the time" : "A lot of the time",
            "Most of the time or all of the time" : "Most of the time or all of the time"}

    def get_tokens_agree(self):
        return {
            # "Strongly agree" 
            "Agree" : "Agree", 
            "Neither agree nor disagree" : "Neither agree nor disagree",
            "Disagree" : "Disagree",
            # "Strongly disagree"
            "Most of the time or all of the time" : "Most of the time or all of the time"}

    def get_tokens_suicide(self):
        return {
            "None" : "None",
            "Once" : "Once",
            "Twice" : "Twice",
            "3 or 4 times" : "3 or 4 times",
            "5 or 6 times" : "5 or 6 times"
        }

    def get_tokens_times7(self):
        return {
            "1 time" : "1 time",
            "2 times" : "2 times",
            "3 times" : "3 times",
            "4 times" : "4 times",
            "5 times" : "5 times", 
            "6 times" : "6 times",
            "7 or more times" : "7 or more times",
        }

    def get_tokens_abuse(self):
        return {
            "one time" : "one time",
            "two times" : "two times",
            "three times" : "three times",
            "six to ten times" : "six to ten times",
            "more than ten times" : "more than ten times",
            "this has never happened" : "this has never happened",
        }

    def get_tokens_prayer(self):
        return {
            "Never" : "Never",
            "Once a month" : "Once a month",
            "A few times a month" : "A few times a month",
            "Once a week" : "Once a week",
            "A few times a day" : "A few times a day",
            "Once a day" : "Once a day",
            "More than once a day" : "More than once a day",
        }



    def get_templates(self):
        pf_ans_or = self.answers_or(self.answers_dv["physical_fight"])
        pf_ans_paren = self.answers_paren(self.answers_dv["physical_fight"])
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

                "anskey1": (lambda row: (f"{self.mb_ans_key(row)}\n In the past year, has someone been shot or stabbed by you? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey2": (lambda row: (f"{self.mb_ans_key(row)}\n Was someone shot or stabbed by you in the past year? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey3": (lambda row: (f"{self.mb_ans_key(row)}\n If these are true about you, have you shot or stabbed someone in the past 12 months? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_yn()),

                "survey1": (lambda row: (f"{self.mb_survey(row)}\n In the past year, have you shot or stabbed someone? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),
                "survey2": (lambda row: (f"{self.mb_survey(row)}\n Has anyone been shot or stabbed by you in the past 12 months? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),

                "mult1": (lambda row: (f"{self.mb_mult(row)}\n You have shot or stabbed someone within 12 months from today. Is this true? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult(["Yes", "No"])),
                "mult2": (lambda row: (f"{self.mb_mult(row)}\n Can you honestly say that in the past 12 months, you have shot or stabbed someone? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult(["Yes", "No"])),
                "mult3": (lambda row: (f"{self.mb_mult(row)}\n In the past 12 months, have you shot or stabbed someone? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult(["Yes", "No"])),
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

                "anskey1": (lambda row: (f"{self.mb_ans_key(row)}\n Have you ever been arrested? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey2": (lambda row: (f"{self.mb_ans_key(row)}\n Have you ever been arrested by someone? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_yn()),
                "anskey3": (lambda row: (f"{self.mb_ans_key(row)}\n If these are true about you, have you been arrested? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_yn()),

                "survey1": (lambda row: (f"{self.mb_survey(row)}\n Is it true that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),
                "survey2": (lambda row: (f"{self.mb_survey(row)}\n Is it likely that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_yn()),

                "mult1": (lambda row: (f"{self.mb_mult(row)}\n You have been arrested before. Is this true? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult(["Yes", "No"])),
                "mult2": (lambda row: (f"{self.mb_mult(row)}\n Can you honestly say that you have been arrested? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult(["Yes", "No"])),
                "mult3": (lambda row: (f"{self.mb_mult(row)}\n Have you ever been arrested? \n{self.get_mult_yn()}"), 
                self.get_tokens_mult(["Yes", "No"])),
            },
            "physical_fight": {
                "first_person1": (lambda row: (f"{self.mb_first_person(row)}\n In the past 12 months, how often did you get into a serious physical fight? Answer {pf_ans_or}: "),  # why can't I use self.answers_or(self.answers_dv["physical_fight"])?
                self.get_tokens_times()),
                "first_person2": (lambda row: (f"{self.mb_first_person(row)}\n How frequently have you had a serious physical fight in the past year?"), 
                self.get_tokens_times()),
                "first_person3": (lambda row: (f"{self.mb_first_person(row)}\n How often have you had a serious physical fight with someone in the past year? Answer {pf_ans_or}"), 
                self.get_tokens_times()),

                # "third_person1": (lambda row: (f"{self.mb_third_person(row)}\n , has {self.GENDER_NEUTRAL_NAME} shot or stabbed someone?"), 
                # self.get_tokens_yn()),
                "third_person2": (lambda row: (f"{self.mb_third_person(row)}\n In the past year, how often has {self.GENDER_NEUTRAL_NAME} been in a serious physical fight?"),  
                self.get_tokens_times()),
                "third_person3": (lambda row: (f"{self.mb_third_person(row)}\n How frequently has {self.GENDER_NEUTRAL_NAME} been in a serious physical fight in the past 12 months? Answer {pf_ans_or}:"), 
                self.get_tokens_times()),
                
                "qa1": (lambda row: (f"{self.mb_qa(row)}\n In the past 12 months, how often did you get into a serious physical fight? \nA:"), 
                self.get_tokens_times()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n How frequently have you had a serious physical fight in the past year? \nA:"), 
                self.get_tokens_times()),
                "qa3": (lambda row: (f"{self.mb_qa(row)}\n What is the frequency in which you have had a serious physical fight in the past year? \nA:"), 
                self.get_tokens_times()),

                "qa1": (lambda row: (f"{self.mb_qa_exp(row)}\n In the past year, how frequently did you get into a serious physical fight? {pf_ans_paren} ? \nA:"), 
                self.get_tokens_times()),
                "qa2": (lambda row: (f"{self.mb_qa(row)}\n How often have you had a serious physical fight in the past 12 months? {pf_ans_paren} ? \nA:"), 
                self.get_tokens_times()),
                # "qa3": (lambda row: (f"{self.mb_qa(row)}\n have you been arrested  (Yes, No) ? \nA:"), 
                # self.get_tokens_yn()),
                
                "conv1": (lambda row: (f"{self.mb_qa(row)}\n How often have you, in the past year, gotten into a serious physical fight? P2:"), 
                self.get_tokens_times()),
                "conv2": (lambda row: (f"{self.mb_qa(row)}\n In the past 12 months, how frequently have you gotten into a serious physical fight? P2:"), 
                self.get_tokens_times()),
                # "conv3": (lambda row: (f"{self.mb_qa(row)}\n Did you shoot or stab someone ? P2:"), 
                # self.get_tokens_yn()),

                "anskey1": (lambda row: (f"{self.mb_ans_key(row)}\n How often have you been in a serious physical fight in the past year? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_times()),
                "anskey2": (lambda row: (f"{self.mb_ans_key(row)}\n How frequently have you had a serious physical fight with someone? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_times()),
                "anskey3": (lambda row: (f"{self.mb_ans_key(row)}\n If these are true about you, how often have you had a serious physical fight in the past 12 months? {self.mb_ans_key_answers(row)}"), 
                self.get_tokens_times()),

                "survey1": (lambda row: (f"{self.mb_survey(row)}\n What is the frequency in which you have had a serious physical fight in the past year? {pf_ans_paren} \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_times()),
                "survey2": (lambda row: (f"{self.mb_survey(row)}\n Is it likely that you have been arrested before? (Yes, No) \nAnswer {self.get_answer_num(row)}: "), 
                self.get_tokens_times()),

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