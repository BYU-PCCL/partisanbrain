from ..dataset_factory import DatasetFactory
from ..surveys.prri_survey import PrriSurvey
from pdb import set_trace as breakpoint
import numpy as np

race_dict = {
    'White, non-Hispanic': 'white',
    'Asian, non-Hispanic': 'Asian',
    'Black, non-Hispanic': 'Black',
    'Other, non-Hispanic': np.nan,
    'Two plus, non-Hispanic': 'multiracial',
    'Hispanic': 'Hispanic',
}

# add I before
marriage_dict = {
    'Never married': 'have never been married',
    'Married': 'am married',
    'Divorced': 'am divorced',
    'Living with partner': 'live with my partner',
    'Widowed': 'am widowed',
    'Separated': 'am separated',
}

# I am
religion_dict = {
    'Just Christian': 'Christian',
    'Buddhist': 'Buddhist',
    'Mormon (Church of Jesus Christ of Latter-day Saints/LDS)': 'Mormon',
    'Agnostic (not sure if there is a God)': 'agnostic',
    'Jewish (Judaism)': 'Jewish',
    'Nothing in particular': 'not religious',
    'Roman Catholic (Catholic)': 'Catholic',
    'Protestant (Baptist, Methodist, non-denominational, Lutheran, Presbyterian, Pentecostal, Episcopalian, Reformed, Church)': 'Protestant',
    'Atheist (do not believe in God)': 'atheist',
    'Hindu': 'Hindu',
    'Muslim (Islam)': 'Muslim',
    'Something else': np.nan,
    'Orthodox (Greek, Russian or some other Orthodox church)': 'Orthodox Christian',
    'Skipped on web': np.nan,
    'Unitarian (Universalist)': 'Unitarian',
    'Refused': np.nan,
    "Don't know (VOL.)": np.nan,
}

# Ideologically, I am
ideology_dict = {
    'Moderate': 'moderate',
    'Liberal': 'liberal',
    'Very liberal': 'very liberal',
    'Conservative': 'conservative',
    'Very conservative': 'very conservative',
    'Skipped on web': np.nan,
    "Don't know (VOL.)": np.nan,
    'Refused': np.nan,
}

# party dictionary
# 'A Democrat', 'A Republican', 'An Independent', 'Other [SPECIFY]', 'Skipped on web', 'Refused', "Don't know (VOL.)"
# Politically, I 
party_dict = {
    'A Democrat': 'am a Democrat',
    'A Republican': 'am a Republican',
    'An Independent': 'am an Independent',
    'Other [SPECIFY]': "don\'t identify as a member of a major party",
    'Skipped on web': np.nan,
    'Refused': np.nan,
    "Don't know (VOL.)": np.nan,
}

party_dict2 = {
    'A Democrat': 'Democrat',
    'A Republican': 'Republican',
    'An Independent': 'Independent',
    'Other [SPECIFY]': "Other",
    'Skipped on web': np.nan,
    'Refused': np.nan,
    "Don't know (VOL.)": np.nan,
}

# '$25,000-$29,999', '$150,000-$174,999', '$60,000-$74,999', '$35,000-$39,999', '$175,000-$199,999', '$75,000-$84,999', '$5,000-$9,999', '$40,000-$49,999', '$50,000-$59,999', '$20,000-$24,999', '$10,000-$14,999', '$85,000-$99,999', '$15,000-$19,999', '$100,000-$124,999', '$30,000-$34,999', 'Less than $5,000', '$125,000-$149,999', '$200,000 or more'
# My annual income is _ a year
income_dict = {
    '$25,000-$29,999': 'between $25,000 and $30,000',
    '$150,000-$174,999': 'between $150,000 and $175,000',
    '$60,000-$74,999': 'between $60,000 and $75,000',
    '$35,000-$39,999': 'between $35,000 and $40,000',
    '$175,000-$199,999': 'between $175,000 and $200,000',
    '$75,000-$84,999': 'between $75,000 and $85,000',
    '$5,000-$9,999': 'between $5,000 and $10,000',
    '$40,000-$49,999': 'between $40,000 and $50,000',
    '$50,000-$59,999': 'between $50,000 and $60,000',
    '$20,000-$24,999': 'between $20,000 and $25,000',
    '$10,000-$14,999': 'between $10,000 and $15,000',
    '$85,000-$99,999': 'between $85,000 and $100,000',
    '$15,000-$19,999': 'between $15,000 and $20,000',
    '$100,000-$124,999': 'between $100,000 and $125,000',
    '$30,000-$34,999': 'between $30,000 and $35,000',
    'Less than $5,000': 'less than $5,000',
    '$125,000-$149,999': 'between $125,000 and $150,000',
    '$200,000 or more': 'more than $200,000',
}

# region dictionary
# 'FL', 'CA', 'NJ', 'PA', 'TX', 'NV', 'WA', 'IN', 'NY', 'NC', 'RI', 'IL', 'TN', 'DE', 'ME', 'MA', 'IA', 'MN', 'AZ', 'GA', 'MI', 'CO', 'SD', 'MO', 'CT', 'OH', 'WI', 'WY', 'ID', 'VA', 'MD', 'UT', 'NE', 'LA', 'OK', 'AL', 'NM', 'OR', 'KS', 'NH', 'KY', 'AR', 'SC', 'MS', 'MT', 'WV', 'ND', 'HI', 'VT', 'DC', 'AK'
region_dict = {
    'FL': 'Florida',
    'CA': 'California',
    'NJ': 'New Jersey',
    'PA': 'Pennsylvania',
    'TX': 'Texas',
    'NV': 'Nevada',
    'WA': 'Washington',
    'IN': 'Indiana',
    'NY': 'New York',
    'NC': 'North Carolina',
    'RI': 'Rhode Island',
    'IL': 'Illinois',
    'TN': 'Tennessee',
    'DE': 'Delaware',
    'ME': 'Maine',
    'MA': 'Massachusetts',
    'IA': 'Iowa',
    'MN': 'Minnesota',
    'AZ': 'Arizona',
    'GA': 'Georgia',
    'MI': 'Michigan',
    'CO': 'Colorado',
    'SD': 'South Dakota',
    'MO': 'Missouri',
    'CT': 'Connecticut',
    'OH': 'Ohio',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
    'ID': 'Idaho',
    'VA': 'Virginia',
    'MD': 'Maryland',
    'UT': 'Utah',
    'NE': 'Nebraska',
    'LA': 'Louisiana',
    'OK': 'Oklahoma',
    'AL': 'Alabama',
    'NM': 'New Mexico',
    'OR': 'Oregon',
    'KS': 'Kansas',
    'NH': 'New Hampshire',
    'KY': 'Kentucky',
    'AR': 'Arkansas',
    'SC': 'South Carolina',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'WV': 'West Virginia',
    'ND': 'North Dakota',
    'HI': 'Hawaii',
    'VT': 'Vermont',
    'DC': 'Washington, D.C.',
    'AK': 'Alaska',
}
#'Associate degree', 'Some college, no degree', "Master's degree", 'High school graduate - high school diploma or the equivalent (GED)', "Bachelor's degree", '11th grade', '12th grade - no diploma', 'Professional or doctorate degree', 'Seventh or eighth grade', '10th grade', 'Ninth grade', 'Fifth or sixth grade', 'No formal education', 'First, second, third or fourth grade'
#I
educ_dict = {
    "Associate degree": "have an associate's degree",
    "Some college, no degree": "did some college",
    "Master's degree": "have a master's degree",
    "High school graduate - high school diploma or the equivalent (GED)": "got my high school diploma",
    "Bachelor's degree": "have a bachelor's degree",
    "11th grade": "dropped out of high school",
    "12th grade - no diploma": "dropped out of high school",
    "Professional or doctorate degree": "went to graduate school",
    "Seventh or eighth grade": "didn't go to high school",
    "10th grade": "dropped out of high school",
    "Ninth grade": "dropped out of high school",
    "Fifth or sixth grade": "didn't go to high school",
    "No formal education": "didn't go to high school",
    "First, second, third or fourth grade": "didn't go to high school"
}

#'Republican', 'Democrat', 'Other, please specify','Skipped on web', 'Refused'
#I lean
lean_dict = {
            np.nan: '',
            'Republican': ", but I lean Republican",
            'Democrat': ", but I lean Democrat",
            'Other, please specify': '',
            'Skipped on web': '',
            'Refused': '',
            "Don't know (VOL.)": '',
            }


class PrriFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        # party is list. remove first element of list and make the 'party' column, and make the second element of list 'leaning' column
        df['leaning'] = df['party'].apply(lambda x: x[1])
        df['party'] = df['party'].apply(lambda x: x[0])
        # change age to int
        df['age'] = df['age'].apply(lambda x: int(x))

        # use dictionary to map race_ethnicity to race_processed
        df['race_ethnicity_processed'] = df['race_ethnicity'].replace(race_dict)
        # marriage
        df['marital_status_processed'] = df['marital_status'].replace(marriage_dict)
        # ideology
        df['ideology_processed'] = df['ideology'].replace(ideology_dict)
        # party
        df['party_processed'] = df['party'].replace(party_dict)
        # income
        df['income_processed'] = df['income'].replace(income_dict)
        # region
        df['region_processed'] = df['region'].replace(region_dict)
        #education
        df['education_processed'] = df['education'].replace(educ_dict)
        #lean
        df['leaning_processed'] = df['leaning'].replace(lean_dict)
        #religion
        df['religion_processed'] = df['religion'].replace(religion_dict)

        #TODO: Rename the values in the DV columns for good templating, like so:
        # 'Always', 'Nearly always', 'In about half of elections', 'Seldom', 'Skipped on web', 'Never'
        voting_dict = {
            "Always": "always",
            "Nearly always": "nearly always",
            "In about half of elections": "in about half of elections",
            "Seldom": "seldom",
            "Skipped on web": np.nan,
            "Never": "never"
        }
        df['voting_frequency'] = df['voting_frequency'].replace(voting_dict)
        

        to_drop = [col for col in df.columns if col.endswith('_processed')]
        #Restrict this list to columns already in df
        df.dropna(subset=to_drop, inplace=True)
        return df

    def make_backstory1(self, row):
        backstory_list = []
        # party_lambda = lambda party, leaning: f"I lean {don\'t identify as a member of a major party" if "None" in (row['party']) else "identify as a member of the " + row['party']
    
        backstory_list.append(f"I am {row['age']} years old.")
        backstory_list.append(f"I am {row['gender'].lower()}.")
        backstory_list.append(f"Politically, I {row['party_processed']}{row['leaning_processed']}.")
        backstory_list.append(f"Ideologically, I am {row['ideology_processed'].lower()}.")
        backstory_list.append(f"I {row['education_processed'].lower()}.")
        backstory_list.append(f"My total income is {row['income_processed']}.")
        backstory_list.append(f"I am {row['religion_processed']}.")
        backstory_list.append(f"I am {row['race_ethnicity_processed']}.")
        backstory_list.append(f"I live in {row['region_processed']}.")
        backstory_list.append(f"I {row['marital_status_processed'].lower()}.")
        # create a generic backstory using values in the demographics
        return " ".join(backstory_list)
    
    def make_backstory2(self, row):
        return (f"Age: {row['age']}, Gender: {row['gender']}, Political Affiliation: {party_dict2[row['party']]}, "
                            f"Education: {row['education']}, Ideology: {row['ideology']}, Total Income: {row['income']}, Religion: {row['religion']}, "
                            f"Race/Ethnicity: {row['race_ethnicity']}, State: {row['region_processed']}, Marital Status: {row['marital_status']}")

    def make_backstory3(self, row):
        return (f"Q: What is your age?\nA: {row['age']}\n\nQ: What is your gender?\nA: {row['gender']}\n\n"
                            f"Q: What is your political affiliation?\nA: {row['party']}\n\nQ: What is your education?\nA: {row['education']}"
                            f"Q: What is your ideology?\nA: {row['ideology']}\n\nQ: What is your income?\nA: {row['income']}\n\n"
                            f"Q: What is your religion?\nA: {row['religion']}\n\n"
                            f"Q: What is your race/ethnicity?\nA: {row['race_ethnicity']}\n\nQ: What state do you live in?\nA: {row['region_processed']}\n\n"
                            f"Q: What is your marital status?\nA: {row['marital_status']}")

    def make_backstory4(self, row):
        return (f"Question 1: What is your age?\nAnswer 1: {row['age']}\n\nQuestion 2: What is your gender?\nAnswer 2: {row['gender']}\n\n"
                            f"Question 3: What is your political affiliation?\nAnswer 3: {row['party']}\n\nQuestion 4: What is your education?\nAnswer 4: {row['education']}"
                            f"Question 5: What is your ideology?\nAnswer 5: {row['ideology']}\n\nQuestion 6: What is your income?\nAnswer 6: {row['income']}\n\n"
                            f"Question 7: What is your religion?\nAnswer 7: {row['religion']}\n\n"
                            f"Question 8: What is your race/ethnicity?\nAnswer 8: {row['race_ethnicity']}\n\nQuestion 9: What region of the country are your from?\nAnswer 9: {row['region']}"
                            f"Question 10: What is your marital status?\nAnswer 10: {row['marital_status']}")

    def get_templates(self):
        # voting_frequency_dict = ['always', 'almost always', 'in about half of elections', 'seldom', 'never']
        voting_frequency_dict = {
            'always': 'always',
            'almost always': 'almost always',
            'in about half of elections': 'in about half of elections',
            'seldom': 'seldom',
            'never': 'never'
        }
        return {
        "voting_frequency" : {
            "finish_sentence" : (lambda row: (f"{self.make_backstory1(row)} "
                f"If asked how often I vote ("
                f"always, nearly always, in about half of elections, seldom, or never), "
                f"I would say that I vote"), voting_frequency_dict),

            "finish_sentence_test5shot" : (lambda row:  self.get_shots('voting_frequency', 'finish_sentence', n=5, sep='.\n\n') + (
                f"\n\n{self.make_backstory1(row)} "
                f"If asked how often I vote ("
                f"always, nearly always, in about half of elections, seldom, or never), "
                f"I would say that I vote"), voting_frequency_dict),

            "delimiter_qa" : (lambda row: (f"{self.make_backstory1(row)}\n\n"
                f"Q: How often would you say you vote? Always, nearly always, "
                f"in about half of elections, seldom, or never?\n\n"
                f"A: I would say that I vote"), voting_frequency_dict),

            "context_question" : (lambda row: (f"TASK: Consider the below demographic "
                f"information and answer the following question:\n"
                f"CONTEXT: {self.make_backstory2(row)}\n"
                f"QUESTION: Would you say that you vote always, nearly always, in about half of elections, "
                f"seldom, or never?\n"
                f"ANSWER:"), voting_frequency_dict),

            "task_context_question_1_shot" : (lambda row: 
                self.get_shots('voting_frequency', '0_shot_task_context_question', n=1, sep='.\n\n') + 
                (f"\n\nTASK: Consider the below demographic information and answer the following question.\n\n"
                f"CONTEXT: {self.make_backstory2(row)}\n\n"
                f"QUESTION: Would you say that you vote always, nearly always, in about half of elections, "
                f"seldom, or never?\n"
                f"ANSWER:"), voting_frequency_dict),

            "explicit_instructions" : (lambda row: (f"PRRI (Public Religion Research Institute) is a nonprofit, nonpartisan organization dedicated to conducting independent research at the intersection of religion, culture, and public policy. "
                f"Below are a respondent's answers to various questions in a PRRI survey.\n\n"
                f"{self.make_backstory3(row)}\n\n"
                f"Q: How often would you say that you vote (always, nearly always, in about half of elections, "
                f"seldom, or never)?\n"
                f"A:"), voting_frequency_dict),

            "implicit_instructions" : (lambda row: (f"P1: {self.make_backstory1(row)}\n"
                f"P2: Between the options of always, "
                f"nearly always, in about half of elections, seldom, or never, how often do you vote?\n"
                f"P1: I would say that I vote"), voting_frequency_dict),

            "enumerated_response" : (lambda row: (f"{self.make_backstory1(row)} Between the options of always, " 
                f"nearly always, in about half of elections, seldom, or never, I would say that I vote"), voting_frequency_dict),
                

            "enumerated_response_5shot" : (lambda row: self.get_shots('voting_frequency', 'enumerated_response', n=5, sep='.\n\n') + (
                f"\n\n{self.make_backstory1(row)} Between the options of always, almost always, in about half of elections,"
                f" seldom, or never, "
                f"I would say that I vote"), voting_frequency_dict),

            "quiz" : (lambda row: (f"QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTION:\n"
                f"According to the above backstory, how often would this person be likely to vote?\n"
                f"A) Always\n"
                f"B) Almost always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"ANSWER:"), voting_frequency_dict),
                
            "quiz_1shot" : (lambda row: self.get_shots('voting_frequency', 'quiz', n=1, sep='.\n\n') + (
                f"QUIZ\n\nBACKSTORY:\n{self.make_backstory1(row)}\n\nQUESTION:\n"
                f"According to the above backstory, how often would this person be likely to vote?\n"
                f"A) Always\n"
                f"B) Almost always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"ANSWER:"), {'Always' : ['A', 'Always'], 'Almost always' : ['B', 'Almost'], 
                'In about half of elections' : ['C', 'In'], 'Seldom':['D','Seldom']}),

            "survey_response" :  (lambda row: (f"SURVEY_RESPONSE\n\n{self.make_backstory4(row)}\n\nQuestion 11: Do you vote always, "
                f"almost always, in about half of elections, seldom, or never?\nAnswer 11:"), voting_frequency_dict),

            "survey_response_3shot" : (lambda row: self.get_shots('voting_frequency', 'survey_response', n=3, sep='.\n\n') + (
                f"SURVEY_RESPONSE\n\n{self.make_backstory4(row)}\n\nQuestion 11: Do you vote always, almost always, in about half of elections, "
                f"seldom, or never?\nAnswer 11:"), voting_frequency_dict),

            "heavy_delimited" : (lambda row: (f"Given the backstory and question, "
                f"please answer the question appropriately.\n\n"
                f"\'\'\'Backstory -- {self.make_backstory1(row)}\'\'\', \'\'\'Would you say that you vote always, "
                f"almost always, in about half of elections, seldom, or never?\'\'\n\n"
                f" -> \'\'\'"), voting_frequency_dict),

            "heavy_delimited_1shot" : (lambda row: self.get_shots('voting_frequency', 'heavy_delimited', n=1, sep='.\n\n') + (
                f"Given the backstory and question, "
                f"please answer the question appropriately.\n\n"
                f"\'\'\'Backstory -- {self.make_backstory1(row)}\'\'\', \'\'\'Do you vote always, almost always,"
                f" in about half of elections, seldom, or never?\'\'\n\n"
                f" -> \'\'\'"), voting_frequency_dict),

            "multiple_choice" : (lambda row: (f"Demographic profile of a voter: {self.make_backstory2(row)}\n\n"
                f"Question: How often does this voter vote?\n"
                f"A: Always\n"
                f"B: Almost Always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"Answer:"), {'Always' : ['A', 'Always'], 'Almost always' : ['B', 'Almost'], 
                'In about half of elections' : ['C', 'In'], 'Seldom':['D','Seldom']}),
                
            "multiple_choice_2shot" : (lambda row: self.get_shots('voting_frequency', 'multiple_choice', n=2, sep='.\n\n') + (
                f"\n\nDemographic profile of a voter: {self.make_backstory2(row)}\n\n"
                f"Question: How often does this voter vote?\n"
                f"A: Always\n"
                f"B: Almost Always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"Answer:"), {'Always' : ['A', 'Always'], 'Almost always' : ['B', 'Almost'], 
                'In about half of elections' : ['C', 'In'], 'Seldom':['D','Seldom']}),
                
            "introspect" : (lambda row: (f"{self.make_backstory1(row)}\n\n"
                f"Question: Do I vote always, almost always, in about half of elections, seldom, or never?\n"
                f"Answer (always, almost always, in about half of elections, seldom, never): I vote"), voting_frequency_dict),
                
            "introspect_2shot" : (lambda row: self.get_shots('voting_frequency', 'introspect', n=2, sep='.\n\n') + (
                f"{self.make_backstory1(row)}\n"                                                      
                f"Question: Do I vote always, almost always, in about half of elections, seldom, or never?\n"
                f"Answer (always, almost always, in about half of elections, seldom, never): I vote"), voting_frequency_dict),

            "introspect_mq" : (lambda row: (f"Question 1: Who am I?\n"
                f"Answer 1: {self.make_backstory1(row)}\n\n"
                f"Question 2: Do I consider voting important?\n"
                f"Answer 2 (yes, no): Yes\n\n"
                f"Question 3: How often do I vote?\n"
                f"Answer 3 (always, almost always, in about half of elections, seldom, never): I vote"), voting_frequency_dict),
            },

        }

if __name__ == "__main__":
    factory = PrriFactory(PrriSurvey(), n=10)
    factory.sample_templates(factory.survey_obj.df, dvs = 'voting_frequency'.split())