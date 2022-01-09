from parent_dir import DatasetFactory
from survey_classes import PrriSurvey


class PrriFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        pass

    def make_backstory1(self):
        party_lambda = lambda row: ("don\'t identify as a member of a major party" if "None" in (row['party']) else "identify as a member of the " + row['party'])
        # create a generic backstory using values in the demographics
        return (lambda row: f"I am {row['age']} years old. I am {row['gender'].lower()}."
                            f"I {party_lambda(row)}."
                            f"My education is {row['education'].lower()}. I am {row['ideology'].lower()}."
                            f"My total income is {row['income']}."
                            f"My religion is described as {row['religion']}."
                            f"I am {row['race_ethnicity']}. I am from the {row['region']}."
                            f"I am {row['marital_status'].lower()}."
                            )
    
    def make_backstory1_shot(self):
        return "I am 21 years old. I am female. I identify as a member of the Democratic party. \
                My education is some college but no degree. My total income is $40,000. \
                My religion is described as not religious. I am white, not-hispanic. I am from the West. \
                I am never married."

    def make_backstory2(self):
        return (lambda row: f"Age: {row['age']}, Gender: {row['gender']}, Political Affiliation: {row['party']}"
                            f"Education: {row['education']}, Ideology: {row['ideology']}, Total Income: {row['income']}, Religion: {row['religion']},"
                            f"Race/Ethnicity: {row['race_ethnicity']}, Region: {row['region']}, Marital Status: {row['marital_status']}")

    def make_backstory2_shot(self):
        return "Age: 21, Gender: Female, Political Affiliation: Democratic Party, Education: Some college but no degree, \
                Ideology: Extremely liberal, Income: $40,000, Religion: Not religious, Race/Ethnicity: Black, non-Hispanic, \
                Region: Northeast, Marital Status: Never married"

    def make_backstory3(self):
        return (lambda row: f"Q: What is your age?\nA: {row['age']}\n\nQ: What is your gender?\nA: {row['gender']}\n\n"
                            f"Q: What is your political affiliation?\nA: {row['party']}\n\nQ: What is your education?\nA: {row['education']}"
                            f"Q: What is your ideology?\nA: {row['ideology']}\n\nQ: What is your income?\nA: {row['income']}\n\n"
                            f"Q: What is your religion?\nA: {row['religion']}\n\n"
                            f"Q: What is your race/ethnicity?\nA: {row['race_ethnicity']}\n\nQ: What region of the country are your from?\nA: {row['region']}"
                            f"Q: What is your marital status?\nA: {row['marital_status']}")

    def make_backstory4(self):
        return (lambda row: f"Question 1: What is your age?\nAnswer 1: {row['age']}\n\nQuestion 2: What is your gender?\nAnswer 2: {row['gender']}\n\n"
                            f"Question 3: What is your political affiliation?\nAnswer 3: {row['party']}\n\nQuestion 4: What is your education?\nAnswer 4: {row['education']}"
                            f"Question 5: What is your ideology?\nAnswer 5: {row['ideology']}\n\nQuestion 6: What is your income?\nAnswer 6: {row['income']}\n\n"
                            f"Question 7: What is your religion?\nAnswer 7: {row['religion']}\n\n"
                            f"Question 8: What is your race/ethnicity?\nAnswer 8: {row['race_ethnicity']}\n\nQuestion 9: What region of the country are your from?\nAnswer 9: {row['region']}"
                            f"Question 10: What is your marital status?\nAnswer 10: {row['marital_status']}")

    def get_templates(self):
        return {
        "voting_frequency" : {
            "finish_sentence" : (lambda row: (f"{make_backstory1(row)}\n\n"
                f"When asked how often I would say that I vote between the "
                f"options of always, nearly always, in about half of elections, seldom, or never,\n\n"
                f"I would say that I vote"), {}),

            "delimiter_qa" : (lambda row: (f"{make_backstory1(row)}\n\n"
                f"Q: How often would you say you vote? Always, nearly always, "
                f"in about half of elections, seldom, or never?\n\n"
                f"A: I would say that I vote"), {}),

            "0_shot_task_context_question" : (lambda row: (f"TASK: Consider the below demographic "
                f"information and answer the following question.\n\n"
                f"CONTEXT: {make_backstory2(row)}\n\n"
                f"QUESTION: Would you say that you vote always, nearly always, in about half of elections, "
                f"seldom, or never?\n\n"
                f"ANSWER:"), {}),

            "1_shot_task_context_question" : (lambda row: (f"TASK: Consider the below demographic "
                "information and answer the following question.\n\n"
                f"CONTEXT: {make_backstory2_shot(row)}\n\n"
                f"QUESTION: How often would you say that you vote?\n\n"
                f"ANSWER: Always\n\n"
                f"TASK: Consider the below demographic information and answer the following question.\n\n"
                f"CONTEXT: {make_backstory2(row)}\n\n"
                f"QUESTION: Would you say that you vote always, nearly always, in about half of elections, "
                f"seldom, or never?\n\n"
                f"ANSWER:"), {}),

            "explicit_instructions" : (lambda row: (f"The PRRI is a nationally representative survey of voters"
                f" in American Elections. Below are examples of respondents answering various questions. "
                f"Please complete what you would guess the right answers to those questions to be.\n\n"
                f"{make_backstory3(row)}\n\n"
                f"Q: How often would you say that you vote?\n"
                f"A:"), {}),

            "implicit_instructions" : (lambda row: (f"P1: {make_backstory1(row)}\n"
                f"P2: When asked how often you vote, between the options of always, "
                f"nearly always, in about half of elections, seldom, or never,\n"
                f"P1: I would say that I vote"), {}),

            "enumerated_response" : (lambda row: (f"{make_backstory1(row)} Between voting always," 
                f"nearly always, in about half of elections, seldom, or never, I would say that I vote"), {}),
                
            "non_enumerated_response" : (lambda row: (f"{make_backstory1(row)} I would say that I vote"), {}),

            "0_shot_first_person_backstory" : (lambda row: (f"{make_backstory1(row)} When asked how often I vote, "
                f"I would say that I vote"), {}),

            "1_shot_first_person_backstory" : (lambda row: (f"{make_backstory1_shot()} I would say that I vote always.\n\n"
                f"{make_backstory1(row)} Between the options of always, almost always, in about half of elections,"
                f" seldom, or never, "
                f"I would say that I vote"), {}),

            "0_shot_chapter_quiz" : (lambda row: (f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTION:\n"
                f"According to the above backstory, how often would this person be likely to vote?\n"
                f"A) Always\n"
                f"B) Almost always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"ANSWER:"), {}),
                
            "1_shot_chapter_quiz" : (lambda row: (f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1_shot(row)}\n\nQUESTION:\n"
                f"According to the above backstory, how often would this person be likely to vote?\n"
                f"A) Always\n"
                f"B) Almost always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"ANSWER:\n"
                f"A"
                f"\n\nBACKSTORY:\n{make_backstory1_shot(row)}\n\nQUESTION:\n"
                f"According to the above backstory, how often would this person be likely to vote?\n"
                f"A) Always\n"
                f"B) Almost always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"ANSWER:"), {'Always' : ['A', 'Always'], 'Almost always' : ['B', 'Almost'], 
                'In about half of elections' : ['C', 'In'], 'Seldom':['D','Seldom']}),

            "0_shot_survey_response" : (lambda row: (f"{make_backstory4(row)}\n\nQuestion 11: Do you vote always, "
                f"almost always, in about half of elections, seldom, or never?\nAnswer 11:"), {}),

            "1_shot_survey_response" : (lambda row: (f"{make_backstory4(row)}\n\nQuestion 11: Do you have the "
                f"ability to vote?\n"
                f"Answer 11: Yes\n\nQuestion 12: Do you vote always, almost always, in about half of elections, "
                f"seldom, or never?\nAnswer 12:"), {}),

            "0_shot_heavy_delimited" : (lambda row: (f"Given the backstory and question, "
                f"please answer the question appropriately.\n\n"
                f"\'\'\'Backstory -- {make_backstory1(row)}\'\'\', \'\'\'Would you say that you vote always, "
                f"almost always, in about half of elections, seldom, or never?\'\'\n\n"
                f" -> \'\'\'"), {}),

            "1_shot_heavy_delimited" : (lambda row: (f"Given the backstory and question, "
                f"please answer the question appropriately.\n\n"
                f"\'\'\'Backstory -- {make_backstory1_shot()}\'\'\', \'\'\'Do you consider voting to be "
                f"important to you?\'\'\n\n"
                f" -> \'\'\'Yes\'\'\'\n\n"
                f"\'\'\'Backstory -- {make_backstory1(row)}\'\'\', \'\'\'Do you vote always, almost always,"
                f" in about half of elections, seldom, or never?\'\'\n\n"
                f" -> \'\'\'"), {}),

            "0_shot_multiple_choice" : (lambda row: (f"Background info: {make_backstory2(row)}\n\n"
                f"Question 1: Do you vote always, almost always, in about half of elections, seldom, or never?\n"
                f"A: Always\n"
                f"B: Almost Always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"ANSWER:"), {'Always' : ['A', 'Always'], 'Almost always' : ['B', 'Almost'], 
                'In about half of elections' : ['C', 'In'], 'Seldom':['D','Seldom']}),
                
            "1_shot_multiple_choice" : (lambda row: (f"Background info: {make_backstory1_shot(row)}\n\n"
                f"Question 1: How often do you vote?\n"
                f"A: Always\n"
                f"B: Almost Always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"Answer: \nA\n"
                f"Background info: {make_backstory1(row)}\n\n"
                f"Question 1: How often do you vote?\n"
                f"A: Always\n"
                f"B: Almost Always\n"
                f"C) In about half of elections\n"
                f"D) Seldom\n"
                f"Answer:"), {'Always' : ['A', 'Always'], 'Almost always' : ['B', 'Almost'], 
                'In about half of elections' : ['C', 'In'], 'Seldom':['D','Seldom']}),
                
            "0_shot_direct_mapping" : (lambda row: (f"{make_backstory1(row)}\n\n"
                f"Question 1: Do you vote always, almost always, in about half of elections, seldom, or never?\n"
                f"Answer 1 (always, almost always, in about half of elections, seldom, never):"), {}),
                
            "1_shot_direct_mapping" : (lambda row: (f"{make_backstory1_shot()}\n\n"
                f"Question 1: Do you consider voting important to you?\n"
                f"Answer 1: Yes"
                f"{make_backstory1(row)}\n\n"                                                      
                f"Question 2: Do you vote always, almost always, in about half of elections, seldom, or never?\n"
                f"Answer 2 (always, almost always, in about half of elections, seldom, never):"), {}),

            }
        }

if __name__ == "__main__":
    factory = PrriFactory(PrriSurvey())
