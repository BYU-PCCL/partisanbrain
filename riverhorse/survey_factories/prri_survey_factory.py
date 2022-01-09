from parent_dir import DatasetFactory
from survey_classes import PrriSurvey


class PrriFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        pass

    def get_templates(self):
        return {
        "voting_frequency" : {
            "finish_sentence" : (lambda row: (f"{make_backstory1(row)}\n\n"
                f"When asked how often I would say that I vote between the options of always, nearly always, in about half of elections, seldom, or never,\n\n"
                f"I would say that I vote"), {}),
            "delimiter_qa" : (lambda row: (f"{make_backstory1(row)}\n\n"
                f"Q: How often would you say you vote? Always, nearly always, in about half of elections, seldom, or never?\n\n"
                f"A: I would say that I vote"), {}),
            "0_shot_task_context_question" : (lambda row: (f"TASK: Consider the below demographic information and answer the following question.\n\n"
                f"CONTEXT: {make_backstory2(row)}\n\n"
                f"QUESTION: Would you say that you vote always, nearly always, in about half of elections, seldom, or never?\n\n"
                f"ANSWER:"), {}),
            "1_shot_task_context_question" : (lambda row: (f"TASK: Consider the below demographic information and answer the following question.\n\n"
                f"CONTEXT: {make_backstory2_shot(row)}\n\n"
                f"QUESTION: How often would you say that you vote?\n\n"
                f"ANSWER: Always\n\n"
                f"TASK: Consider the below demographic information and answer the following question.\n\n"
                f"CONTEXT: {make_backstory2(row)}\n\n"
                f"QUESTION: Would you say that you vote always, nearly always, in about half of elections, seldom, or never?\n\n"
                f"ANSWER:"), {}),
            "explicit_instructions" : (lambda row: (f"The PRRI is a nationally representative survey of voters in American Elections. Below are examples of respondents answering various questions. Please complete what you would guess the right answers to those questions to be.\n\n"
                f"{make_backstory3(row)}\n\n"
                f"Q: How often would you say that you vote?\n"
                f"A:"), {}),
            "implicit_instructions" : (lambda row: (f"P1: {make_backstory1(row)}\n"
                f"P2: When asked how often you vote, between the options of always, nearly always, in about half of elections, seldom, or never,\n"
                f"P1: I would say that I vote"), {}),
                "enumerated_response" : (lambda row: (f"{make_backstory1(row)} Between increased, decreased, or stay the same, federal spending on protecting the environment should be"), {}),
                
                "non_enumerated_response" : (lambda row: (f"{make_backstory1(row)} Federal spending on protecting the environment should be"), {}),

                "0_shot_first_person_backstory" : (lambda row: (f"{make_backstory1(row)} I think federal spending on protecting the environment should be"), {}),

                "1_shot_first_person_backstory" : (lambda row: (f"{make_backstory1_shot()} I think environmental protection is important.\n\n"
                                                                f"{make_backstory1(row)} I think federal spending on protecting the environment should be"), {}),

                "0_shot_chapter_quiz" : (lambda row: (f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
                                                f"1) According to the above backstory, would this person likely support green reforms?\n"
                                                f"2) According to the above backstory, would this person want the federal spending protecting the environment to be increased, decreased, or kept the same?\n\n"
                                                f"ANSWERS:\n1) Yes\n2)"), {}),
                
                "1_shot_chapter_quiz" : (lambda row: (f"CHAPTER QUIZ\n\nBACKSTORY:\n{make_backstory1(row)}\n\nQUESTIONS:\n"
                                                f"1) According to the above backstory, would this person likely support green reforms?\n"
                                                f"2) According to the above backstory, would this person want the federal spending protecting the environment to be increased, decreased, or kept the same?\n\n"
                                                f"ANSWERS:\n1) Yes\n2)"), {}),

                "0_shot_survey_response" : (lambda row: (f"{make_backstory4(row)}\n\nQuestion 11: Should federal spending on protecting the environment be increased, decreased, or kept the same?\nAnswer 11:"), {}),

                "1_shot_survey_response" : (lambda row: (f"{make_backstory4(row)}\n\nQuestion 11: Do you consider environment protection to be important to you?\n"
                                                        f"Answer 11: Yes\n\nQuestion 12: Should federal spending on protecting the environment be increased, decreased, or kept the same?\nAnswer 12:"), {}),

                "0_shot_heavy_delimited" : (lambda row: (f"Given the backstory and question, please answer the question appropriately.\n\n"
                                                f"\'\'\'Backstory -- {make_backstory1(row)}\'\'\', \'\'\'Should federal spending on protecting the environment be increased, decreased, or kept the same?\'\'\n\n"
                                                f" -> \'\'\'"), {}),

                "1_shot_heavy_delimited" : (lambda row: (f"Given the backstory and question, please answer the question appropriately.\n\n"
                                                f"\'\'\'Backstory -- {make_backstory1_shot()}\'\'\', \'\'\'Do you consider environment protection to be important to you?\'\'\n\n"
                                                f" -> \'\'\'Yes\'\'\'\n\n"
                                                f"\'\'\'Backstory -- {make_backstory1(row)}\'\'\', \'\'\'Should federal spending on protecting the environment be increased, decreased, or kept the same?\'\'\n\n"
                                                f" -> \'\'\'"), {}),

                "0_shot_multiple_choice" : (lambda row: (f"Background info: {make_backstory2(row)}\n\n"
                                                        f"Question 1: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
                                                        f"A: Increased\n"
                                                        f"B: Decreased\n"
                                                        f"C: Kept the same\n"
                                                        f"Answer 1:"), {'Increased' : ['A', 'Increased'], 'Decreased' : ['B', 'Decreased'], 'Kept the same' : ['C', 'Kept']}),
                
                "1_shot_multiple_choice" : (lambda row: (f"Background info: {make_backstory2(row)}\n\n"
                                                        f"Question 1: Do you consider environment protection to be important to you?\n"
                                                        f"A: Yes\n"
                                                        f"B: No\n"
                                                        f"C: Indifferent\n"
                                                        f"Answer 1: Yes\n\n"
                                                        f"Question 2: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
                                                        f"A: Increased\n"
                                                        f"B: Decreased\n"
                                                        f"C: Kept the same\n"
                                                        f"Answer 2:"), {}),
                
                "0_shot_direct_mapping" : (lambda row: (f"{make_backstory1(row)}\n\n"
                                                        f"Question 1: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
                                                        f"Answer 1 (Increased, Decreased, Kept the same):"), {}),
                
                "1_shot_direct_mapping" : (lambda row: (f"{make_backstory1_shot()}\n\n"
                                                        f"Question 1: Do you consider environment protection to be important to you?\n"
                                                        f"Answer 1: Yes"
                                                        f"{make_backstory1(row)}\n\n"                                                      
                                                        f"Question 2: Should federal spending on protecting the environment be increased, decreased, or kept the same?\n"
                                                        f"Answer 2 (Increased, Decreased, Kept the same):"), {}),

            }
        }

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


if __name__ == "__main__":
    factory = PrriFactory(PrriSurvey())
