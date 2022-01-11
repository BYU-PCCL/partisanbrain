from ..dataset_factory import DatasetFactory
from ..surveys.baylor_survey import BaylorSurvey


class BaylorFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        pass

    def education_func (row):
        if "High school graduate (Grade 12 with diploma or GED certificate)" == row['education']: "completed high school"
        else if "Some college" == row['education']: "completed some college"
        else if "Four year bachelor's degree from a college or university (e.g., BS, BA, AB)" == row['education']: "a bachelor\'s degree"
        else if "Postgraduate" == row['education']: "completed postgraduate education"
        else : "no high school diploma"

    def religion_func (row):
        if "Other" == row['religion']: "I am religious"
        else if "Asian Folk Religion" == row['religion']: "I am part of an asian folk religion"
        else if "Non-denominational Christian" == row['religion']: "I am a non denominational christian"
        else if "No religion" == row['religion']: "I am not religious"
        else if "Don\'t know" == row['religion']: "I don't know my religious beliefs"
        else : "I am part of the " + row['education'] + " faith"

    def backstory1(row):
        return ( f"I am {row['age']} years old. I am {row['gender'].lower()}. "
                 f"I consider myself a {row['party']}. "
                 f"I have {education_lambda(row)}. I am {row['ideology'].lower()}. "
                 f"My total income is {row['income']}. "
                 f"{religion_lambda(row)}. "
                 f"I am {row['race_ethnicity']}. "
                 f"I am {row['marital_status'].lower()}.")

    def backstory1_shot(self):
        return "I am 21 years old. I am female. I identify as a member of the Democratic party. I have completed some college. My total income is $40,000. I am not religious. I am white. I am from the West. I have never been married."

    def backstory2(row):
        return (f"Q: How old are you?\n"
                f"A: I am {row['age']} years old.\n"
                f"Q: What is your gender?\n"
                f"A: I am {row['gender'].lower()}.\n"
                f"Q: Do you think of yourself as a Republican, Democrat, or Independent?\n"
                f"A: I consider myself a {row['party']}.\n"
                f"Q: What is the highest level of school you have completed or the highest degree you have received?\n"
                f"A: I have {education_func(row)}.\n"
                f"Q: How would you describe yourself politically?\n"
                f"A: I am {row['ideology'].lower()}.\n"
                f"Q: What was your total household income last year?\n"
                f"A: My total income is {row['income']}.\n"
                f"Q: With what religious family do you most closely identify?\n"
                f"A: {religion_func(row)}.\n"
                f"Q: What is your race or ethnicity?\n"
                f"A: I am {row['race_ethnicity']}.\n"
                f"Q: What is your marital_status?\n"
                f"A: I am {row['marital_status'].lower()}.")

    def backstory3(row):
        return (f"P1: How old are you?\nP2: I am {row['age']} years old.\n"
                f"P1: What is your gender?\nP2: I am {row['gender'].lower()}.\n"
                f"P1: Do you think of yourself as a Republican, Democrat, or Independent?\nP2: I consider myself a {row['party']}.\n"
                f"P1: What is the highest level of school you have completed or the highest degree you have received?\nP2: I have {education_func(row)}.\n"
                f"P1: How would you describe yourself politically?\nP2: I am {row['ideology'].lower()}.\n"
                f"P1: What was your total household income last year?\nP2: My total income is {row['income']}.\n"
                f"P1: With what religious family do you most closely identify?\nP2: {religion_func(row)}.\n"
                f"P1: What is your race or ethnicity?\nP2: I am {row['race_ethnicity']}.\n"
                f"P1: What is your marital_status?\nP2: I am {row['marital_status'].lower()}.\n")

    def backstory4(row):
        return (f"Age: {row['age']}, Gender: {row['gender']}, Political Affiliation: {row['party']}Education: {row['education']}, Ideology: {row['ideology']}, Total Income: {row['income']}, Religion: {row['religion']},Race/Ethnicity: {row['race_ethnicity']}, Marital Status: {row['marital_status']}")

    def backstory5(row):
        return (f"Question 1: How old are you?\n"
                f"Answer 1: I am {row['age']} years old.\n"
                f"Question 2: What is your gender?\n"
                f"Answer 2: I am {row['gender'].lower()}.\n"
                f"Question 3: Do you think of yourself as a Republican, Democrat, or Independent?\n"
                f"Answer 3: I consider myself a {row['party']}.\n"
                f"Question 4: What is the highest level of school you have completed or the highest degree you have received?\n"
                f"Answer 4: I have {education_func(row)}.\n"
                f"Question 5: How would you describe yourself politically?\n"
                f"Answer 5: I am {row['ideology'].lower()}.\n"
                f"Question 6: What was your total household income last year?\n"
                f"Answer 6: My total income is {row['income']}.\n"
                f"Question 7: With what religious family do you most closely identify?\n"
                f"Answer 7: {religion_func(row)}.\n"
                f"Question 8: What is your race or ethnicity?\n"
                f"Answer 8: I am {row['race_ethnicity']}.\n"
                f"Question 9 : What is your marital_status?\n"
                f"Answer 9: I am {row['marital_status'].lower()}.")

    def get_templates(self):
        return{
            #this one takes has answers Strongly Disagree, Disagree, Agree, Strongly Agree. I wasn't sure how to map that or what I should do about that or if I should change something up in the modify data function.
            "tech_employment": {
                "implicit_instructions": (lambda row: (f"{backstory1(row)}\n\n"
                    f"When asked whether technology gives me better employment opportunities, I would say"), {}),
                "QA": (lambda row: (f"{backstory2(row)}\n\n"
                    f"Q:Do you think that technology gives you new and better employment opportunities?\n\n"
                    f"A:"), {}),
                "conversation": (lambda row: (f"{backstory3(row)}\n\n"
                    f"P1: Do you think that technology gives you new and better employment opportunities?\n"
                    f"P2:"), {}),
                "verbatim_mc": (lambda row: (f"{backstory1(row)}\n\n"
                    f"To what extent do you agree with the following? Technology gives me new and better employment opportunities.\n\n"
                    f"1: Strongly disagree\n"
                    f"2: Disagree\n"
                    f"3: Agree\n"
                    f"4: Strongly agree\n"
                    f"Answer:"), {}),
                "verbatim_mc_w_letters": (lambda row: (f"{backstory1(row)}\n\n"
                    f"To what extent do you agree with the following? Technology gives me new and better employment opportunities.\n\n"
                    f"A: Strongly disagree\n"
                    f"B: Disagree\n"
                    f"C: Agree\n"
                    f"D: Strongly agree\n"
                    f"Answer:"), {}),
                "paraphrased_mc": (lambda row: (f"{backstory1(row)}\n\n"
                    f"Which statement do you agree with most?\n\n"
                    f"1: Technology does not give me new and better employment opportunities.\n"
                    f"2: Technology gives me new and better employment opportunities."
                    f"Answer:"), {}),
                "true_or_false": (lambda row: (f"{backstory1(row)}\n\n"
                    f"True or false? Technology gives me new and better employment opportunities.\n\n"
                    f"Answer:"), {}),
                "yes_no_question": (lambda row: (f"{backstory1(row)}\n\n"
                    f"Question: Do you believe that technology gives you new and better employment opportunities?\n"
                    f"Answer:"), {}),
                "task_based": (lambda row: (f"TASK: Consider the below demographic "
                    "information and answer the following question.\n\n"
                    f"CONTEXT: {backstory4(row)}\n\n"
                    f"QUESTION: Does technology give you new and better employment opportunities?\n"
                    f"Answer:"), {}),
                "finish_sentence": (lambda row: (f"{backstory1(row)}\n\n"
                    f"When asked whether technology gives me new and better employment opportunities, I would say that technology"), {}),
                "fill_in_blank" : (lambda row: (f"{backstory1(row)}\n\n"
                    f"I think that technology _ give me new and better employment opportunities.\n"
                    f"The word in the blank is"), {}),
                "fill_in_blank_mc": (lambda row: (f"{backstory1(row)}\n\n"
                    f"Fill in the blank:\n"
                    f"I think that technology _ give me new and better employment opportunities.\n"
                    f"A. does\n"
                    f"B. does\'nt\n\n"
                    f"The answer is"), {}),
                "rephrased_implicit_instructions" : (lambda row: (f"{backstory1(row)}\n\n"
                    f"When asked whether technology makes my employment opportunities better or worse, I would say that technology makes my employment opportunities"), {}),
                "0_shot_direct_mapping": (lambda row: (f"{backstory1(row)}\n\n"
                    f"Question 1: To what extent do you agree with the following? Technology gives me new and better employment opportunities.\n"
                    f"Answer 1: (Strongly disagree, Disagree, Agree, Strongly agree):"), {}),
                "1_shot_direct_mapping": (lambda row: (f"{backstory1_shot()}\n\n"
                    f"Question 1: Does technology have a significant impact upon your life?\n"
                    f"Answer 1: Yes\n"
                    f"{backstory1(row)}\n\n"
                    f"Question 2: To what extent do you agree with the following? Technology gives me new and better employment opportunities.\n"
                    f"Answer 2: (Strongly disagree, Disagree, Agree, Strongly agree):"), {}),
                "chapter_quiz": (lambda row:(f"CHAPTERQUIZ\n\nBACKSTORY:\n{backstory1(row)}\n\n"
                    f"STATEMENT: Technology gives me new and better employment opportunities.\n"
                    f"QUESTION:\n"
                    f"According to the above backstory, to what extent would this person agree with the statement?\n"
                    f"A) Strongly disagree\n"
                    f"B) Disagree\n"
                    f"C) Agree\n"
                    f"D) Strongly agree\n"
                    f"ANSWER:"), {}),
                "non_enumerated_response": (lambda row: (f"{backstory1(row)}When asked how"
                 f"technology impacts my employment opportunities, I would say"), {}),
                "survey_example": (lambda row: (f"{backstory5(row)}\n\n"
                     f"Question 10: Do you think that technology gives you new and better employment opportunities?\n"
                     f"Answer 10:"), {}),
                "survey_example_with_answers": (lambda row: (f"{backstory5(row)}\n\n"
                     f"Question 10: Do you think that technology gives you new and better employment opportunities? (Answers: Strongly disagree, Disagree, Agree, Strongly agree)\n"
                     f"Answer 10:"), {}),
                "survey_example_with_mc": (lambda row: (f"{backstory5(row)}\n\n"
                     f"Question 10: Do you think that technology gives you new and better"
                     f"employment opportunities?\n"
                     f"(A: Strongly disagree\n"
                     f"B: Disagree\n"
                     f"C: Agree\n"
                     f"D: Strongly agree\n"
                     f"Answer 10:"), {}),


            }
        }



if __name__ == "__main__":
    factory = BaylorFactory(BaylorSurvey())
