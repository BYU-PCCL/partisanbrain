from ..survey import Survey, UserInterventionNeededError
from ..constants import DEMOGRAPHIC_COLNAMES

import pandas
import requests
import os

SURVEY_URL = "https://osf.io/69vx7/download"

class BaylorSurvey(Survey):

    def __init__(self):
        super().__init__()

    def download_data(self):
        if not os.path.exists("survey_data/baylor_survey/raw.csv"):
            # download the data
            fsav = requests.get(SURVEY_URL)
            # save
            with open("survey_data/baylor_survey/raw.sav", "wb") as f:
                f.write(fsav.content)
            f.close()
            #convert sav file to csv
            df = pandas.read_spss("survey_data/baylor_survey/raw.sav")
            df.to_csv("survey_data/baylor_survey/raw.csv", index=False)


        return pandas.read_csv("survey_data/baylor_survey/raw.csv")


    def modify_data(self, df):
        #rename demo columns
        mod_df = df.rename(columns={"AGE" : "age",
                                    "Q77" : "gender",
                                    "Q32" : "party",
                                    "I_EDUC" : "education",
                                    "Q31" : "ideology",
                                    "Q95" : "income",
                                    "Q1" : "religion",
                                    "RACE" : "race_ethnicity",
                                    "D9" : "marital_status"})

        # Drop na rows
        mod_df = mod_df.dropna(subset=["age", "gender", "party", "education", "ideology", "income", "religion", "race_ethnicity", "marital_status"])

        # Rename DV columns
        mod_df = mod_df.rename(columns={"T7G" : "tech_employment",
                                        "MP4A" : "legal_gay_marriage",
                                        "Q61D" : "gay_choice",
                                        "MP4G" : "husband_salary",
                                        "MP4F" : "women_childcare",
                                        "MP4D" : "men_politics",
                                        "MP4J" : "refugee_terrorist",
                                        "MP4K" : "mexican_criminals",
                                        "Q45" : "life_happy",
                                        "H13E" : "week_depression",
                                        "H12" : "days_exercise",
                                        "MP4H" : "police_race",
                                        "MP4I" : "black_violent",
                                        "Q17" : "bible_belief",
                                        "Q18" : "god_belief",
                                        "Q19A" : "god_concern_world",
                                        "Q19D" : "god_concern_personal",
                                        "Q4" : "religious_attendance",
                                        "MP12F" : "school_prayer",
                                        "R20F" : "god_plan"})

        # Drop extra columns
        column_lst = ["age", "gender", "party", "education", "ideology", "income", "religion", "race_ethnicity", "marital_status"] + ["tech_employment", "legal_gay_marriage", "gay_choice", "husband_salary", "women_childcare", "men_politics", "refugee_terrorist", "mexican_criminals", "life_happy", "week_depression", "days_exercise", "police_race", "black_violent", "bible_belief", "god_belief", "god_concern_world", "god_concern_personal", "religious_attendance", "school_prayer", "god_plan"]
        mod_df = mod_df[column_lst]

        return mod_df

    def get_dv_questions(self):
        dv_questions = {
            "tech_employment": "To what extent do you agree with the following? Technology gives me new and better employment opportunities.",
            "legal_gay_marriage": "Please rate the extent to which you agree or disagree with the following statements: Gays and lesbians should be allowed to legally marry.",
            "gay_choice": "Please rate the extent to which you agree or disagree with the following statements: People choose to be gay/lesbian.",
            "husband_salary": "Please rate the extent to which you agree or disagree with the following statements: A husband should earn a larger salary than his wife.",
            "women_childcare": "Please rate the extent to which you agree or disagree with the following statements: It is God's will that women care for children.",
            "men_politics": "Please rate the extent to which you agree or disagree with the following statements: Men are better suited emotionally for politics than women.",
            "refugee_terrorist": "Please rate the extent to which you agree or disagree with the following statements: Refugees from the Middle East pose a terrorist threat to the United States.",
            "mexican_criminals": "Please rate the extent to which you agree or disagree with the following statements: Illegal immigrants from Mexico are mostly dangerous criminals.",
            "life_happy": "In general, how happy are you with your life as a whole these days?",
            "week_depression": "In the past WEEK, about how often have you had the following feelings? I felt depressed.",
            "days_exercise": "How many DAYS per WEEK do you do exercise for at least 30 minutes?",
            "police_race": "Please rate the extent to which you agree or disagree with the following statements: Police officers in the United States treat blacks the same as whites.",
            "black_violent": "Please rate the extent to which you agree or disagree with the following statements: Police officers in the United States shoot blacks more often because they are more violent than whites.",
            "bible_belief": "Which one statement comes closest to your personal beliefs about the Bible?",
            "god_belief": "Which one statement comes closest to your personal beliefs about God?",
            "god_concern_world": "Based on your personal understanding of God, please rate the extent to which you agree or disagree with the following statements: God is concerned with the well-being of the world.",
            "god_concern_personal": "Based on your personal understanding of God, please rate the extent to which you agree or disagree with the following statements: God is concerned with my personal well-being.",
            "religious_attendance": "How often do you attend religious services at a place of worship?",
            "school_prayer": "Please rate the extent to which you agree or disagree with the following statements: The federal government should allow prayer in public schools.",
            "god_plan": "Please rate the extent to which you agree or disagree with the following statements: When good or bad things happen to me, I see it as part of God's plan for me."
        }

        return dv_questions


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = BaylorSurvey()
