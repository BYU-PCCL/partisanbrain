from ..survey import Survey
import os
import pandas as pd
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


class GssSurvey(Survey):
    def __init__(self):
        super().__init__()

    def download_data(self):
        url = "https://gss.norc.org/Documents/spss/2018_spss.zip"
        directory = "survey_data/gss_survey/"

        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(directory)
        os.rename(
            "survey_data/gss_survey/GSS2018.sav", "survey_data/gss_survey/gss.sav"
        )
        df = pd.read_spss("survey_data/gss_survey/gss.sav")
        return df

    def modify_data(self, df):
        # Rename demographic columns
        mod_df = df.rename(
            columns={
                "AGE": "age",
                "SEX": "gender",
                "PARTYID": "party",
                "EDUC": "ideology",
                "POLVIEWS": "education",
                "INCOME": "income",
                "RELIG": "religion",
                "RACECEN1": "race_ethnicity",
                "REGION": "region",
                "MARITAL": "marital_status",
            }
        )
        # Drop NA
        mod_df = mod_df.dropna(
            subset=[
                "age",
                "gender",
                "party",
                "ideology",
                "education",
                "income",
                "religion",
                "race_ethnicity",
                "region",
                "marital_status",
            ]
        )
        # Rename DV columns.
        mod_df = mod_df.rename(
            columns={
                "NATENVIR": "spending_protecting_environment",
                "NATENRGY": "spending_developing_alternate_energy",
                "CAPPUN": "capital_punishment",
                "POLHITOK": "police_striking_adult_male",
                "GRASS": "marijuana_legality",
                "WKSEXISM": "gender_discrimination",
                "FEPOL": "men_better_suited_politics",
                "HLTHMNTL": "mental_health",
                "MENTLOTH": "who_to_visit_for_mental_health",
                "FAMMHNEG": "family_opinions_mental_health",
                "DIAGNOSD": "diagnosed_mental_health",
                "OTHMHNEG": "outside_family_opinions_mental_health",
                "HLTHPHYS": "physical_health",
                "VOTE16": "voted_in_2016",
                "SPKRAC": "free_speech_for_racists",
                "BIBLE": "bible_feelings",
                "PRAYER": "prayer_in_public_schools",
                "CONCLERG": "organized_religion_leader_opinions",
                "RELITEN": "religious_conviction",
                "POSTLIFE": "life_after_death",
            }
        )
        # Drop all extra columns
        mod_df = mod_df[
            [
                "age",
                "gender",
                "party",
                "ideology",
                "education",
                "income",
                "religion",
                "race_ethnicity",
                "region",
                "marital_status",
                "spending_protecting_environment",
                "spending_developing_alternate_energy",
                "capital_punishment",
                "police_striking_adult_male",
                "marijuana_legality",
                "gender_discrimination",
                "men_better_suited_politics",
                "mental_health",
                "who_to_visit_for_mental_health",
                "family_opinions_mental_health",
                "diagnosed_mental_health",
                "outside_family_opinions_mental_health",
                "physical_health",
                "voted_in_2016",
                "free_speech_for_racists",
                "bible_feelings",
                "prayer_in_public_schools",
                "organized_religion_leader_opinions",
                "religious_conviction",
                "life_after_death",
            ]
        ]
        return mod_df

    def get_dv_questions(self):
        dv_questions = {
            "spending_protecting_environment": "Are we spending too much, too little, or about the right amount on improving and protecting the environment?",
            "spending_developing_alternate_energy": "Are we spending too much, too little, or about the right amount on developing alternative energy sources",
            "capital_punishment": "Do you favor or oppose the death penalty for persons convicted of murder?",
            "police_striking_adult_male": "Are there any situations you can imagine in which you would approve of a policeman striking an adult male citizen?",
            "marijuana_legality": "Do you think the use of marijuana should be made legal or not?",
            "gender_discrimination": "Do you feel in any way discriminated against on your job because of your gender?",
            "men_better_suited_politics": "Tell me if you agree or disagree with this statement: Most men are better suited emotionally for politics than are most women",
            "mental_health": "In general, how would you rate your mental health, including your mood and your ability to think?",
            "who_to_visit_for_mental_health": "John is a white man with a college education. For the past two weeks John has been feeling really down. He wakes up in the morning with a flat heavy feeling that sticks with him all day long. He isn't enjoying things the way he normally would. In fact nothing gives him pleasure. Even when good things happen, they don't seem to make John happy. He pushes on through his days, but it is really hard. The smallest tasks are difficult to accomplish. He finds it hard to concentrate on anything. He feels out of energy and out of steam. And even though John feels tired, when night comes he can't go to sleep. John feels pretty worthless, and very discouraged. John's family has noticed that he hasn't been himself for about the last month and that he has pulled away from them. John just doesn't feel like talking. Should John go to a therapist, or counselor, like a psychologist, social worker, or other mental health professional for help?",
            "family_opinions_mental_health": "Thinking about your family, to what extent do they hold negative attitudes about people with mental health problems?",
            "diagnosed_mental_health": "Have you ever been diagnosed with a mental health problem?",
            "outside_family_opinions_mental_health": "Thinking about other people you know personally outside of your family, to what extent do they hold negative attitudes about people with mental health problems?",
            "physical_health": "In general, how would you rate your physical health?",
            "voted_in_2016": "In 2016, you remember that Clinton ran for President on the Democratic ticket against Trump for the Republicans. Do you remember for sure whether or not you voted in that election?",
            "free_speech_for_racists": "Consider a person who believes that Blacks are genetically inferior. If such a person wanted to make a speech in your community claiming that Blacks are inferior, should he be allowed to speak, or not?",
            "bible_feelings": "Which of these statements comes closest to describing your feelings about the Bible? The Bible is the actual word of God and is to be taken literally, word for word. The Bible is the inspired word of God but not everything in it should be taken literally, word for word. The Bible is an ancient book of fables, legends, history, and moral precepts recorded by man.",
            "prayer_in_public_schools": "The United States Supreme Court has ruled that no state or local government may require the reading of the Lord's Prayer or Bible verses in public schools. What are your views on this--do you approve or disapprove of the court ruling?",
            "organized_religion_leader_opinions": "As far as the people running organized religion are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?",
            "religious_conviction": "Would you call yourself a strong (PREFERENCE NAMED IN RELIG) or a not very strong (PREFERENCE NAMED IN RELIG)?",
            "life_after_death": "Do you believe there is a life after death?",
        }

        return dv_questions


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = GssSurvey()
