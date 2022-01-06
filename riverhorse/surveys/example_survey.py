from parent_dir import DEMOGRAPHIC_COLNAMES, Survey
from parent_dir import UserInterventionNeededError

import os
import pandas as pd


class ExampleSurvey(Survey):

    def __init__(self):
        # Do any other initialization (if needed) here
        # before the super() call.
        super().__init__()

    def download_data(self):
        # Below is an example of how you'd handle requiring
        # manual intervention to download data. Of course,
        # if the data can be downloaded automatically,
        # that is ideal.
        if os.path.exists("survey_data/example/raw.csv"):
            df = pd.read_csv("survey_data/example/raw.csv")
        else:
            msg = "You need to login to get this data from www.example.com"
            raise UserInterventionNeededError(msg)

        # Here you could have more processing that is
        # after the manual intervention (if needed).

        return df

    def modify_data(self, df):
        # Rename demographic columns.
        mod_df = df.rename(columns={"V10344": "age",
                                    "V10346": "gender",
                                    "T23kfk": "party",
                                    "Esdfk3": "ideology",
                                    "Y23kfr": "education",
                                    "C23k33": "income",
                                    "B230vv": "religion",
                                    "P67222": "race_ethnicity",
                                    "X10002": "region",
                                    "M1M233": "marital_status"})

        # Drop all rows from df that have a null
        # value for the demographic columns that are
        # present *in this data* (not all 10 will
        # necessarily be present). You may have to combine
        # multiple demographic columns to get the
        # demographic information you need.
        mod_df = mod_df.dropna(subset=DEMOGRAPHIC_COLNAMES)

        # Rename DV columns.
        mod_df = mod_df.rename(columns={"H32k44": "papusas_best_food",
                                        "XRR323": "dark_light_whiteboard",
                                        "Nk3333": "naacl_tackle"})

        # More processing here to get the data super nice and clean
        # like changing responses to exactly match what is in the
        # codebook
        return mod_df

    def get_dv_questions(self):
        dv_questions = {
            "papusas_best_food": "Are papusas the best food?",
            "dark_light_whiteboard": "Dark or light whiteboards?",
            "naacl_tackle": "Does NAACL rhyme with tackle?"
        }

        return dv_questions


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub for your subclass - this will
    # have errors because there's no real data behind it.
    s = ExampleSurvey()
