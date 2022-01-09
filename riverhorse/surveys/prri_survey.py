''''Code by: MK'''

from parent_dir import Survey, UserInterventionNeededError
import os
import pandas as pd
import requests


class PrriSurvey(Survey):

    def __init__(self):
        super().__init__()

    def download_data(self):

        if not os.path.exists("survey_data/prri_survey/raw.csv"):
            # download the data
            fsav = requests.get("https://osf.io/x6gnf/download")
            # save
            with open("survey_data/prri_survey/raw.sav", "wb") as f:
                f.write(fsav.content)
            f.close()
            #convert sav file to csv
            df = pd.read_spss("survey_data/prri_survey/raw.sav")
            df.to_csv("survey_data/prri_survey/raw.csv", index=False)


        return pd.read_csv("survey_data/prri_survey/raw.csv")

    # Helper function that returns combined columns in a list
    def combine(self, col1, col2, df):

        partyDf = df[[col1,col2]]
        partyDf['combined'] = partyDf.values.tolist()
        return partyDf['combined']

    def modify_data(self, df):
        
        demo = {   
            "PARTY": "party", # includes party and party leaning
            "RELIG": "religion",
            "IDEO": "ideology",
            "GENDER": "gender",
            "AGE": "age",
            "RACETHNICITY": "race_ethnicity",
            "EDUC": "education", # includes education and employment status
            "I_MARITAL": "marital_status",
            "INCOME": "income", # includes income  and class
            "REGION4": "region", # includes region and state
        }

        #Renaming Demographic Columns
        mod_df = df.rename(columns=demo)

        # Adding extra information to the columns
        mod_df['party'] = self.combine('party','PARTYLN', mod_df)
        mod_df['income'] = self.combine('income','CLASS', mod_df)
        mod_df['region'] = self.combine('region','STATE', mod_df)
        mod_df['education'] = self.combine('education','EMPLOY2', mod_df)
        
        #Cleaning
        demo_cols = list(demo.values())
        mod_df = mod_df.dropna(subset=demo_cols)

        # Drop all rows from df that have a null
        # value for the demographic columns that are
        # present *in this data* (not all 10 will
        # necessarily be present). You may have to combine
        # multiple demographic columns to get the
        # demographic information you need.

        # Rename DV columns.
        dvs={   
            "Q1": "voting_frequency",
            "Q5": "donald_trump_as_president",
            "Q18C": "vladimere_putin_favority",
            "Q20A": "women_in_political_office",
            "Q20B": "lgbtq_in_political_office",
            "Q20C": "electing_racial_ethnic_minority",
            "Q20D": "electing_non_christians",
            "Q22": "killing_african_americans",
            "Q26E": "discrimination_against_asains",
            "Q26F": "discrimination_against_hispanic",
            "Q27N": "minorities_use_racism_as_excuse",
            "Q27O": "discriminiation_against_white_problem",
            "Q27R": "country_changed",
            "Q27P": "prefrence_to_immigrant_in_western_europe",
            "Q29": "lgbtq_equal_rights",
            "Q30": "view_on_immigration",
            "Q31": "perspective_on_immigration",
            "Q33": "impact_of_mixed_racial",
            "Q35C": "refugees_entering_us",
            "Q36": "views_illegal_immigrants_us"
            }

        mod_df = mod_df.rename(columns= dvs)

        #Removing all other unecessary columns that we wont be using
        cols = list(dvs.values()) + list(demo.values())
        mod_df = mod_df[cols]

        # More processing here to get the data super nice and clean
        # like changing responses to exactly match what is in the
        # codebook
        

        return mod_df

    def get_dv_questions(self):
        return {
            "voting_frequency": "How often would you say you vote?",
            "donald_trump_as_president": "Do you strongly approve, somewhat approve, somewhat disapprove or strongly disapprove of the job Donald Trump is doing as president?",
            "vladimere_putin_favority": "Please say whether your overall opinion of each of the following is very favorable, mostly favorable, mostly unfavorable, or very unfavorable. First, Russian President Vladimir Putin",
            "women_in_political_office": "Do you think electing more women to political office would make things in the U.S. better, make them worse or would not make much difference?",
            "lgbtq_in_political_office": "Do you think electing more lesbian, gay, bisexual and transgender people to political office would make things in the U.S. better, make them worse or would not make much difference?",
            "electing_racial_ethnic_minority": "Do you think electing more people from racial and ethnic minority groups to political office would make things in the U.S. better, make them worse or would not make much difference?",
            "electing_non_christians": "Do you think electing more people from non-Christian religious groups to political office would make things in the U.S. better, make them worse or would not make much difference?",
            "killing_african_americans": "Do you think recent killings of African American men by police are isolated incidents or are they part of a broader pattern of how police treat African Americans?",
            "discrimination_against_asains": "Just your impression, in the United States today, is there a lot of discrimination against asians or not?",
            "discrimination_against_hispanic": "Just your impression, in the United States today, is there a lot of discrimination against hispanics or not?",
            "minorities_use_racism_as_excuse": "Now, read each statement and please say if you completely agree, mostly agree, mostly disagree or completely disagree with each one. Racial minorities use racism as an excuse more than they should",
            "discriminiation_against_white_problem": "Now, read each statement and please say if you completely agree, mostly agree, mostly disagree or completely disagree with each one. Today discrimination against whites has become as big a problem as discrimination against blacks and other minorities",
            "country_changed": "Now, read each statement and please say if you completely agree, mostly agree, mostly disagree or completely disagree with each one. Things have changed so much that I often feel like a stranger in my own country",
            "prefrence_to_immigrant_in_western_europe": "Now, read each statement and please say if you completely agree, mostly agree, mostly disagree or completely disagree with each one. Overall, we should give preference to immigrants from Western Europe, who share our values",
            "lgbtq_equal_rights": "Which of the following statements comes closest to your own view?",
            "view_on_immigration": "Would you say that, in general, the growing number of newcomers from other countries...",
            "perspective_on_immigration": "Which of the following statements comes closer to your own views...",
            "impact_of_mixed_racial": "As you may know, U.S. Census projections show that by 2045 African Americans, Latinos, Asians, and other mixed racial and ethnic groups will together be a majority of the population. do you think the likely impack of this coming demographic change will be mostly positive or mostly negative?",
            "refugees_entering_us": "We would like to get your views on some issues that are being discussed in the country today. Do you strongly favor, favor, oppose or strongly oppose the following? Passing a law to prevent refugees from entering the U.S.",
            "views_illegal_immigrants_us": "Which statement comes closest to your view about how the immigration system should deal with immigrants who are currently living in the U.S. illegally? The immigration system should.."
            }


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = PrriSurvey()
