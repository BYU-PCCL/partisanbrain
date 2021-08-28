import sys
sys.path.append('src')
import dataset
from pdb import set_trace as bp
import pandas as pd

class AnesDataset(dataset.Dataset):
    def __init__(self, n_exemplars):
        survey_fname = "data/anes/anes_timeseries_2020_csv_20210719.csv"
        #Read this into a pandas df
        df = pd.read_csv(survey_fname)
        bp()
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):

        rename_dic = {
                "V201507x": 'age',
                "V202637" : "gender" ,
                "V201018" : "party",
                "V201510" : 'education',
                "V201200" : "ideo",
                "V201607" :  "income",
                "V201458x": "religion",
                "V201549x": 'race',
                "V203003" : "region" ,
                "V201508" : "marital",
                "V201321" : "protect_environment", #Protect environment
                "V201401" : "government_temperatures", #Government temps
                "V201309" : "federal_spending_crime", #Federal spending crime
                "V201130" : "trump_economy", #Trump economy
                "V201235" : "government_waste", #Government waste
                "V201300" : "social_security", #social security
                "V201318" : "spending_poor", #spending poor
                "V201324" : "economy_good", #Economy good
                "V201594" : "economy_worse_better", #Economy worse or better
                "V201312" : "welfare", #Welfare programs
                "V201416" : "gay_marriage", #Gay marriage
                "V201133" : "trump_foreign_relations", #Trump foreign relations
                "V201139" : "trump_immigration", #Trump handling immigration
                "V201350" : "military", #military international problems
                "V201619" : "sleep", #Restless sleep
                "V201620" : "health_insurance", #Health insurance
                "V201006" : "political_campaigns", #Political campaigns
                "V201223" : "voting_duty", #Voting duty
                "V201234" : "government_elite", #government elite?
        }

        # Dropping all but relevant columns
        new_df = df[list(rename_dic.keys())]

        # Dropping rows with NA values
        new_df = new_df.dropna(axis=0)

        # Renaming columns for convenience
        new_df = new_df.rename(rename_dic, axis=1)

        return new_df

    def _make_backstory(self, row, format):

        # Renaming columns for convenience
        code_dic = {

                "age": {
                    'q': 'What is your age?',
                    'a': {k:str(k) for k in range(80)}.update({'80': "80 or older"}),
                    'abs': {k:f"I am {k} years old." for k in range(80)}.update({'80': "I am 80 years old."}),
                },
                "gender": {
                    'q': "What's your gender (male or female)?",
                    'a': {
                        1: "Male",
                        2: "Female",
                    },
                    'abs': {
                        1: "I am male.",
                        2: "I am female.",
                    },
                },
                "party": {
                    'q':"What political party are you registered with, if any (Republican party, Democratic party, independent)?",
                    'a': {
                        -9: "Refused",
                        -8: "Don’t know",
                        -1: "Inapplicable",
                        1: "Democratic party",
                        2: "Republican party",
                        4: "None or ‘independent’",
                        5: "Other {SPECIFY}",
                    },
                    'abs': {
                        1: "I'm a Democrat.",
                        2: "I'm a Republican.",
                        4: "I'm an independent",
                    },
                },
                "education": {
                    'q': "What is the highest level of school you have completed or the highest degree you have received ('Less than high school credential', 'High school graduate - High school diploma or equivalent (e.g: GED)', 'Some college but no degree', 'Associate degree in college - occupational/vocational', 'Associate degree in college - academic', 'Bachelor’s degree (e.g. BA, AB, BS)', 'Master’s degree (e.g. MA, MS, MEng, MEd, MSW, MBA)', 'Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)')?",
                    'a': {
                        -9: "Refused",
                        -8: "Don’t know",
                        1: "Less than high school credential",
                        2: "High school graduate - High school diploma or equivalent (e.g: GED)",
                        3: "Some college but no degree",
                        4: "Associate degree in college - occupational/vocational",
                        5: "Associate degree in college - academic",
                        6: "Bachelor’s degree (e.g. BA, AB, BS)",
                        7: "Master’s degree (e.g. MA, MS, MEng, MEd, MSW, MBA)",
                        8: "Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)",
                        95: "Other \{SPECIFY\}}",
                        },
                    'abs': {
                        1: "I didn't graduate from high school.",
                        2: "I graduated from high school.",
                        3: "I did some college but didn't get a degree.",
                        4: "I have an associate's degree",
                        5: "I have an associate's degree",
                        6: "I have a bachelor’s degree.",
                        7: "I have a master’s degree.",
                        8: "I went to grad school.",
                        },
                },
                "ideo": {
                    'q': "Where would you place yourself on this scale, or haven’t you thought much about this ('Extremely liberal', 'Liberal', 'Slightly liberal', 'Moderate; middle of the road', 'Slightly conservative', 'Conservative', 'Extremely conservative', 'Haven’t thought much about this')?",
                    'a': {
                        1: "Extremely liberal",
                        2: "Liberal",
                        3: "Slightly liberal",
                        4: "Moderate; middle of the road",
                        5: "Slightly conservative",
                        6: "Conservative",
                        7: "Extremely conservative",
                        99: "Haven’t thought much about this",
                    },
                    'abs': {
                        1: "i am extremely liberal.",
                        2: "i am liberal.",
                        3: "i am slightly liberal.",
                        4: "i am moderate; middle of the road.",
                        5: "i am slightly conservative.",
                        6: "i am conservative.",
                        7: "i am extremely conservative.",
                        99: "I don't think much about politics.",
                    },
                },
                #This is restricted so it doesn't matter
                "income": {
                    'q': "The next question is about [the total combined income of all "
                        "members of your family / your total income] during the past 12 "
                        "months. This includes money from jobs, net income from "
                        "business, farm or rent, pensions, dividends, interest, Social "
                        "Security payments, and any other money income received by "
                        "members of your family who are 15 years of age or older. What "
                        "was the total income of your family during the past 12 months? "
                        "TYPE THE NUMBER. YOUR BEST GUESS IS FINE.",
                    "abs": "I"
                },
                "religion": {
                    'q': "What's your religion ('Mainline Protestant', 'Evangelical Protestant', 'Black Protestant', 'Undifferentiated Protestant', 'Roman Catholic', 'Other Christian', 'Jewish', 'Other religion', 'Not religious')?",
                    'a': {
                        1: "Mainline Protestant",
                        2: "Evangelical Protestant",
                        3: "Black Protestant",
                        4: "Undifferentiated Protestant",
                        5: "Roman Catholic",
                        6: "Other Christian",
                        7: "Jewish",
                        8: "Other religion",
                        9: "Not religious",
                    },
                    'abs': {
                        1: "I am Mainline Protestant.",
                        2: "I am Evangelical Protestant.",
                        3: "I am Black Protestant.",
                        4: "I am Undifferentiated Protestant.",
                        5: "I am Roman Catholic.",
                        6: "I am Christian.",
                        7: "I am Jewish.",
                        8: "I belong to a non-standard religion.",
                        9: "I am not religious.",
                    },
                },
                "race": {
                    'q': "What's your race/ethnicity ('White, non-Hispanic', 'Black, non-Hispanic', 'Hispanic', 'Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone', 'Native American/Alaska Native or other race, non-Hispanic alone', 'Multiple races, non-Hispanic')?",
                    'a': {
                        1: "White, non-Hispanic",
                        2: "Black, non-Hispanic",
                        3: "Hispanic",
                        4: "Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone",
                        5: "Native American/Alaska Native or other race, non-Hispanic alone",
                        6: "Multiple races, non-Hispanic",
                    },
                    'abs': {
                        1: "I am White.",
                        2: "I am Black.",
                        3: "I am Hispanic.",
                        4: "I am Asian or Native Hawaiian/other Pacific Islander.",
                        5: "I am Native American/Alaska Native.",
                        6: "I am Multiple races.",
                    },
                },
                "region": {
                    'q': "Which census region do you live in (Northeast, Midwest, South, or West)?",
                    'a': {
                        1: "Northeast",
                        2: "Midwest",
                        3: "South",
                        4: "West",
                    },
                    'abs': {
                        1: "I am from the Northeast.",
                        2: "I am from the Midwest.",
                        3: "I am from the South.",
                        4: "I am from the West.",
                    },
                },
                "marital": {
                    'q': "Are you now married, widowed, divorced, separated or never married?",
                    'a': {
                        -9: "Refused",
                        -8: "Don’t know",
                        1: "Married: spouse present",
                        2: "Married: spouse absent {VOL - video/phone only}",
                        3: "Widowed",
                        4: "Divorced",
                        5: "Separated",
                        6: "Never married",
                    },
                    'abs': {
                        1: "I am married.",
                        2: "I am married.",
                        3: "I am widowed.",
                        4: "I am divorced.",
                        5: "I am separated.",
                        6: "I have never married.",
                    },
                },
        }



        backstory = ""
        if "format" == "QA":
            #For every Demographic question, ask the question asked in the survey.
            for code, dic in code_dic.items():
                backstory+= f"Q: {dic['q']}\nA: {dic['a'][row[code]]}\n\n"
        elif "format" == "FPBS":

            for code, dic in code_dic.items():
                backstory+= f"{dic['abs'][row[code]]} "
        else:
            raise Exception("Invalid format")
        return backstory

    def _get_prompt_instructions(self):
        return {
                "protect_environment": (( "If you ask me whether federal spending on protecting the environment should be (increased, decreased, or kept the same), I would answer that it should be"),
                                lambda x: {
					1: " increased",
					2: " decreased",
					3: " kept the same",
                    }[x]
                ),
                "government_temperatures": (("If you were to ask me whether the federal government should be doing (more, less, or same) about rising temperatures than it is currently, I would answer that it should be doing"),
                lambda x: {
					1: " more",
					2: " less",
					3: " same",
                    }[x]
                ),
                "federal_spending_crime": (("If you ask me whether the federal government should spend (more, less, or same) on crime, I would answer that it should spend"),
                lambda x: {
					1: " more",
					2: " less",
					3: " same",
                    }[x]
                    ),
                "trump_economy": (("If asked whether I (approve or disapprove) of the way the way Donald Trump is handling the economy, I would answer that I"),
					                lambda x: {
					-8: " don’t know",
					1: " approve",
					2: " disapprove",
                    }[x]
					),
                "government_waste": (("If asked whether people in government waste (lots, some, little) of the money we pay in taxes, I would answer that I think they waste"),
                lambda x: {
					1: " lots",
					2: " some",
					3: " little",
                    }[x]
                ),
                "social_security": (("If asked whether the federal government should spend (more, less, same) on Social Security, I would answer that it should spend" ),
                lambda x: {
					1: " more",
					2: " less",
					3: " same",
                    }[x]
                ),
                "spending_poor": (("If asked whether the federal government should spend (more, less, or same) on aid to the poor, I would answer that it should spend"),
                lambda x: {
					1: " more",
					2: " less",
					3: " same",
                    }[x]
                    ),
                "economy_good": (("If asked whether the state of the economy these days is (excellent, good, neither, bad, or terrible), I would answer that it is"),
                lambda x: {
					1: " excellent",
					2: " good",
					3: " neither",
					4: " bad",
					5: " terrible",
                    }[x]
                ),
                "economy_worse_better": (("If asked whether the economy in the country as a whole over the past year has (improved, worsened, or stayed about the same), I would answer that it has"),
                lambda x: {
					1: " improved",
					2: " stayed about the same",
					3: " worsened",
                    }[x]
                ),
                "welfare": (("If asked whether the federal government should spend (more, less, or same) on welfare programs, I would answer that it should spend"),
                lambda x: {
					1: " more",
					2: " less",
					3: " same",
                    }[x]
                ),
                "gay_marriage": (("If asked whether gay and lesbian couples should be allowed to enter into (marriage, civil union, or neither), I would say that they should be allowed to enter into"),
                lambda x: {
					1: " marriage",
					2: " civil union",
					3: " neither",
                    }[x]
                ),
                "trump_foreign_relations": (("If asked whether I (approve, disapprove, or don't know) of the way Donald Trump is handling relations with foreign countries, I would answer that I"),
                lambda x: {
					-8: " don’t know",
					1: " approve",
					2: " disapprove",
                    }[x]
                ),
                "trump_immigration": (("If asked whether I (approve, disapprove, or don't know) of the way Donald Trump is handling immigration, I would answer that I"),
                lambda x: {
					-8: " don’t know",
					1: " approve",
					2: " disapprove",
                    }[x]
                    ),
                "military": (("If asked whether the United States should be (extremely, very, moderately, a little, or not at all) willing to use military force to solve international problems, I would answer that it should be"),
                lambda x: {
					1: " extremely",
					2: " very",
					3: " moderately",
					4: " a little",
					5: " not at all",
                    }[x]
                ),
                "sleep": (("If asked whether in the past week my sleep has been restless (all the time, often, sometimes, rarely, or never), I would answer that it has been restless"),
                lambda x: {
					1: " all the time",
					2: " often",
					3: " sometimes",
					4: " rarely",
					5: " never",
                    }[x]
                ),
                "health_insurance": (("If asked whether I (do or don't) presently have any kind of health insurance, I answer that I"),
                lambda x: {
					1: " do",
					2: " don't",
                    }[x]
                ),
                "political_campaigns": (("If asked whether I have been (very much, somewhat, or not much) interested in political campaigns so far this year, I answer that I have been"),
                lambda x: {
					1: " very much",
					2: " somewhat",
					3: " not much",
                    }[x]
                ),
                "voting_duty": (("If asked whether I feel (very, moderately, a little) strongly that voting is a duty, I answer that I feel"),
                lambda x: {
					1: " very",
					2: " moderately",
					3: " a little",
                    }[x]
                ),
                "government_elite": (("If asked whether government is run by the (people or big interests), I answer that government is run by the"),
                lambda x: {
					1: " big interests",
					2: " people",
                    }[x]
                ),
            }









if __name__ == '__main__':
    import random

    ds = AnesDataset(n_exemplars=5)