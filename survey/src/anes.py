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

        # Dropping all but relevant columns
        new_df = df[[
                     "V201507x", #Age
                     "V202637",
                     "V201018", #Party
                     "V201510",
                     "V201200",
                     "V201607",
                     "V201458x",
                     "V201549x",
                     "V203003",
                     "V201508",
                     "V201321", #Protect environment
                     "V201401",#Government temps
                     "V201309",#Federal spending crime
                     "V201130", #Trump economy
                     "V201235", #Government waste
                     "V201300", #social security
                     "V201318", #spending poor
                     "V201324", #Economy good
                     "V201594", #Economy worse or better
                     "V201312", #Welfare programs
                     "V201416", #Gay marriage
                     "V201133", #Trump foreign relations
                     "V201139", #Trump handling immigration
                     "V201350", #military international problems
                     "V201619", #Restless sleep
                     "V201620", #Health insurance
                     "V201006", #Political campaigns
                     "V201223", #Voting duty
                     "V201234", #government elite?
        ]]

        # Dropping rows with NA values
        new_df = new_df.dropna(axis=0)




        # Removing "I don't understand this question" response
        new_df = new_df.loc[new_df["shot_first"].isin(["Han", "Greedo"])]

        # Get only top 8 rows to keep things simple for testing
        new_df = new_df.head(8)

        return new_df

    def _make_backstory(self, row, format):
        """
        format needs to be in {'QA_exact', 'QA_colloq', 'FPBS_colloq'} 
        """


        # Renaming columns for convenience
        code_dic = {

                "V201507x": {
                    'name': 'age',
                    'q': 'What is your age?',
                    'a': {k:str(k) for k in range(80)}.update({'80': "80 or older"}),
                    'abs': {k:f"I am {k} years old." for k in range(80)}.update({'80': "I am 80 years old."}),
                },
                "V202637": {
                    'name': "gender" ,
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
                "V201018": {
                    'name': "party",
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
                "V201510": {
                    'name': 'education',
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
                "V201200": {
                    'name': "ideo",
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
                "V201607": {
                    'name': "income",
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
                "V201458x": {
                    'name': "religion",
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
                "V201549x": {
                    'name': 'race',
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
                "V203003": {
                    'name': "region" ,
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
                "V201508": {
                    'name': "marital",
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

        if self.format == "QA":
            return {
                
                "protect_environment": (("What about protecting the environment? Should federal "
                                            "spending on protecting the environment be increased, "
                                            "decreased, or kept the same?"),
                                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                ),
                "government_temperatures": (("Do you think the federal government should be doing more "
                        "about rising temperatures, should be doing less, or is it currently "
                        "doing the right amount?"),
                lambda x: {
					-9: "Refused"
					-8: "Don’t know"
					1: "Should be doing more"
					2: "Should be doing less"
					3: "Is currently doing the right amount",
                    }[x]
                ),
                "federal_spending_crime": (("What about dealing with crime? Should federal spending on "
                        "dealing with crime be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused"
					-8: "Don’t know"
					1: "Increased"
					2: "Decreased"
					3: "Kept the same"
                    }[x]
                    ),
                "trump_economy": (("Do you approve or disapprove of the way Donald Trump is handling "
                                        "the economy?"),
					                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Approve",
					2: "Disapprove",
                    }[x]
					),
                "government_waste": (("Do you think that people in government waste a lot of the money "
                        "we pay in taxes, waste some of it, or don’t waste very much of "
                        "it?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Waste a lot",
					2: "Waste some",
					3: "Don’t waste very much",
                    }[x]
                ),
                "social_security": (("What about Social Security? Should federal spending on Social "
                                        "Security be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                ),
                "spending_poor": (("What about aid to the poor? Should federal spending on aid to "
                                        "the poor be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                    ),
                "economy_good": (("What do you think about the state of the economy these days in "
                                    "the United States? Would you say the state of the economy is "
                                    "very good,good, neither good nor bad, bad, or very bad?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Very good",
					2: "Good",
					3: "Neither good nor bad",
					4: "Bad",
					5: "Very bad",
                    }[x]
                ),
                "economy_worse_better": (("Now thinking about the economy in the country as a whole, "
                                            "would you say that over the past year the nation’s economy has "
                                            "gotten better, stayed about the same, or gotten worse?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Gotten better",
					2: "Stayed about the same",
					3: "Gotten worse",
                    }[x]
                ),
                "welfare": (("What about welfare programs? Should federal spending on "
                            "welfare programs be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                ),
                "gay_marriage": (("Which comes closest to your view? You can just tell me the "
                                    "number of your choice."),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Gay and lesbian couples should be allowed to legally marry",
					2: "Gay and lesbian couples should be allowed to form civil unions but not legally marry",
					3: "There should be no legal recognition of gay or lesbian couples’ relationship",
                    }[x]
                ),
                "trump_foreign_relations": (("Do you approve or disapprove of the way Donald Trump is
                                                    "handlingrelations with foreign countries?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Approve",
					2: "Disapprove",
                    }[x]
                ),
                "trump_immigration": (("Do you approve or disapprove of the way Donald Trump is "
                        "handling immigration?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Approve",
					2: "Disapprove",
                    }[x]
                    ),
                "military": (("How willing should the United States be to use military force to "
                            "solve international problems?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Extremely willing",
					2: "Very willing",
					3: "Moderately willing",
					4: "A little willing",
					5: "Not at all willing ",
                    }[x]
                ),
                "sleep": (("In the past week, how often has your sleep been restless?"),
                lambda x: {
					-9: "Refused",
					-5: "Interview breakoff (sufficient partial IW)",
					1: "All the time",
					2: "Often",
					3: "Sometimes",
					4: "Rarely",
					5: "Never",
                    }[x]
                    )
                "health_insurance": (("Do you presently have any kind of health insurance?"),
                lambda x: {
					-9: "Refused",
					-5: "Interview breakoff (sufficient partial IW)",
					1: "Yes",
					2: "No",
                    }[x]
                ),
                "political_campaigns": (("Some people don’t pay much attention to political campaigns. "
                        "How about you? Would you say that you have been very much "
                        "interested, somewhat interested or not much interested in the "
                        "political campaigns so far this year?"),
                lambda x: {
					-9: "Refused",
					1: "Very much interested",
					2: "Somewhat interested",
					3: "Not much interested",
                    }[x]
                ),
                "voting_duty": (("How strongly do you feel that voting is a duty?"),
                lambda x: {
					-9: "Refused",
					-1: "Inapplicable",
					1: "Very strongly",
					2: "Moderately strongly",
					3: "A little strongly",
                    }[x]
                ),
                "government_elite": (("Would you say the government is pretty much run by a few big "
                        "interests looking out for themselves or that it is run for the "
                        "benefit of all the people?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Run by a few big interests",
					2: "For the benefit of all the people",
                    }[x]
                ),
            }

        elif self.format == "FPBS":
            return {
                
                "protect_environment": (("What about protecting the environment? Should federal "
                                            "spending on protecting the environment be increased, "
                                            "decreased, or kept the same?"),
                                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                ),
                "government_temperatures": (("Do you think the federal government should be doing more "
                        "about rising temperatures, should be doing less, or is it currently "
                        "doing the right amount?"),
                lambda x: {
					-9: "Refused"
					-8: "Don’t know"
					1: "Should be doing more"
					2: "Should be doing less"
					3: "Is currently doing the right amount",
                    }[x]
                ),
                "federal_spending_crime": (("What about dealing with crime? Should federal spending on "
                        "dealing with crime be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused"
					-8: "Don’t know"
					1: "Increased"
					2: "Decreased"
					3: "Kept the same"
                    }[x]
                    ),
                "trump_economy": (("Do you approve or disapprove of the way Donald Trump is handling "
                                        "the economy?"),
					                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Approve",
					2: "Disapprove",
                    }[x]
					),
                "government_waste": (("Do you think that people in government waste a lot of the money "
                        "we pay in taxes, waste some of it, or don’t waste very much of "
                        "it?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Waste a lot",
					2: "Waste some",
					3: "Don’t waste very much",
                    }[x]
                ),
                "social_security": (("What about Social Security? Should federal spending on Social "
                                        "Security be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                ),
                "spending_poor": (("What about aid to the poor? Should federal spending on aid to "
                                        "the poor be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                    ),
                "economy_good": (("What do you think about the state of the economy these days in "
                                    "the United States? Would you say the state of the economy is "
                                    "very good,good, neither good nor bad, bad, or very bad?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Very good",
					2: "Good",
					3: "Neither good nor bad",
					4: "Bad",
					5: "Very bad",
                    }[x]
                ),
                "economy_worse_better": (("Now thinking about the economy in the country as a whole, "
                                            "would you say that over the past year the nation’s economy has "
                                            "gotten better, stayed about the same, or gotten worse?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Gotten better",
					2: "Stayed about the same",
					3: "Gotten worse",
                    }[x]
                ),
                "welfare": (("What about welfare programs? Should federal spending on "
                            "welfare programs be increased, decreased, or kept the same?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Increased",
					2: "Decreased",
					3: "Kept the same",
                    }[x]
                ),
                "gay_marriage": (("Which comes closest to your view? You can just tell me the "
                                    "number of your choice."),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Gay and lesbian couples should be allowed to legally marry",
					2: "Gay and lesbian couples should be allowed to form civil unions but not legally marry",
					3: "There should be no legal recognition of gay or lesbian couples’ relationship",
                    }[x]
                ),
                "trump_foreign_relations": (("Do you approve or disapprove of the way Donald Trump is
                                                    "handlingrelations with foreign countries?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Approve",
					2: "Disapprove",
                    }[x]
                ),
                "trump_immigration": (("Do you approve or disapprove of the way Donald Trump is "
                        "handling immigration?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Approve",
					2: "Disapprove",
                    }[x]
                    ),
                "military": (("How willing should the United States be to use military force to "
                            "solve international problems?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Extremely willing",
					2: "Very willing",
					3: "Moderately willing",
					4: "A little willing",
					5: "Not at all willing ",
                    }[x]
                ),
                "sleep": (("In the past week, how often has your sleep been restless?"),
                lambda x: {
					-9: "Refused",
					-5: "Interview breakoff (sufficient partial IW)",
					1: "All the time",
					2: "Often",
					3: "Sometimes",
					4: "Rarely",
					5: "Never",
                    }[x]
                    )
                "health_insurance": (("Do you presently have any kind of health insurance?"),
                lambda x: {
					-9: "Refused",
					-5: "Interview breakoff (sufficient partial IW)",
					1: "Yes",
					2: "No",
                    }[x]
                ),
                "political_campaigns": (("Some people don’t pay much attention to political campaigns. "
                        "How about you? Would you say that you have been very much "
                        "interested, somewhat interested or not much interested in the "
                        "political campaigns so far this year?"),
                lambda x: {
					-9: "Refused",
					1: "Very much interested",
					2: "Somewhat interested",
					3: "Not much interested",
                    }[x]
                ),
                "voting_duty": (("How strongly do you feel that voting is a duty?"),
                lambda x: {
					-9: "Refused",
					-1: "Inapplicable",
					1: "Very strongly",
					2: "Moderately strongly",
					3: "A little strongly",
                    }[x]
                ),
                "government_elite": (("Would you say the government is pretty much run by a few big "
                        "interests looking out for themselves or that it is run for the "
                        "benefit of all the people?"),
                lambda x: {
					-9: "Refused",
					-8: "Don’t know",
					1: "Run by a few big interests",
					2: "For the benefit of all the people",
                    }[x]
                ),
            }









if __name__ == '__main__':
    import random

    ds = AnesDataset(n_exemplars=5)