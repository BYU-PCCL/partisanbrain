import numpy as np

demo_rephrasing_dicts = {
    "party": {
        "Democratic party": "a Democrat",
        "Republican party": "a Republican",
        "None or 'independent'": "an independent",
    },
    # 'High school graduate - High school diploma or equivalent (e.g: GED)', 'Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)', "Bachelor's degree (e.g. BA, AB, BS)", "Master's degree (e.g. MA, MS, MEng, MEd, MSW, MBA)", 'Associate degree in college - academic', 'Some college but no degree', 'Less than high school credential', 'Associate degree in college - occupational/vocational'
    "education": {
        "High school graduate - High school diploma or equivalent (e.g: GED)": "graduated from high school",
        "Professional school degree (e.g. MD, DDS, DVM, LLB, JD)/Doctoral degree (e.g: PHD, EDD)": "went to graduate school",
        "Bachelor's degree (e.g. BA, AB, BS)": "got a bachelor's degree",
        "Master's degree (e.g. MA, MS, MEng, MEd, MSW, MBA)": "got a master's degree",
        "Associate degree in college - academic": "got an associate's degree",
        "Some college but no degree": "went to some college but didn't graduate",
        "Less than high school credential": "didn't graduate from high school",
        "Associate degree in college - occupational/vocational": "got an associate's degree",
    },
    # 'Slightly conservative', 'Liberal', 'Conservative', 'Moderate; middle of the road', 'Slightly liberal', 'Extremely liberal', 'Extremely conservative'
    "ideology": {
        "Slightly conservative": "slightly conservative",
        "Liberal": "liberal",
        "Conservative": "conservative",
        "Moderate; middle of the road": "moderate",
        "Slightly liberal": "slightly liberal",
        "Extremely liberal": "extremely liberal",
        "Extremely conservative": "extremely conservative",
    },
    # -"Refused", -"Interview breakoff", "Under $9,999", "$10,000-14,999", "$15,000-19,999", "$20,000-24,999", "$25,000-29,999", "$30,000-34,999", "$35,000-39,999", "$40,000-44,999", "$45,000-49,999", "$50,000-59,999", "$60,000-64,999", "$65,000-69,999", "$70,000-74,999", "$75,000-79,999", "$80,000-89,999", "$90,000-99,999", "$100,000-109,999", "$110,000-124,999", "$125,000-149,999", "$150,000-174,999", "$175,000-249,999", "$250,00 or more",
    "income": {
        "Under $9,999": "under $9,999",
        "$10,000-14,999": "between $10,000-$14,999",
        "$15,000-19,999": "between $15,000-$19,999",
        "$20,000-24,999": "between $20,000-$24,999",
        "$25,000-29,999": "between $25,000-$29,999",
        "$30,000-34,999": "between $30,000-$34,999",
        "$35,000-39,999": "between $35,000-$39,999",
        "$40,000-44,999": "between $40,000-$44,999",
        "$45,000-49,999": "between $45,000-$49,999",
        "$50,000-59,999": "between $50,000-$59,999",
        "$60,000-64,999": "between $60,000-$64,999",
        "$65,000-69,999": "between $65,000-$69,999",
        "$70,000-74,999": "between $70,000-$74,999",
        "$75,000-79,999": "between $75,000-$79,999",
        "$80,000-89,999": "between $80,000-$89,999",
        "$90,000-99,999": "between $90,000-$99,999",
        "$100,000-109,999": "between $100,000-$109,999",
        "$110,000-124,999": "between $110,000-$124,999",
        "$125,000-149,999": "between $125,000-$149,999",
        "$150,000-174,999": "between $150,000-$174,999",
        "$175,000-249,999": "between $175,000-$249,999",
        "$250,00 or more": "over $250,000",
    },
    # 'Undifferentiated Protestant', 'Not religious', 'Mainline Protestant', 'Roman Catholic', 'Other religion', 'Other Christian', 'Evangelical Protestant', 'Jewish'
    "religion": {
        "Undifferentiated Protestant": "Religiously speaking, I am a Protestant",
        "Not religious": "I am not religious",
        "Mainline Protestant": "Religiously speaking, I am a Protestant",
        "Roman Catholic": "Religiously speaking, I am Roman Catholic",
        "Other religion": "I am religious and I am not Christian",
        "Other Christian": "Religiously speaking, I am Christian",
        "Evangelical Protestant": "Religiously speaking, I am an Evangelical Protestant",
        "Jewish": "Religiously speaking, I am Jewish",
    },
    # 'White, non-Hispanic'|'Black, non-Hispanic'|'Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone'|'Hispanic'|'Multiple races, non-Hispanic'|'Native American/Alaska Native or other race, non-Hispanic alone'
    "race_ethnicity": {
        "White, non-Hispanic": "white",
        "Black, non-Hispanic": "Black",
        "Asian or Native Hawaiian/other Pacific Islander, non-Hispanic alone": "Asian-American/Pacific Islander",
        "Hispanic": "Hispanic",
        "Multiple races, non-Hispanic": "multiracial",
        "Native American/Alaska Native or other race, non-Hispanic alone": "Native American",
    },
    # 'Married: spouse present', 'Never married', 'Divorced', 'Married: spouse absent {VOL - video/phone only}', 'Widowed', 'Separated'
    "marital_status": {
        "Married: spouse present": "am married",
        "Never married": "am single and have never been married",
        "Divorced": "am divorced",
        "Married: spouse absent {VOL - video/phone only}": "am married but don't live with my spouse",
        "Widowed": "am widowed",
        "Separated": "am separated from my spouse",
    },
}

dv_rephrasing_dicts = {
    # DVS
    "gay_marriage": {
        "colloquial": {
            "rephrasing": {
                "Gay and lesbian couples should be allowed to legally marry": "legal marriages",
                "Gay and lesbian couples should be allowed to form civil unions but not legally marry": "civil unions",
                "There should be no legal recognition of gay or lesbian couples' relationship": "neither",
            },
            "tokens": {
                "Gay and lesbian couples should be allowed to legally marry": [
                    "legal marriages"
                ],
                "Gay and lesbian couples should be allowed to form civil unions but not legally marry": [
                    "civil unions"
                ],
                "There should be no legal recognition of gay or lesbian couples' relationship": [
                    "neither"
                ],
            },
        },
        "mc": {
            "rephrasing": {
                "Gay and lesbian couples should be allowed to legally marry": "A. Legal marriages",
                "Gay and lesbian couples should be allowed to form civil unions but not legally marry": "B. Civil unions",
                "There should be no legal recognition of gay or lesbian couples' relationship": "C. Neither",
            },
            "tokens": {
                "Gay and lesbian couples should be allowed to legally marry": [
                    "A",
                    "legal marriages",
                ],
                "Gay and lesbian couples should be allowed to form civil unions but not legally marry": [
                    "B",
                    "civil unions",
                ],
                "There should be no legal recognition of gay or lesbian couples' relationship": [
                    "C",
                    "neither",
                ],
            },
        },
    },
    "trump_handling_relations": {
        "colloquial": {
            "rephrasing": {
                "Don't know": "don't know",
                "Approve": "approve",
                "Disapprove": "disapprove",
            },
            "tokens": {
                "Don't know": "don't know",
                "Approve": "approve",
                "Disapprove": "disapprove",
            },
        },
        "mc": {
            "rephrasing": {
                "Don't know": "A. Don't know",
                "Approve": "B. Approve",
                "Disapprove": "C. Disapprove",
            },
            "tokens": {
                "Don't know": "don't know",
                "Approve": "approve",
                "Disapprove": "disapprove",
            },
        },
    },
    "trump_handling_relations_mc": {
        "colloquial": {
            "rephrasing": {
                "Don't know": "I don't know",
                "Approve": "I approve",
                "Disapprove": "I disapprove",
            },
            "tokens": {
                "Don't know": "I don't know",
                "Approve": "I approve",
                "Disapprove": "I disapprove",
            },
        },
        "mc": {
            "rephrasing": {
                "Don't know": "I don't know",
                "Approve": "I approve",
                "Disapprove": "I disapprove",
            },
            "tokens": {
                "Don't know": "I don't know",
                "Approve": "I approve",
                "Disapprove": "I disapprove",
            },
        },
    },
    "trump_handling_immigration": {
        "colloquial": {
            "rephrasing": {
                "Don't know": "",
                "Approve": "",
                "Disapprove": "",
            },
            "tokens": {
                "Don't know": "",
                "Approve": "",
                "Disapprove": "",
            },
        },
        "mc": {
            "rephrasing": {
                "Don't know": "",
                "Approve": "",
                "Disapprove": "",
            },
            "tokens": {
                "Don't know": "",
                "Approve": "",
                "Disapprove": "",
            },
        },
    },
    "willing_military_force": {
        "colloquial": {
            "rephrasing": {},
            "Don't know": "",
            "Extremely willing": "",
            "Very willing": "",
            "Moderately willing": "",
            "A little willing": "",
            "Not at all willing": "",
            "tokens": {
                "Don't know": "",
                "Extremely willing": "",
                "Very willing": "",
                "Moderately willing": "",
                "A little willing": "",
                "Not at all willing": "",
            },
        },
        "mc": {
            "rephrasing": {
                "Don't know": "",
                "Extremely willing": "",
                "Very willing": "",
                "Moderately willing": "",
                "A little willing": "",
                "Not at all willing": "",
            },
            "tokens": {
                "Don't know": "",
                "Extremely willing": "",
                "Very willing": "",
                "Moderately willing": "",
                "A little willing": "",
                "Not at all willing": "",
            },
        },
    },
    "restless_sleep": {
        "colloquial": {
            "rephrasing": {
                "All the time": "",
                "Often": "",
                "Sometimes": "",
                "Rarely": "",
                "Never": "",
            },
            "tokens": {
                "All the time": "",
                "Often": "",
                "Sometimes": "",
                "Rarely": "",
                "Never": "",
            },
        },
        "mc": {
            "rephrasing": {
                "All the time": "",
                "Often": "",
                "Sometimes": "",
                "Rarely": "",
                "Never": "",
            },
            "tokens": {
                "All the time": "",
                "Often": "",
                "Sometimes": "",
                "Rarely": "",
                "Never": "",
            },
        },
    },
    "have_health_insurance": {
        "colloquial": {
            "rephrasing": {
                "Yes": "",
                "No": "",
            },
            "tokens": {
                "Yes": "",
                "No": "",
            },
        },
        "mc": {
            "rephrasing": {
                "Yes": "",
                "No": "",
            },
            "tokens": {
                "Yes": "",
                "No": "",
            },
        },
    },
    "attn_to_politics": {
        "colloquial": {
            "rephrasing": {
                "Very much interested": "",
                "Somewhat interested": "",
                "Not much interested": "",
            },
            "tokens": {
                "Very much interested": "",
                "Somewhat interested": "",
                "Not much interested": "",
            },
        },
        "mc": {
            "rephrasing": {
                "Very much interested": "",
                "Somewhat interested": "",
                "Not much interested": "",
            },
            "tokens": {
                "Very much interested": "",
                "Somewhat interested": "",
                "Not much interested": "",
            },
        },
    },
    "feel_voting_is_duty": {
        "colloquial": {
            "rephrasing": {
                "Very strongly": "",
                "Moderately strongly": "",
                "A little strongly": "",
            },
            "tokens": {
                "Very strongly": "",
                "Moderately strongly": "",
                "A little strongly": "",
            },
        },
        "mc": {
            "rephrasing": {
                "Very strongly": "",
                "Moderately strongly": "",
                "A little strongly": "",
            },
            "tokens": {
                "Very strongly": "",
                "Moderately strongly": "",
                "A little strongly": "",
            },
        },
    },
    "govt_run_by_who": {
        "colloquial": {
            "rephrasing": {
                "Don't know": "",
                "Run by a few big interests": "",
                "For the benefit of all people": "",
            },
            "tokens": {
                "Don't know": "",
                "Run by a few big interests": "",
                "For the benefit of all people": "",
            },
        },
        "mc": {
            "rephrasing": {
                "Don't know": "",
                "Run by a few big interests": "",
                "For the benefit of all people": "",
            },
            "tokens": {
                "Don't know": "",
                "Run by a few big interests": "",
                "For the benefit of all people": "",
            },
        },
    },
    "vote_2016": {
        "colloquial": {
            "rephrasing": {
                "Donald Trump": "Donald Trump",
                "Hillary Clinton": "Hillary Clinton",
            },
            "tokens": {
                "Donald Trump": "Donald Trump",
                "Hillary Clinton": "Hillary Clinton",
            },
        },
        "mc": {
            "rephrasing": {
                "Donald Trump": "Donald Trump",
                "Hillary Clinton": "Hillary Clinton",
            },
            "tokens": {
                "Donald Trump": "Donald Trump",
                "Hillary Clinton": "Hillary Clinton",
            },
        },
    },
}


# "social_security" : {
#     "Increased": "Increased",
#     "Decreased": "Decreased",
#     "Kept the same": "Kept the same",
# },

# "protecting_environment_spending" : {
#     "Increased": "Increased",
#     "Decreased": "Decreased",
#     "Kept the same": "Kept the same",
# },

# "rising_temp_action" : {
#     "Should be doing more": "more",
#     "Should be doing less": "less",
#     "Is currently doing the right amount": ["the", "about"],
# },
# "dealing_with_crime" : None

# "trump_handling_economy" : {
#     "Approve": "approves",
#     "Disapprove": "disapproves",
# },
# "govt_waste_money" : {
#     "Don't waste very much": ["none", "none of it"],
#     "Waste a lot": ["a", "all", "a lot"],
#     "Waste some": ["some", "some of it"],
# },
# "worried_financial_situation" : {
#     "Moderately worried": ["worried"],
#     "Extremely worried": ["extremely"],
#     "Very worried": ["very"],
#     "A little worried": ["a little"],
#     "Not at all worried": ["not"],
# },
# "gay_marriage" : {
#     "Gay and lesbian couples should be allowed to legally marry": "fully",
#     "Gay and lesbian couples should be allowed to form civil unions but not legally marry": "somewhat",
#     "There should be no legal recognition of gay or lesbian couples' relationship": "not",
# },

# # Social Security Spending
# "social_security_tokens" : {
#     "increased": ["increased"],
#     "decreased": ["decreased"],
#     "kept the same": ["kept the same"],
# },

# "social_security_mc_tokens" : {
#     "increased": ["A", "increased"],
#     "decreased": ["B", "decreased"],
#     "kept the same": ["C", "kept the same"],
# },

# # protecting the environment
# "protecting_environment_tokens" : {
#     "Increased": ["increased"],
#     "Decreased": ["decreased"],
#     "Kept the same": ["kept the same"],
# },

# "protecting_environment_mc_tokens" : {
#     "Increased": ["A", "increased"],
#     "Decreased": ["B", "decreased"],
#     "Kept the same": ["C", "kept the same"],
# },

# # Vote 2016
# "vote_2016" : {
#     "Hillary Clinton": ["Hillary", "Clinton"],
#     "Donald Trump": ["Donald", "Trump"],
# },

# "vote_2016_mc" : {
#     "Hillary Clinton": ["A", "Hillary", "Clinton"],
#     "Donald Trump": ["B", "Donald", "Trump"],
# },

# # rising_temp_action
# "rising_temp_action_tokens" : {
#     "More": ["more"],
#     "Less": ["less"],
#     "The same": ["the same", "the", "same", "about the same"],
# },
# "rising_temp_action_mc_tokens" : {
#     "More": ["A", "more"],
#     "Less": ["B", "less"],
#     "The same": [
#         "C",
#         "currently doing the right amount",
#         "the same",
#         "the",
#         "same",
#         "about the same",
#     ],
# },
