from dataset import Dataset, PromptSpecs


class GSSDataset(Dataset):

    def __init__(self):
        survey_fname = "data/GSS2018.sav"
        super().__init__(survey_fname)

    def _filter_demographics(self, df):
        new_df = df.loc[~df['partisanship'].isin(['OTHER PARTY'])]
        new_df = new_df.loc[~new_df['religiosity'].isin(['INTER-NONDENOMINATIONAL',
                                                         'OTHER EASTERN',
                                                         'OTHER',
                                                         'NATIVE AMERICAN'])]
        new_df = new_df.loc[~new_df['race'].isin(['SOME OTHER RACE'])]

        return new_df

    def _filter_to_usa(self, df):
        return df

    def _get_dv_col_names(self):
        return {
            'NATENVIR': 'environment',
            'NATENRGY': 'green_energy',
            'CAPPUN': 'death_penalty',
            'POLHITOK': 'police_violence',
            'GRASS': 'marijuana',
            'WKSEXISM': 'gender_discrimination',
            'FEPOL': 'women_in_politics',
            'HLTHMNTL': 'mental_health_rating',
            'MENTLOTH': 'therapist_rec',
            'FAMMHNEG': 'fam_mental_health_views',
            'DIAGNOSD': 'mental_health_diagnosis',
            'OTHMHNEG': 'non_fam_mental_health_views',
            'HLTHPHYS': 'physical_health_rating',
            'VOTE16': 'voted',
            'SPKRAC': 'racist_speech',
            'BIBLE': 'bible_beliefs',
            'PRAYER': 'church_power',
            'CONCLERG': 'rel_leader_confidence',
            'RELITEN': 'religious_rating',
            'POSTLIFE': 'afterlife',
        }

    def _get_demographic_col_names(self):
        return {
            'AGE': 'age',
            'SEX': 'gender',
            'PARTYID': 'partisanship',
            'EDUC': 'education',
            'POLVIEWS': 'ideology',
            'INCOME': 'income',
            'RELIG': 'religiosity',
            'RACECEN1': 'race',
            'REGION': 'region',
            'MARITAL': 'marital',
        }

    def _make_backstory(self, row):
        backstory = []

        # Age
        if row['age'] == '89 OR OLDER':
            backstory.append('I am over 89 years old.')
        else:
            backstory.append(f'I am {int(row["age"])} years old.')

        # Gender
        gender = row["gender"].lower()
        backstory.append(f'I am {gender}.')

        # Partisanship
        party_dict = {
            'NOT STR REPUBLICAN': 'am slightly Republican',
            'IND,NEAR DEM': 'lean Democrat',
            'IND,NEAR REP': 'lean Republican',
            'STRONG REPUBLICAN': 'am strongly Republican',
            'STRONG DEMOCRAT': 'am strongly Democrat',
            'INDEPENDENT': 'am independent',
            'NOT STR DEMOCRAT': 'am slightly Democrat',
        }
        party = party_dict[row['partisanship']]
        backstory.append(f'In terms of political parties I {party}.')

        # Education
        educ_int = int(row['education'])
        education_dict = {
            0: 'I have never gone to school.',
            1: '1st grade',
            2: '2nd grade',
            3: '3rd grade',
            4: '4th grade',
            5: '5th grade',
            6: '6th grade',
            7: '7th grade',
            8: '8th grade',
            9: '9th grade',
            10: '10th grade',
            11: '11th grade',
            12: '12th grade',
            13: '1 year',
            14: '2 years',
            15: '3 years',
            16: '4 years',
            17: '5 years',
            18: '6 years',
            19: '7 years',
            20: '8 years',
        }
        education = education_dict[educ_int]

        if educ_int == 0:
            backstory.append(education)
        elif 1 <= educ_int <= 12:
            backstory.append(f"The highest year of school I have completed is {education}.")
        else:
            backstory.append(f"I have finished {education} of college.")

        # Ideology
        ideology_dict = {
            'CONSERVATIVE': 'conservative',
            'SLGHTLY CONSERVATIVE': 'slightly conservative',
            'MODERATE': 'moderate',
            'EXTRMLY CONSERVATIVE': 'extremely conservative',
            'SLIGHTLY LIBERAL': 'slightly liberal',
            'LIBERAL': 'liberal',
            'EXTREMELY LIBERAL': 'extremely liberal',
        }
        ideology = ideology_dict[row['ideology']]
        backstory.append(f"In terms of political ideology, I would consider myself to be {ideology}.")

        # Income
        income_dict = {
            '$25000 OR MORE': 'over $25,000',
            '$15000 - 19999': 'between $15,000 and $19,999',
            '$5000 TO 5999': 'between $5,000 and $5,999',
            '$20000 - 24999': 'between $20,000 and $24,999',
            '$1000 TO 2999': 'between $1,000 and $2,999',
            '$10000 - 14999': 'between $10,000 and $14,999',
            '$8000 TO 9999': 'between $8,000 and $9,999',
            '$7000 TO 7999': 'between $7,000 and $7,999',
            'LT $1000': 'less than $1,000',
            '$4000 TO 4999': 'between $4,000 and $4,999',
            '$3000 TO 3999': 'between $3,000 and $3,999',
            '$6000 TO 6999': 'between $6,000 and $6,999',
        }
        income = income_dict[row['income']]
        backstory.append(f"My family income is {income} per year.")

        # Religiosity
        religion_dict = {
            'CHRISTIAN': 'Christian',
            'CATHOLIC': 'Catholic',
            'NONE': 'not religious',
            'PROTESTANT': 'Protestant',
            'ORTHODOX-CHRISTIAN': 'an Orthodox Christian',
            'BUDDHISM': 'Buddhist',
            'JEWISH': 'Jewish',
            'HINDUISM': 'Hindu',
            'MOSLEM/ISLAM': 'Muslim',
        }
        religion = religion_dict[row['religiosity']]
        backstory.append(f"I am {religion}.")

        # Race
        race_dict = {
            'AMERICAN INDIAN OR ALASKA NATIVE': 'Native American',
            'ASIAN INDIAN': 'Indian',
            'BLACK OR AFRICAN AMERICAN': 'Black',
            'CHINESE': 'Chinese',
            'FILIPINO': 'Filipino',
            'GUAMANIAN OR CHAMORRO': 'Guamanian',
            'HISPANIC': 'Hispanic',
            'JAPANESE': 'Japanese',
            'KOREAN': 'Korean',
            'OTHER ASIAN': 'Asian',
            'OTHER PACIFIC ISLANDER': 'a Pacific Islander',
            'VIETNAMESE': 'Vietnamese',
            'WHITE': 'White',
        }
        race = race_dict[row['race']]
        backstory.append(f"I am {race}.")

        # Region
        region_dict = {
            'E. NOR. CENTRAL': 'Midwest',
            'E. SOU. CENTRAL': 'South',
            'MIDDLE ATLANTIC': 'Eastern Seaboard',
            'MOUNTAIN': 'West',
            'NEW ENGLAND': 'New England',
            'PACIFIC': 'West',
            'SOUTH ATLANTIC': 'South',
            'W. NOR. CENTRAL': 'Midwest',
            'W. SOU. CENTRAL': 'South',
        }
        region = region_dict[row['region']]
        backstory.append(f"I live in the {region}.")

        # Marital
        marital_status_dict = {
            'DIVORCED': 'am divorced',
            'MARRIED': 'am married',
            'NEVER MARRIED': 'have never been married',
            'SEPARATED': 'am separated from my spouse',
            'WIDOWED': 'am widowed',
        }
        marital_status = marital_status_dict[row['marital']]
        backstory.append(f"I {marital_status}.")

        backstory.append("It is 2018.")

        return " ".join(backstory)

    def _get_col_prompt_specs(self):
        return {
            'environment': PromptSpecs(
                question="Is our spending excessive, inadequate, or about the right amount on improving and protecting the environment?",
                answer_prefix="our spending is",
                answer_map={'TOO LITTLE': 'inadequate', 'ABOUT RIGHT': 'about right', 'TOO MUCH': 'excessive'}
            ),
            'green_energy': PromptSpecs(
                question="Is our spending excessive, inadequate, or about the right amount on developing alternative energy sources?",
                answer_prefix="our spending is",
                answer_map={'Too little': 'inadequate', 'About right': 'about right', 'Too much': 'excessive'}
            ),
            'death_penalty': PromptSpecs(
                question="Do you favor or oppose the death penalty for persons convicted of murder?",
                answer_prefix="I",
                answer_map={'FAVOR': 'favor', 'OPPOSE': 'oppose'}
            ),
            'police_violence': PromptSpecs(
                question="Are there any situations you can imagine in which you would approve of a policeman striking an adult male citizen?",
                answer_prefix="",
                answer_map={'NO': 'no', 'YES': 'yes'}
            ),
            'marijuana': PromptSpecs(
                question="Do you think the use of marijuana should be legal or illegal?",
                answer_prefix="marijuana should be",
                answer_map={'LEGAL': 'legal', 'NOT LEGAL': 'illegal'}
            ),
            'gender_discrimination': PromptSpecs(
                question="Do you feel in any way discriminated against on your job because of your gender?",
                answer_prefix="",
                answer_map={'YES': 'yes', 'NO': 'no'}
            ),
            'women_in_politics': PromptSpecs(
                question="Tell me if you agree or disagree with this statement: Most men are better suited emotionally for politics than are most women",
                answer_prefix="I",
                answer_map={'DISAGREE': 'disagree', 'AGREE': 'agree'}
            ),
            'mental_health_rating': PromptSpecs(
                question="In general, how would you rate your mental health, including your mood and your ability to think?",
                answer_prefix="my mental health is",
                answer_map={'Good': 'good', 'Very good': 'excellent', 'Fair': 'fair', 'Excellent': 'excellent', 'Poor': 'poor'}
            ),
            'therapist_rec': PromptSpecs(
                question=("John is a white man with a college education. For the past two weeks John has been feeling really "
                "down. He wakes up in the morning with a flat heavy feeling that sticks with him all day long. He is not "
                "enjoying things the way he normally would. In fact nothing gives him pleasure. Even when good things happen, "
                "they do not seem to make John happy. He pushes on through his days, but it is really hard. The smallest tasks "
                "are difficult to accomplish. He finds it hard to concentrate on anything. He feels out of energy and out of "
                "steam. And even though John feels tired, when night comes he cannot go to sleep. John feels pretty worthless, "
                "and very discouraged. John's family has noticed that he has not been himself for about the last month and that "
                "he has pulled away from them. John just does not feel like talking. Should John go to a therapist, or counselor, "
                "like a psychologist, social worker, or other mental health professional for help?"),
                answer_prefix="",
                answer_map={'YES': 'yes', 'NO': "no"}
            ),
            'fam_mental_health_views': PromptSpecs(
                question="Thinking about your family, to what extent do they hold negative attitudes about people with mental health problems?",
                answer_prefix="their attitudes are",
                answer_map={'Somewhat': 'somewhat negative', 'Not at all': 'not negative', 'Not very much': 'not negative', 'Very much': 'very negative'}
            ),
            'mental_health_diagnosis': PromptSpecs(
                question="Have you ever been diagnosed with a mental health problem?",
                answer_prefix="",
                answer_map={'No': 'no', 'Yes': 'yes'}
            ),
            'non_fam_mental_health_views': PromptSpecs(
                question=("Thinking about other people you know personally outside of your family, to what extent do they hold negative attitudes "
                          "about people with mental health problems?"),
                answer_prefix="their attitudes are",
                answer_map={'Somewhat': 'somewhat negative', 'Not at all': 'not negative', 'Not very much': 'not negative', 'Very much': 'very negative'}
            ),
            'physical_health_rating': PromptSpecs(
                question="In general, how would you rate your physical health?",
                answer_prefix="my physical health is",
                answer_map={'Good': 'good', 'Very good': 'excellent', 'Fair': 'fair', 'Excellent': 'excellent', 'Poor': 'poor'}
            ),
            'voted': PromptSpecs(
                question=("In 2016, you remember that Clinton ran for President on the Democratic ticket against Trump for the Republicans. "
                          "Do you remember for sure whether or not you voted in that election?"),
                answer_prefix="I",
                answer_map={'Voted': 'voted', 'Did not vote': "didn't vote", 'Ineligible': 'am ineligible'}
            ),
            'racist_speech': PromptSpecs(
                question=("Consider a person who believes that Blacks are genetically inferior. If such a person wanted to make a speech "
                          "in your community claiming that Blacks are inferior, should he be allowed to speak?"),
                answer_prefix="",
                answer_map={'ALLOWED': 'yes', 'NOT ALLOWED': 'no'}
            ),
            'bible_beliefs': PromptSpecs(
                question=("Which of these statements comes closest to describing your feelings about the Bible? The Bible is the actual "
                          "word of God and is to be taken literally, word for word. The Bible is the inspired word of God but not "
                          "everything in it should be taken literally, word for word. The Bible is an ancient book of fables, "
                          "legends, history, and moral precepts recorded by man."),
                answer_prefix="the Bible is",
                answer_map={'INSPIRED WORD': 'inspired word', 'BOOK OF FABLES': 'a book of fables', 'WORD OF GOD': 'the word of god'}
            ),
            'church_power': PromptSpecs(
                question=("The United States Supreme Court has ruled that no state or local government may require the reading of the "
                          "Lord's Prayer or Bible verses in public schools. What are your views on this--do you approve or disapprove "
                          "of the court ruling?"),
                answer_prefix="I",
                answer_map={'APPROVE': 'approve', 'DISAPPROVE': 'disapprove'}
            ),
            'rel_leader_confidence': PromptSpecs(
                question=("As far as the people running organized religion are concerned, would you say you have a lots of "
                          "confidence, some confidence, or hardly any confidence at all in them?"),
                answer_prefix="I have",
                answer_map={'HARDLY ANY': 'hardly any confidence', 'A GREAT DEAL': 'lots of confidence', 'ONLY SOME': 'some confidence'}
            ),
            'religious_rating': PromptSpecs(
                question=("How religious are you?"),
                answer_prefix="I am",
                answer_map={'NO RELIGION': 'not religious', 'STRONG': 'very religious', 'NOT VERY STRONG': 'somewhat religious', 'SOMEWHAT STRONG': 'somewhat religious'}
            ),
            'afterlife': PromptSpecs(
                question=("Do you believe there is a life after death?"),
                answer_prefix="",
                answer_map={'YES': 'yes', 'NO': 'no'}
            ),
        }

if __name__ == '__main__':
    ds = GSSDataset()
    backstories = ds.get_backstories_all_demos()
    for backstory in backstories:
        print(f"{backstory[0]}\n\n{backstory[1]}\n\n")
    prompts = ds.get_prompts_sample()
    for prompt in prompts:
        print(f"{prompt}\n\n")