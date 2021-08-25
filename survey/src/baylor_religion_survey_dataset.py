class BaylorReligionSurveyDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "data/Baylor Religion Survey, Wave V (2017).SAV"
        super().__init__(survey_fname, n_exemplars)


        #issues: no region demographic

    def _format(self, df):

        # Dropping all but relevant columns
        new_df = df[["AGE",
                     "Q77",
                     "Q32",
                     "I-EDUC",
                     "Q31",
                     "Q95",
                     "Q1",
                     "RACE",
                     "D9",
                     "L14_2F",
                     "MP4A",
                     "Q16D",
                     "MP4G",
                     "MP4F",
                     "MP4D",
                     "MP4J",
                     "MP4K",
                     "Q45",
                     "H13E",
                     "H12",
                     "MP4H",
                     "MP4I",
                     "Q17",
                     "Q18",
                     "Q19A",
                     "Q19D",
                     "Q4",
                     "MP12F",
                     ("R20F")]]

        # Dropping rows with NA values
        new_df = new_df.dropna(axis=0)

        # Renaming columns for convenience
        new_df = new_df.rename({"Gender": "age",
                                "Q77": "gender",
                                "Q32": "party",
                                "I-EDUC": "edu",
                                "Q31": "ideology",
                                "Q95": "income",
                                "Q1": "religion",
                                "RACE": "race",
                                "D9": "marital",
                                "L14_2F": "lost_job",
                                "MP4A": "trans_restrooms",
                                "Q61D": "gay_is_it_choice",
                                "MP4G": "husband_salary",
                                "MP4F": "women_childcare",
                                "MP4D": "men_suited_politics",
                                "MP4J": "refugees_terrorist_threat",
                                "MP4K": "mexican_immigrants_criminals",
                                "Q45": "life_happiness",
                                "H13E": "depressed_freq",
                                "H12": "days_of_exercise",
                                "MP4H": "police_racial_treatment",
                                "MP4I": "racial_violence",
                                "Q17": "bible_beliefs",
                                "Q18": "god_beliefs",
                                "Q19A": "god_concern_for_world",
                                "Q19D": "god_concern_for_individuals",
                                "Q4": "church_attendance",
                                "MP12F": "prayer_in_school",
                                ("R20F"): "gods_plan"},
                               axis=1)

        # Randomly sample columns!

        # Get only top 8 rows to keep things simple for testing
        new_df = new_df.head(105)

        return new_df

    def _make_backstory(self, row):
        return f"I am a {row['age']} years old. I am {row['gender'].lower()}. I am in the {row['party']} political party. education sentence. In terms of political ideology, I'd consider myself to be {row['ideology']}. My family income is ${row['income']} per year. religion sentence. I'm {row['marital']}."


    def _get_prompt_instructions(self):
        return {"shot_first": (("Between Han and Greedo I think the one "
                                "who shot first was"),
                               lambda x: x),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}
