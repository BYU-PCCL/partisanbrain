class BaylorReligionSurveyDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "data/Baylor Religion Survey, Wave V (2017).SAV"
        super().__init__(survey_fname, n_exemplars)
        #issues:
        # no region demographic
        # L14_2F -> lost job DV is so weird

    def _format(self, df):

        # Dropping all but relevant columns
        demographic_col_names = ["AGE",
                     "Q77",
                     "Q32",
                     "I-EDUC",
                     "Q31",
                     "Q95",
                     "Q1",
                     "RACE",
                     "D9"]

        dv_col_names = ["T7G",
                        "MP4A",
                        "Q61D",
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
                        "R20F"]

        new_df = df[demographic_col_names + dv_col_names]

        # Renaming columns for convenience
        new_df = new_df.rename({"AGE": "age",
                                "Q77": "gender",
                                "Q32": "party",
                                "I-EDUC": "edu",
                                "Q31": "ideology",
                                "Q95": "income",
                                "Q1": "religion",
                                "RACE": "race",
                                "D9": "marital",
                                "T7G": "tech_oppor",
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

        # Dropping rows with problematic values
        new_df = new_df.dropna(axis=0)
        new_df = new_df[new_df["religion"] != "Other" OR "Don't know"]
        new_df = new_df[new_df["race"] != "No races chosen"]

        # Randomly sample 500 + self._n_exemplars rows
        new_df = new_df.sample(n=500+self._n_exemplars, random_state=0)

        return new_df

    def _make_backstory(self, row):
        backstory = []

        #AGE
        backstory.append(f"I am a {row['age']} years old. ")

        #GENDER
        if row["gender"] == "Other":
            backstory.append("I don't identify as male or female. ")
        else:
            backstory.append(f"I am {row['gender'].lower()}. ")

        #POLITICAL PARTY
        backstory.append(f"In terms of partisan politics, I am a {row['party']}. ")

        #EDUCATION
        if row["edu"] == "No high school degree":
            backstory.append("I did not graduate high school. ")
        if row["edu"] == "High school graduate (Grade 12 with diploma or GED certificate)":
            backstory.append("I am a high school graduate. ")
        if row["edu"] == "Some college":
            backstory.append("I have some college education. ")
        if row["edu"] == "Four year bachelor's degree from a college or university (e.g., BS, BA, AB)":
            backstory.append("I have a bachelor's degree from a college or university. ")
        if row["edu"] == "Postgraduate":
            backstory.append("I have a postgraduate degree. ")

        #POLITICAL IDEOLOGY
        backstory.append(f"In terms of political ideology, I'd consider myself to be {row['ideology']}. ")

        #INCOME
        backstory.append(f"My family income is ${row['income']} per year. ")

        #RELIGION
        #main cases
        if row["religion"] == "Assemblies of God" OR "Bible Church" OR "Brethren" OR "Christian & Missionary Alliance" OR "Christian Reformed" OR "Christian Science" OR "Congregational" OR "Holiness" OR "Lutheran" OR "Pentecostal" OR "Unitarian Universalist":
            backstory.append(f"I am a member of the {row['religion']} faith. ")
        if row["religion"] == "Baha'i" OR "Adventist" OR "African Methodist" OR "Anabaptist" OR "Baptist" OR "Buddhist" OR "Hindu" OR "Jewish" OR "Mennonite" OR "Methodist" OR "Muslim" OR "Presbyterian" OR "Seventh-Day Adventist" OR "Sikh":
            backstory.append(f"In terms of religion, I am a {row['religion']}. ")
        if row["religion"] == "Church of Christ" OR "Church of God" OR "Church of the Nazarene" OR "Jehovah's Witnesses" OR "Salvation Army" OR "United Church of Christ":
            backstory.append(f"I am a member of the {row['religion']}. ")

        #special cases
        if row["religion"] == "Asian Folk Religion":
            backstory.append("I am part of an Asian Folk Religion. ")
        if row["religion"] == "Catholic/Roman Catholic":
            backstory.append("In terms of religion, I am a Catholic. ")
        if row["religion"] == "Episcopal/Anglican":
            backstory.append("I am a member of the Anglican faith. ")
        if row["religion"] == "Latter-day Saints":
            backstory.append("In terms of religion, I am a Mormon. ")
        if row["religion"] == "Orthodox (Eastern, Russian, Greek)":
            backstory.append("I am a member of the Orthodox Catholic Church. ")
        if row["religion"] == "Quaker/Friends":
            backstory.append("In terms of religion, I am a Quaker. ")
        if row["religion"] == "Reformed Church in America/Dutch Reformed":
            backstory.append("I am a member of the Dutch Reformed Church. ")
        if row["religion"] == "Non-denominational Christian":
            backstory.append("I am a Christian. ")
        if row["religion"] == "No religion":
            backstory.append("I am not religious. ")


        #RACE
        backstory.append(f"I am {row['race']}")


        #REGION
        #missing for baylor study

        #MARITAL STATUS
        if row["marital"] == "Single/never been married":
            backstory.append("I am single.")
        elif row["marital"] == "Domestic partnership/living with partner (not legally married)":
            backstory.append("I am not married but I am living with my partner.")
        else:
            backstory.append(f"I'm {row['marital']}.")

        return backstory


    def _get_prompt_instructions(self):
        return {"tech_oppor":(("Did any of these things occur in the PAST YEAR? "
                            "What was its effect on you? Impact on you "
                            "personally: Lost a job"),
                            lambda x: {"Excellent": "excellent",
                                        "Good": "good",
                                        "Only fair": "fair",
                                        "Poor": "poor"}[x]),
                "trans_restrooms":((),
                "gay_is_it_choice":((),
                "husband_salary":((),
                "women_childcare":((),
                "men_suited_politics":((),
                "refugees_terrorist_threat":((),
                "mexican_immigrants_criminals":((),
                "life_happiness":((),
                "depressed_freq":((),
                "days_of_exercise":((),
                "police_racial_treatment":((),
                "racial_violence":((),
                "bible_beliefs":((),
                "god_beliefs":((),
                "god_concern_for_world":((),
                "god_concern_for_individuals":((),
                "church_attendance":((),
                "prayer_in_school":((),
                "gods_plan":((),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}
