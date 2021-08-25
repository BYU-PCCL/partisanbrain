class BaylorReligionSurveyDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "data/Baylor Religion Survey, Wave V (2017).SAV"
        super().__init__(survey_fname, n_exemplars)


        #issues:
        # no region demographic

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

        dv_col_names = ["L14_2F",
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

        # Dropping rows with problematic values
        new_df = new_df.dropna(axis=0)
        # TODO : get rid of don't know or other religion rows


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
        backstory = []

        #AGE
        backstory.append(f"I am a {row['age']} years old. ")

        #GENDER
        backstory.append(f"I am {row['gender'].lower()}. ")

        #POLITICAL PARTY
        backstory.append(f"I am in the {row['party']} political party. ")

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
        #
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
            backstory.append("I am a member of the Reformed Church in America. ")
        if row["religion"] == "Non-denominational Christian":
            backstory.append("I am a Christian. ")
        if row["religion"] == "No religion":
            backstory.append("I am not religious. ")





        #REGION
        #missing for baylor study

        #MARITAL STATUS
        backstory.append(f"I'm {row['marital']}.")

        return backstory


    def _get_prompt_instructions(self):
        return {"shot_first": (("Between Han and Greedo I think the one "
                                "who shot first was"),
                               lambda x: x),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}
