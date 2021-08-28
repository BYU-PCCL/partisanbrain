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
        new_df = new_df[new_df]["bible_beliefs"] != "I don't know."

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
        return {"tech_oppor":(("To what extent do you agree with the following? "
                                "Technology gives me new and better employment "
                                "opportunities."),
                                lambda x: {"Strongly agree": "agree",
                                           "Agree": "agree",
                                           "Disagree": "disagree",
                                           "Strongly Disagree": "disagree"}[x]),
                "trans_restrooms":(("Please rate the extent to which you agree "
                                    "or disagree with the following statements: "
                                    "Transgender people should be allowed to "
                                    "use the public restroom of their choice."),
                                    lambda x: {"Strongly agree": "agree",
                                               "Agree": "agree",
                                               "Disagree": "disagree",
                                               "Strongly Disagree": "disagree"}[x]),
                "gay_is_it_choice":(("Please rate the extent to which you agree "
                                     "or disagree with the following "
                                     "statements: People choose to be "
                                     "gay/lesbian."),
                                     lambda x: {"Strongly agree": "agree",
                                                "Agree": "agree",
                                                "Disagree": "disagree",
                                                "Strongly Disagree": "disagree"}[x]),
                "husband_salary":(("Please rate the extent to which you agree "
                                   "or disagree with the following statements: "
                                   "A husband should earn a larger salary than "
                                   "his wife."),
                                   lambda x: {"Strongly agree": "agree",
                                              "Agree": "agree",
                                              "Disagree": "disagree",
                                              "Strongly Disagree": "disagree"}[x]),
                "women_childcare":(("Please rate the extent to which you agree "
                                    "or disagree with the following "
                                    "statements: It is God's will that women "
                                    "care for children."),
                                    lambda x: {"Strongly agree": "agree",
                                               "Agree": "agree",
                                               "Disagree": "disagree",
                                               "Strongly Disagree": "disagree"}[x]),
                "men_suited_politics":(("Please rate the extent to which you "
                                        "agree or disagree with the following "
                                        "statements: Men are better suited "
                                        "emotionally for politics than women."),
                                        lambda x: {"Strongly agree": "agree",
                                                   "Agree": "agree",
                                                   "Disagree": "disagree",
                                                   "Strongly Disagree": "disagree"}[x]),
                "refugees_terrorist_threat":(("Please rate the extent to which "
                                              "you agree or disagree with the "
                                              "following statements: Refugees "
                                              "from the Middle East pose a "
                                              "terrorist threat to the United "
                                              "States."),
                                              lambda x: {"Strongly agree": "agree",
                                                         "Agree": "agree",
                                                         "Disagree": "disagree",
                                                         "Strongly Disagree": "disagree"}[x]),
                "mexican_immigrants_criminals":(("Please rate the extent to "
                                                 "which you agree or disagree "
                                                 "with the following "
                                                 "statements: Illegal "
                                                 "immigrants from Mexico are "
                                                 "mostly dangerous criminals."),
                                                 lambda x: {"Strongly agree": "agree",
                                                            "Agree": "agree",
                                                            "Disagree": "disagree",
                                                            "Strongly Disagree": "disagree"}[x]),
                "life_happiness":(("In general, how happy are you with your "
                                   "life as a whole these days?"),
                                   lambda x: {"Not too happy": "sad",
                                              "Pretty happy": "happy"
                                              "Very happy": "happy"}[x]),
                "depressed_freq":(("In the past WEEK, about how often have you "
                                   "had the following feelings? I felt "
                                   "depressed."),
                                   lambda x: {"Never": "never",
                                              "Hardly ever": "rarely",
                                              "Some of the time": "sometimes"
                                              "Most or all of the time": "frequently"}[x]),
                "days_of_exercise":(("How many DAYS per WEEK do you do exercise "
                                     "for at least 30 minutes?"),
                                     lambda x: x),
                "police_racial_treatment":(("Please rate the extent to which "
                                            "you agree or disagree with the "
                                            "following statements: Police "
                                            "officers in the United States "
                                            "treat blacks the same as whites."),
                                            lambda x: {"Strongly agree": "agree",
                                                       "Agree": "agree",
                                                       "Disagree": "disagree",
                                                       "Strongly Disagree": "disagree"}[x]),
                "racial_violence":(("Please rate the extent to which you agree "
                                    "or disagree with the following statements: "
                                    "Police officers in the United States shoot "
                                    "blacks more often because they are more "
                                    "violent than whites."),
                                    lambda x: {"Strongly agree": "agree",
                                               "Agree": "agree",
                                               "Disagree": "disagree",
                                               "Strongly Disagree": "disagree"}[x]),
                "bible_beliefs":(("I think the Bible is"),
                                  lambda x: {"The Bible means exactly what it says. It should be "
                                             "taken literally, word-for-word, on all subjects.": "literal",
                                             "The Bible is perfectly true, but it should not be taken literally, word-for-word. We must interpret its meaning.": "true but not literal",
                                             "The Bible contains some human error.": "flawed",
                                             "The Bible is an ancient book of history and legends.": "legend"}[x]),
                "god_beliefs":(("I think the existence of God is"),
                                lambda x: {"I have no doubts that God exists": "real",
                                           "I believe in God, but with some doubts": "real",
                                           "I sometimes believe in God": "possible",
                                           "I believe in a higher power of cosmic force": "complicated, not simple",
                                           "I don't know and there is no way to find out": "unknown",
                                           "I do not believe in God": "false",
                                           "I have no opinion": "unimportant"}[x]),
                "god_concern_for_world":(("Based on your personal understanding "
                                          "of God, please rate the extent to "
                                          "which you agree or disagree with "
                                          "the following statements: God is "
                                          "concerned with the well-being of the "
                                          "world."),
                                          lambda x: {"Strongly agree": "agree",
                                                     "Agree": "agree",
                                                     "Disagree": "disagree",
                                                     "Strongly Disagree": "disagree"}[x]),
                "god_concern_for_individuals":(("Based on your personal "
                                                "understanding of God, please "
                                                "rate the extent to which you "
                                                "agree or disagree with the "
                                                "following statements: God is "
                                                "concerned with my personal "
                                                "well-being."),
                                                lambda x: {"Strongly agree": "agree",
                                                           "Agree": "agree",
                                                           "Disagree": "disagree",
                                                           "Strongly Disagree": "disagree"}[x]),
                "church_attendance":(("How often do you attend religious "
                                      "services at a place of worship?"),
                                      lambda x: {"Never": "never",
                                                 "Less than once a year": "rarely",
                                                 "Once or twice a year": "annually",
                                                 "Several times a year": "sometimes",
                                                 "Once a month": "monthly",
                                                 "2 to 3 times a month": "biweekly",
                                                 "About once a week": "weekly",
                                                 "Several times a week": "frequently"}[x]),
                "prayer_in_school":(("Please rate the extent to which you agree "
                                     "or disagree with the following "
                                     "statements: The federal government "
                                     "should allow prayer in public schools."),
                                     lambda x: {"Strongly agree": "agree",
                                                "Agree": "agree",
                                                "Disagree": "disagree",
                                                "Strongly Disagree": "disagree"}[x]),
                "gods_plan":(("Please rate the extent to which you agree or "
                              "disagree with the following statements: When "
                              "good or bad things happen to me, I see it as "
                              "part of God's plan for me."),
                              lambda x: {"Strongly agree": "agree",
                                         "Agree": "agree",
                                         "Disagree": "disagree",
                                         "Strongly Disagree": "disagree"}[x]),
