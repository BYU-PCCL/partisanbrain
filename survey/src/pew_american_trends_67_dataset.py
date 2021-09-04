from dataset import Dataset
from dataset import PromptSpecs


class PewAmericanTrendsWave67Dataset(Dataset):

    def __init__(self):
        # This is the desired survey_fname format - "data/name"
        survey_fname = "data/ATP W67.sav"
        # Do not copy samples=500 for other survey subclasses
        # It is unique to the Pew American Trends datasets
        super().__init__(survey_fname,
                         samples=500)

    def _filter_to_usa(self, df):
        # Since everyone in Pew American Trends Wave 67
        # is from the US we don't need to do anything here
        return df

    def _get_demographic_col_names(self):
        return {"F_AGECAT": "age",
                "F_SEX": "gender",
                "F_PARTY_FINAL": "party",
                "F_EDUCCAT2": "educ",
                "F_IDEO": "ideo",
                "F_INCOME": "income",
                "F_RELIG": "religion",
                "F_RACETHN": "race_1",
                "F_CREGION": "census_reg",
                "F_MARITAL": "marital"}

    def _get_dv_col_names(self):
        return {"ENV2_c_W67": "coal",
                "ENV2_d_W67": "solar",
                "ENV2_f_W67": "wind",
                "CCPOLICY_a_W67": "trees",
                "ENVIR8_e_W67": "gov_climate",
                "EN7_W67": "human_act_climate",
                "CLIM9_W67": "climate_local",
                "RQ1_F1A_W67": "med_researcher_view",
                "PQ1_F2A_W67": "med_doc_view",
                "CLIN_TRIAL1_W67": "clin_trial"}

    def _filter_demographics(self, df):
        new_df = df[df["party"].isin(["Democrat",
                                      "Republican",
                                      "Independent"])]
        new_df = new_df[new_df["religion"] != "Other"]
        new_df = new_df[new_df["race_1"] != "Other"]

        for col_name in list(new_df):
            new_df = new_df[new_df[col_name] != "Refused"]

        print(new_df["race_1"].value_counts())

        return new_df

    def _make_backstory(self, row):
        backstory = []

        # Age
        age_map = {
            "18-29": "I am between 18 and 29 years old.",
            "30-49": "I am between 30 and 49 years old.",
            "50-64": "I am between 50 and 64 years old.",
            "65+": "I am at least 65 years old."
        }
        backstory.append(age_map[row["age"]])

        # Gender
        gender_map = {
            "Female": "I am female.",
            "Male": "I am male."
        }
        backstory.append(gender_map[row["gender"]])

        # Party
        pfx = "In terms of political parties, I am"
        party_map = {
            "Democrat": f"{pfx} a Democrat.",
            "Republican": f"{pfx} a Republican.",
            "Independent": f"{pfx} independent."
        }
        backstory.append(party_map[row["party"]])

        # Education
        pfx = "In terms of educational attainment,"
        educ_map = {
            "Less than high school": (f"{pfx} I have less than a high "
                                      "school education."),
            "High school graduate": (f"{pfx} I am a high school graduate."),
            "Some college, no degree": (f"{pfx} I have completed some "
                                        "college but have not earned a "
                                        "college degree."),
            "Associate's degree": (f"{pfx} I have earned an "
                                   "Associate's degree."),
            "College graduate/some post grad": (f"{pfx} I have graduated "
                                                "from college."),
            "Postgraduate": (f"{pfx} I have earned a postgraduate degree.")
        }
        backstory.append(educ_map[row["educ"]])

        # Ideology
        pfx = "In terms of political ideology,"
        ideo_map = {
            "Conservative": f"{pfx} I am conservative.",
            "Liberal": f"{pfx} I am liberal.",
            "Moderate": f"{pfx} I am moderate.",
            "Very conservative": f"{pfx} I am very conservative.",
            "Very liberal": f"{pfx} I am very liberal."
        }
        backstory.append(ideo_map[row["ideo"]])

        # Income
        pfx = "My annual family income is"
        income_map = {
            "Less than $10,000": f"{pfx} less than $10,000.",
            "$10,000 to less than $20,000": (f"{pfx} between $10,000 "
                                             "and $20,000."),
            "$20,000 to less than $30,000": (f"{pfx} between $20,000 "
                                             "and $30,000."),
            "$30,000 to less than $40,000": (f"{pfx} between $30,000 "
                                             "and $40,000."),
            "$40,000 to less than $50,000": (f"{pfx} between $40,000 "
                                             "and $50,000."),
            "$50,000 to less than $75,000": (f"{pfx} between $50,000 "
                                             "and $75,000."),
            "$75,000 to less than $100,000": (f"{pfx} between $75,000 "
                                              "and $100,000."),
            "$100,000 to less than $150,000": (f"{pfx} between $100,000 "
                                               "and $150,000."),
            "$150,000 or more": f"{pfx} $150,000 or more."
        }
        backstory.append(income_map[row["income"]])

        # Religiosity
        pfx = "In terms of religion I am"
        religion_map = {
            "Roman Catholic": f"{pfx} Roman Catholic.",
            "Protestant": f"{pfx} Protestant.",
            "Buddhist": f"{pfx} Buddhist.",
            "Atheist": f"{pfx} atheist.",
            "Muslim": f"{pfx} Muslim.",
            "Hindu": f"{pfx} Hindu.",
            "Jewish": f"{pfx} Jewish.",
            ("Orthodox (such as Greek, Russian, "
             "or some other Orthodox church)"): f"{pfx} Orthodox.",
            ("Mormon (Church of Jesus Christ of "
             "Latter-day Saints or LDS)"): f"{pfx} Mormon.",
            "Agnostic": f"{pfx} agnostic.",
            "Nothing in particular": ("I do not identify with any "
                                      "religion in particular.")
        }
        backstory.append(religion_map[row["religion"]])

        # Race/Ethnicity
        pfx = "I am"
        race_map = {
            "White non-Hispanic": f"{pfx} White.",
            "Hispanic": f"{pfx} Hispanic.",
            "Black non-Hispanic": f"{pfx} Black.",
        }
        backstory.append(race_map[row["race_1"]])

        # Region
        pfx = "I live in the"
        region_map = {
            "Midwest": f"{pfx} Midwest.",
            "West": f"{pfx} western United States.",
            "Northeast": f"{pfx} northeast of the United States.",
            "South": f"{pfx} South."
        }
        backstory.append(region_map[row["census_reg"]])

        # Marital Status
        marital_map = {
            "Widowed": "I am widowed.",
            "Never been married": "I have never been married.",
            "Married": "I am married.",
            "Divorced": "I am divorced.",
            "Living with a partner": "I am living with a partner.",
            "Separated": ("I got married, but I am now "
                          "separated from my partner.")
        }
        backstory.append(marital_map[row["marital"]])

        # Date
        backstory.append("It is spring 2020.")

        return " ".join(backstory)

    def _get_col_prompt_specs(self):
        return {"coal": PromptSpecs(("Do you support or oppose "
                                     "expansion of coal mining "
                                     "in the United States?"),
                                    "",
                                    {"Favor": "support",
                                     "Oppose": "oppose"}),
                "solar": PromptSpecs(("Do you support or oppose "
                                      "expansion of solar power "
                                      "\"farms\" "
                                      "in the United States?"),
                                     "",
                                     {"Favor": "support",
                                      "Oppose": "oppose"}),
                "wind": PromptSpecs(("Do you support or oppose "
                                     "expansion of wind turbine "
                                     "\"farms\" "
                                     "in the United States?"),
                                    "",
                                    {"Favor": "support",
                                     "Oppose": "oppose"}),
                "trees": PromptSpecs(("Do you support or oppose "
                                      "planting about a trillion "
                                      "trees around the world to "
                                      "absorb carbon emissions "
                                      "in the atmosphere?"),
                                     "",
                                     {"Favor": "support",
                                      "Oppose": "oppose"}),
                "gov_climate": PromptSpecs(("Do you think what "
                                            "the federal government is "
                                            "doing to reduce the effects "
                                            "of global climate change is "
                                            "excessive, sufficient, or "
                                            "insufficient?"),
                                           ("I think the federal government's "
                                            "efforts are"),
                                           {"Too much": "excessive",
                                            "Too little": "insufficient",
                                            "About the right amount":
                                            "sufficient"}),
                "human_act_climate": PromptSpecs(("Do you think human "
                                                  "activity, such as the "
                                                  "burning of fossil "
                                                  "fuels, contributes at "
                                                  "least some "
                                                  "to global climate change?"),
                                                 "",
                                                 {"A great deal": "yes",
                                                  "Some": "yes",
                                                  "Not too much": "no",
                                                  "Not at all": "no"}),
                "climate_local": PromptSpecs(("Do you think global climate "
                                              "change is currently "
                                              "having some affect "
                                              "on your local community?"),
                                             "",
                                             {"A great deal": "yes",
                                              "Some": "yes",
                                              "Not too much": "no",
                                              "Not at all": "no"}),
                "med_researcher_view": PromptSpecs(("Medical research "
                                                    "scientists conduct "
                                                    "research to "
                                                    "investigate human "
                                                    "diseases, and test "
                                                    "methods to prevent "
                                                    "and treat them. In "
                                                    "general, would you "
                                                    "say your view of "
                                                    "medical research "
                                                    "scientists is positive, "
                                                    "negative, or neutral?"),
                                                   "",
                                                   {"Mostly positive":
                                                    "positive",
                                                    "Mostly negative":
                                                    "negative",
                                                    ("Neither positive "
                                                     "nor negative"):
                                                    "neutral"}),
                "med_doc_view": PromptSpecs(("Medical doctors provide "
                                             "patients with diagnoses of "
                                             "disease and/or treatment "
                                             "recommendations to promote, "
                                             "maintain or restore a "
                                             "patient’s health. "
                                             "In general, would you say "
                                             "your view of medical doctors "
                                             "is positive, negative, or "
                                             "neutral?"),
                                            "",
                                            {"Mostly positive": "positive",
                                             "Mostly negative": "negative",
                                             ("Neither positive "
                                              "nor negative"):
                                             "neutral"}),
                "clin_trial": PromptSpecs(("Some medical research studies "
                                           "are called clinical trials in "
                                           "which volunteers participate in "
                                           "a study to help test the safety "
                                           "and effectiveness of new "
                                           "treatments, drugs or devices. "
                                           "Do you think it is at least "
                                           "somewhat important "
                                           "to go through the process of "
                                           "conducting clinical trials, even "
                                           "if it will lengthen the time it "
                                           "takes to develop new treatments?"),
                                          "",
                                          {"Very important": "yes",
                                           "Somewhat important": "yes",
                                           "Not too important": "no",
                                           "Not at all important": "no"})}


if __name__ == "__main__":
    ds = PewAmericanTrendsWave67Dataset()
    backstories = ds.get_backstories_all_demos()
    for backstory in backstories:
        print(f"{backstory[0]}\n\n{backstory[1]}\n\n")
    prompts = ds.get_prompts_sample()
    for prompt in prompts:
        print(f"{prompt}\n\n")
