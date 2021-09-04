from dataset import Dataset
from dataset import PromptSpecs


class PewAmericanTrendsWave78Dataset(Dataset):

    def __init__(self):
        # This is the desired survey_fname format - "data/name"
        survey_fname = "data/ATP W78.sav"
        # Do not copy samples=500 for other survey subclasses
        # It is unique to the Pew American Trends datasets
        super().__init__(survey_fname,
                         samples=500)

    def _filter_to_usa(self, df):
        # Since everyone in Pew American Trends Wave 78
        # is from the US we don't need to do anything here
        return df

    def _get_demographic_col_names(self):
        return {"F_AGECAT": "age",
                "F_GENDER": "gender",
                "F_PARTY_FINAL": "party",
                "F_EDUCCAT2": "educ",
                "F_IDEO": "ideo",
                "F_INC_SDT1": "income",
                "F_RELIG": "religion",
                "F_RACETHNMOD": "race",
                "F_CREGION": "census_reg",
                "F_MARITAL": "marital"}

    def _get_dv_col_names(self):
        return {"ECON1_W78": "econ_today",
                "ECON1B_W78": "econ_year_away",
                "SATIS_W78": "country_satisfied",
                "VTADMIN_POST_US_W78": "election_wellness",
                "ELECTNTFOL_W78": "follow_election",
                "COVID_2ASSISTLD_W78": "covid_assist_pack",
                "POL12_W78": "rep_dem_relationship",
                "COVID_OPENMORE_W78": "covid_restrict",
                "DIVISIONSCONC_W78": "rep_dem_division",
                "VOTELIST_US_W78": "more_votes_better"}

    def _filter_demographics(self, df):
        new_df = df[df["party"].isin(["Democrat",
                                      "Republican",
                                      "Independent"])]
        new_df = new_df[new_df["religion"] != "Other"]
        new_df = new_df[new_df["race"] != "Other"]

        for col_name in list(new_df):
            new_df = new_df[new_df[col_name] != "Refused"]
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
            "A woman": "I am a woman.",
            "A man": "I am a man.",
            "In some other way": "I do not identify as a man or a woman."
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
                                   "associate's degree."),
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
            "Less than $30,000": f"{pfx} less than $30,000.",
            "$30,000 to less than $40,000": (f"{pfx} between $30,000 "
                                             "and $40,000."),
            "$40,000 to less than $50,000": (f"{pfx} between $40,000 "
                                             "and $50,000."),
            "$50,000 to less than $60,000": (f"{pfx} between $50,000 "
                                             "and $60,000."),
            "$60,000 to less than $70,000": (f"{pfx} between $60,000 "
                                             "and $70,000."),
            "$70,000 to less than $80,000": (f"{pfx} between $70,000 "
                                             "and $80,000."),
            "$80,000 to less than $90,000": (f"{pfx} between $80,000 "
                                             "and $90,000."),
            "$90,000 to less than $100,000": (f"{pfx} between $90,000 "
                                              "and $100,000."),
            "$100,000 or more": f"{pfx} $100,000 or more."
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
            "Asian non-Hispanic": f"{pfx} Asian.",
            "Hispanic": f"{pfx} Hispanic.",
            "Black non-Hispanic": f"{pfx} Black."
        }
        backstory.append(race_map[row["race"]])

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
        backstory.append("It is November 2020.")

        return " ".join(backstory)

    def _get_col_prompt_specs(self):
        return {"econ_today": PromptSpecs(question=("How would you rate "
                                                    "economic conditions "
                                                    "in this country today?"),
                                          answer_prefix="conditions are",
                                          answer_map={"Excellent": "excellent",
                                                      "Good": "good",
                                                      "Only fair": "fair",
                                                      "Poor": "poor"}),
                "econ_year_away": PromptSpecs(("How do you expect economic "
                                               "conditions in the United "
                                               "States a year from now will "
                                               "be compared to economic "
                                               "conditions in the United "
                                               "States now?"),
                                              "I think conditions will be",
                                              {"Better": "better",
                                               "Worse": "worse",
                                               "About the same as now":
                                               "the same as now"}),
                "country_satisfied": PromptSpecs(("All in all, are you "
                                                  "satisfied or dissatisfied "
                                                  "with the way things are "
                                                  "going in the United States "
                                                  "today?"),
                                                 "I am",
                                                 {"Satisfied": "satisfied",
                                                  "Dissatisfied":
                                                  "dissatisfied"}),
                "election_wellness": PromptSpecs(("Do you think elections "
                                                  "this November in the "
                                                  "United States were run "
                                                  "and administered well "
                                                  "or poorly?"),
                                                 ("I think they were run and "
                                                  "administered"),
                                                 {"Very well": "well",
                                                  "Somewhat well": "well",
                                                  "Not too well": "poorly",
                                                  "Not at all well":
                                                  "poorly"}),
                "follow_election": PromptSpecs(("Did you follow the results "
                                                "of the presidential election "
                                                "after polls closed on "
                                                "Election Day?"),
                                               "",
                                               {("Followed them almost "
                                                 "constantly"): "yes",
                                                ("Checked in fairly "
                                                 "often"): "yes",
                                                ("Checked in "
                                                 "occasionally"): "yes",
                                                ("Tuned them out "
                                                 "entirely"): "no"}),
                "covid_assist_pack": PromptSpecs(("Congress and President "
                                                  "Trump passed a $2 trillion "
                                                  "economic assistance "
                                                  "package in March in "
                                                  "response to the "
                                                  "economic impact of the "
                                                  "coronavirus outbreak. Do "
                                                  "you think another economic "
                                                  "assistance package is "
                                                  "necessary?"),
                                                 "",
                                                 {"Necessary": "yes",
                                                  "Not necessary": "no"}),
                "rep_dem_relationship": PromptSpecs(("Do you think relations "
                                                     "between Republicans and "
                                                     "Democrats in Washington "
                                                     "a year from now will "
                                                     "be better, worse, or "
                                                     "the same as now?"),
                                                    "I think they will be",
                                                    {"Get better": "better",
                                                     "Get worse": "worse",
                                                     ("Stay about "
                                                      "the same"):
                                                        "the same as now"}),
                "covid_restrict": PromptSpecs(("Thinking about restrictions "
                                               "on public activity because "
                                               "of the coronavirus outbreak "
                                               "in your area, do you think "
                                               "the number of restrictions "
                                               "should be increased, "
                                               "decreased, or "
                                               "maintained?"),
                                              ("I think the number of "
                                               "restrictions should be"),
                                              {("MORE restrictions "
                                                "right now"): "increased",
                                               ("FEWER restrictions "
                                                "right now"): "decreased",
                                               ("About the same number "
                                                "of restrictions "
                                                "right now"): "maintained"}),
                "rep_dem_division": PromptSpecs(("Are you at least somewhat "
                                                 "concerned about divisions "
                                                 "between Republicans and "
                                                 "Democrats?"),
                                                "",
                                                {"Very concerned": "yes",
                                                 "Somewhat concerned": "yes",
                                                 "Not too concerned": "no",
                                                 ("Not at all "
                                                  "concerned"): "no"}),
                "more_votes_better": PromptSpecs(("Do you think the United "
                                                  "States would be better "
                                                  "off if more Americans "
                                                  "voted?"),
                                                 "",
                                                 {("The country would not be "
                                                   "better off if more "
                                                   "Americans voted"): "yes",
                                                  ("The country would be "
                                                   "better off if more "
                                                   "Americans "
                                                   "voted"): "no"})}


if __name__ == "__main__":
    ds = PewAmericanTrendsWave78Dataset()
    backstories = ds.get_backstories_all_demos()
    for backstory in backstories:
        print(f"{backstory[0]}\n\n{backstory[1]}\n\n")
    prompts = ds.get_prompts_sample()
    for prompt in prompts:
        print(f"{prompt}\n\n")
