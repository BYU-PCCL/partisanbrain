from pdb import set_trace as bp
import pandas as pd
from dataset import Dataset


class CCESDataset(Dataset):

    def __init__(self, n_exemplars):
        survey_fname = "data/cces/CES20_Common_OUTPUT_vv.csv"
        df = pd.read_csv(survey_fname)
        bp()
        super().__init__(survey_fname, n_exemplars)

    def _format(self, df):

        # Reduce rows (e.g., down to US)

        # Dropping all but relevant columns
        new_df = df[[
                    "birthyr",
                    "gender",
                    "CL_party",
                    "educ",
                    "CC20_340a",
                    "faminc_new",
                    "religpew",
                    "race",
                    "region_post",
                    "marstat",
        """

If asked to choose either “[OPTIONAL-PREFIX + Option 1]" or
 “[OPTIONAL-PREFIX + Option 2]" in response to the question, "[QUESTION]?" I'd choose "


def _get_col_prompt_specs(self):
        return {"econ_today": PromptSpecs(question=("How would you rate "
                                                    "economic conditions "
                                                    "in this country today?"),
                                          answer_prefix="conditions are",
                                          answer_map={"Excellent": "excellent",
                                                      "Good": "good",
                                                      "Only fair": "fair",
                                                      "Poor": "poor"})} (edited) 
￼
                         
                         """



            
            "Gender",
                     "Age",
                     "Which character shot first?",
                     ("Do you consider yourself to be a fan "
                      "of the Star Wars film franchise?")]]

        # Dropping rows with NA values
        new_df = new_df.dropna(axis=0)

        # Renaming columns for convenience
        new_df = new_df.rename({"Gender": "gender",
                                "Age": "age",
                                "Which character shot first?": "shot_first",
                                ("Do you consider yourself to be a fan "
                                 "of the Star Wars film franchise?"): "fan"},
                               axis=1)

        # Removing "I don't understand this question" response
        new_df = new_df.loc[new_df["shot_first"].isin(["Han", "Greedo"])]

        # Randomly sample columns!

        # Get only top 8 rows to keep things simple for testing
        new_df = new_df.head(105)

        return new_df

    def _make_backstory(self, row):
        code_dic = {
            "birthyr": {
                {k:f"I am {2020 - int(k)} years old" for k in range(1900,2020)}
                },
            "gender": {
                1: "I am male.",
                2: "I am female.",
                },
            "CL_party": {
                1:  "I am a member of the Conservative Party.",
                2:  "I am a member of the Constitution Party.",
                3:  "I am a member of the Democratic Party.",
                5:  "I am a member of the Green Party.",
                6:  "I am an Independent.",
                7:  "I am a member of the Libertarian Party.",
                8:  "I am not a member of any political party.",
                10: "I am a member of the Reform Party.",
                11: "I am a member of the Republican Party.",
                12: "I am a member of the Socialist Party.",
                14: "I am a member of the Working Families Party.",
                },
            "educ": {
                1: "I didn't graduate from high school.",
                2: "I graduated from high school.",
                3: "I went to some college.",
                4: "I got my associate's degree.",
                5: "I got my bachelor's degree.",
                6: "I went to graduate school.",
                },
            "CC20_340a": {
                1: "Ideologically, I am very liberal.",
                2: "Ideologically, I am liberal.",
                3: "Ideologically, I am somewhat liberal.",
                4: "Ideologically, I am middle of the road.",
                5: "Ideologically, I am somewhat conservative.",
                6: "Ideologically, I am conservative.",
                7: "Ideologically, I am very conservative.",
                8: "Ideologically, I am not sure.",
                },
            "faminc_new": {
                1 : "Over the last year, my family's income was less than $10,000.",
                2 : "Over the last year, my family's income was $10,000 - $19,999.",
                3 : "Over the last year, my family's income was $20,000 - $29,999.",
                4 : "Over the last year, my family's income was $30,000 - $39,999.",
                5 : "Over the last year, my family's income was $40,000 - $49,999.",
                6 : "Over the last year, my family's income was $50,000 - $59,999.",
                7 : "Over the last year, my family's income was $60,000 - $69,999.",
                8 : "Over the last year, my family's income was $70,000 - $79,999.",
                9 : "Over the last year, my family's income was $80,000 - $99,999.",
                10: "Over the last year, my family's income was $100,000 - $119,999.",
                11: "Over the last year, my family's income was $120,000 - $149,999.",
                12: "Over the last year, my family's income was $150,000 - $199,999.",
                13: "Over the last year, my family's income was $200,000 - $249,999.",
                14: "Over the last year, my family's income was $250,000 - $349,999.",
                15: "Over the last year, my family's income was $350,000 - $499,999.",
                16: "Over the last year, my family's income was $500,000 or more.",
                },
            "religpew": {
                1 : "I am Protestant.",
                2 : "I am Roman Catholic.",
                3 : "I am Mormon.",
                4 : "I am Eastern or Greek Orthodox.",
                5 : "I am Jewish.",
                6 : "I am Muslim.",
                7 : "I am Buddhist.",
                8 : "I am Hindu.",
                9 : "I am atheist.",
                10: "I am agnostic.",
                11: "I am not religious.",
                },       
            "race": {        
                1: "I am White.",
                2: "I am Black.",
                3: "I am Hispanic.",
                4: "I am Asian.",
                5: "I am Native American.",
                6: "I am bi-racial.",
                8: "I am Middle Eastern.",
                },                                 
            "region_post": {
                1: "I live in the Northeast.",
                2: "I live in the Midwest.",
                3: "I live in the South.",
                4: "I live in the West.",
                },
            "marstat": {
                1: "I am married.",
                2: "I am separated.",
                3: "I am divorced.",
                4: "I am widowed.",
                5: "I have never been married.",
                6: "I am in a domestic/civil partnership.",
                },
        }
        return " ".join([code_dic[k][row[k]] for k in code_dic])

    def _get_prompt_instructions(self):
        return {
            "CC20_333a": PromptSpecs(
                question="Do you support or oppose giving the Environmental Protection Agency power to regulate carbon dioxide emissions?",
                answer_prefix="",
                answer_map={
                    "": "",
                    })
        }
        return {"shot_first": (("Between Han and Greedo I think the one "
                                "who shot first was"),
                               lambda x: x),
                "fan": ("When asked if I'm a Star Wars fan I say",
                        lambda x: x.lower())}



if __name__ == '__main__':
    import random

    ds = CCESDataset(n_exemplars=5)