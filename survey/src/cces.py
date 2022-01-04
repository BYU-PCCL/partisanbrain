from pdb import set_trace as bp
import pandas as pd
from dataset import Dataset
from dataset import PromptSpecs


class CCESDataset(Dataset):

    def __init__(self):
        survey_fname = "data/CES20_Common_OUTPUT_vv.csv"
        super().__init__(survey_fname)

    def _get_dv_filter_funcs(self):
        return {
                    "CC20_333a"  : lambda x: x, 
                    "CC20_333b"  : lambda x: x, 
                    "CC20_333c"  : lambda x: x, 
                    "CC20_305_9" : lambda x: x, 
                    "CC20_307"   : lambda x: x, 
                    "CC20_334b"  : lambda x: x, 
                    "CC20_334c"  : lambda x: x, 
                    "CC20_334d"  : lambda x: x, 
                    "CC20_302"   : lambda x: x, 
                    "CC20_303"   : lambda x: x, 
                    "union"      : lambda x: x, 
                    "trans"      : lambda x: x, 
                    "CC20_331a"  : lambda x: x, 
                    "CC20_331b"  : lambda x: x, 
                    "CC20_331c"  : lambda x: x, 
                    "CC20_331d"  : lambda x: x, 
                    "CC20_441g"  : lambda x: x, 
                    "CC20_441b"  : lambda x: x, 
                    "CC20_441e"  : lambda x: x, 
        }

    def _filter_demographics(self, df):
        df = df[~df["party"].isin([9,13])]
        df = df[~df["race"].isin([7])]
        df = df[~df["ideo"].isin([8])]
        df = df[~df["income"].isin([97])]
        df = df[~df["party"].isin([4])]
        new_df = df[~df["religion"].isin([12])]
        return new_df

    def _filter_to_usa(self, df):
        """Return a new dictionary where all respondents are from USA"""
        return df

    def _get_dv_col_names(self):
        return {
                    "CC20_333a"  : "CC20_333a",
                    "CC20_333b"  : "CC20_333b",
                    "CC20_333c"  : "CC20_333c",
                    "CC20_305_9" : "CC20_305_9",
                    "CC20_307"   : "CC20_307",
                    "CC20_334b"  : "CC20_334b",
                    "CC20_334c"  : "CC20_334c",
                    "CC20_334d"  : "CC20_334d",
                    "CC20_302"   : "CC20_302",
                    "CC20_303"   : "CC20_303",
                    "union"      : "union",
                    "trans"      : "trans",
                    "CC20_331a"  : "CC20_331a",
                    "CC20_331b"  : "CC20_331b",
                    "CC20_331c"  : "CC20_331c",
                    "CC20_331d"  : "CC20_331d",
                    "CC20_441g"  : "CC20_441g",
                    "CC20_441b"  : "CC20_441b",
                    "CC20_441e"  : "CC20_441e",
        }

    def _get_demographic_col_names(self):
        return {"birthyr"     : "age",
                "gender"      : "gender",
                "CL_party"    : "party",
                "educ"        : "educ",
                "CC20_340a"   : "ideo",
                "faminc_new"  : "income",
                "religpew"    : "religion",
                "race"        : "race",
                "region_post" : "census_reg",
                "marstat"     : "marital"}

    def _make_backstory(self, row):

        code_dic = {
            "age": { k:f"I am {2020 - int(k)} years old." for k in range(1900,2020)},
            "gender": {
                1:  "I am male.",
                2:  "I am female.",
                },
            "party": {
                1:  "I am a member of the Conservative Party.",
                2:  "I am a member of the Constitution Party.",
                3:  "I am a member of the Democratic Party.",
                5:  "I am a member of the Green Party.",
                6:  "I am an Independent, politically speaking.",
                7:  "I am a member of the Libertarian Party.",
                8:  "I am not a member of any political party.",
                10: "I am a member of the Reform Party.",
                11: "I am a member of the Republican Party.",
                12: "I am a member of the Socialist Party.",
                14: "I am a member of the Working Families Party.",
                },
            "educ": {
                1:  "I didn't graduate from high school.",
                2:  "I graduated from high school.",
                3:  "I went to some college.",
                4:  "I got my associate's degree.",
                5:  "I got my bachelor's degree.",
                6:  "I went to graduate school.",
                },
            "ideo": {
                1:  "Ideologically, I am very liberal.",
                2:  "Ideologically, I am liberal.",
                3:  "Ideologically, I am somewhat liberal.",
                4:  "Ideologically, I am middle of the road.",
                5:  "Ideologically, I am somewhat conservative.",
                6:  "Ideologically, I am conservative.",
                7:  "Ideologically, I am very conservative.",
                },
            "income": {
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
            "religion": {
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
                1:  "I am White.",
                2:  "I am Black.",
                3:  "I am Hispanic.",
                4:  "I am Asian.",
                5:  "I am Native American.",
                6:  "I am bi-racial.",
                8:  "I am Middle Eastern.",
                },                                 
            "census_reg": {
                1:  "I live in the northeast of the United States.",
                2:  "I live in the Midwest.",
                3:  "I live in the South.",
                4:  "I live in the western United States.",
                },
            "marital": {
                1:  "I am married.",
                2:  ("I got married, but I am now "
                     "separated from my partner."),
                3:  "I am divorced.",
                4:  "I am widowed.",
                5:  "I have never been married.",
                6:  "I am in a domestic/civil partnership.",
                },
        }
        date_statement = ["It is fall 2020."]
        backstory = None
        try:
            backstory = " ".join([code_dic[k][row[k]] for k in code_dic] + date_statement)
        except Exception as exc:
            print(exc)
        return backstory

    def _get_col_prompt_specs(self):
        return {
            "CC20_333a": PromptSpecs(
                question="Do you support or oppose giving the Environmental Protection Agency power to regulate carbon dioxide emissions?",
                answer_prefix="I",
                answer_map={
                    1: "support",
                    2: "oppose",
                    }),
            "CC20_333b": PromptSpecs(
                question="Do you support or oppose requiring that each state use a minimum amount of renewable fuels (wind, solar, and hydroelectric) in the generation of electricity even if electricity prices increase a little?",
                answer_prefix="I",
                answer_map={
                    1: "support",
                    2: "oppose",
                    }),
            "CC20_333c": PromptSpecs(
                question="Do you support or oppose strengthening the Environmental Protection Agency enforcement of the Clean Air Act and Clean Water Act even if it costs U.S. jobs",
                answer_prefix="I",
                answer_map={
                    1: "support",
                    2: "oppose",
                    }),
            "CC20_305_9": PromptSpecs(
                question="Over the past year, have you been a victim of a crime?",
                answer_prefix="",
                answer_map={
                    1 : "yes",
                    2 : "no",
                    }),
            "CC20_307": PromptSpecs(
                question="Do the police make you feel safe or unsafe?",
                answer_prefix="the police make me feel",
                answer_map={
                    1 : "safe",
                    2 : "safe",
                    3 : "unsafe",
                    4 : "unsafe",
                    }),
            "CC20_334b": PromptSpecs(
                question="Do you support or oppose requiring police officers to wear body cameras that record all of their activities while on duty?",
                answer_prefix="I",
                answer_map={
                    1: "support",
                    2: "oppose",
                    }),
            "CC20_334c": PromptSpecs(
                question="Do you support or oppose increasing the number of police on the street by 10 percent, even if it means fewer funds for other public services?",
                answer_prefix="I",
                answer_map={
                    1 : "support",
                    2 : "oppose",
                    }),
            "CC20_334d": PromptSpecs(
                question="Do you support or oppose decreasing the number of police on the street by 10 percent and increasing funding for other public services?",
                answer_prefix="I",
                answer_map={
                    1 : "support",
                    2 : "oppose",
                    }),
            "CC20_302": PromptSpecs(
                question="Would you say that OVER THE PAST YEAR the nation's economy has improved, stayed the same, or worsened?",
                answer_prefix="over the past year the nation's economy has",
                answer_map={
                    1 : "improved",
                    2 : "improved",
                    3 : "stayed the same",
                    4 : "worsened",
                    5 : "worsened",
                    }),
            "CC20_303": PromptSpecs(
                question="OVER THE PAST YEAR, would you say that your household's income has increased, stayed the same, or decreased?",
                answer_prefix="over the past year, my household's income has",
                answer_map={
                    1 : "increased",
                    2 : "increased",
                    3 : "stayed the same",
                    4 : "decreased",
                    5 : "decreased",
                    }),
            "union": PromptSpecs(
                question="Are you a member of a labor union?",
                answer_prefix="I",
                answer_map={
                    1 : "do belong to a union",
                    2 : "did belong to a union and no longer do",
                    3 : "don't belong to a union",
                    }),
            "trans": PromptSpecs(
                question="Have you ever undergone any part of a process (including any thought or action) to change your gender / perceived gender from the one you were assigned at birth? This may include steps such as changing the type of clothes you wear, name you are known by or undergoing surgery.",
                answer_prefix="",
                answer_map={
                    1 : "yes",
                    2 : "no",
                    }),
            "CC20_331a": PromptSpecs(
                question="Do you support or oppose granting legal status to all illegal immigrants who have held jobs and paid taxes for at least 3 years, and not been convicted of any felony crimes?",
                answer_prefix="I",
                answer_map={
                    1 : "support",
                    2 : "oppose",
                    }),
            "CC20_331b": PromptSpecs(
                question="Do you support or oppose increasing the number of border patrols on the US-Mexican border?",
                answer_prefix="I",
                answer_map={
                    1 : "support",
                    2 : "oppose",
                    }),
            "CC20_331c": PromptSpecs(
                question="Do you support or oppose increasing the number of border patrols on the US-Mexican border?",
                answer_prefix="I",
                answer_map={
                    1 : "support",
                    2 : "oppose",
                    }),
            "CC20_331d": PromptSpecs(
                question="Do you support or oppose reducing legal immigration by 50 percent over the next 10 years by eliminating the visa lottery and ending family-based migration?",
                answer_prefix="I",
                answer_map={
                    1 : "support",
                    2 : "oppose",
                    }),
            "CC20_441g": PromptSpecs(
                question="Do you agree or disagree that whites do not go to great lengths to understand the problems African Americans face?",
                answer_prefix="I",
                answer_map={
                    1 : "agree",
                    2 : "agree",
                    3 : "neither agree nor disagree",
                    4 : "disagree",
                    5 : "disagree",
                    }),
            "CC20_441b": PromptSpecs(
                question="Do you agree or disagree that generations of slavery and discrimination have created conditions that make it difficult for blacks to work their way out of the lower class?",
                answer_prefix="I",
                answer_map={
                    1 : "agree",
                    2 : "agree",
                    3 : "neither agree nor disagree",
                    4 : "disagree",
                    5 : "disagree",
                    }),
            "CC20_441e": PromptSpecs(
                question="Do you agree or disagree with the statement 'I resent when Whites deny the existence of racial discrimination.'?",
                answer_prefix="I",
                answer_map={
                    1 : "agree",
                    2 : "agree",
                    3 : "neither agree nor disagree",
                    4 : "disagree",
                    5 : "disagree",
                    }),
        }

if __name__ == '__main__':
    ds = CCESDataset()
    backstories = ds.get_backstories_all_demos()
    for backstory in backstories:
        print(f"{backstory[0]}\n\n{backstory[1]}\n\n")
    prompts = ds.get_prompts_sample()
    for prompt in prompts:
        print(f"{prompt}\n\n")
