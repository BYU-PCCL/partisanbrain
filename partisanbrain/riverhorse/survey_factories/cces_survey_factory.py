"""
Author: Alex Shaw
Email: alexgshaw64@gmail.com
"""

from ..dataset_factory import DatasetFactory
from ..surveys.cces_survey import CcesSurvey


class CcesFactory(DatasetFactory):
    def __init__(self, survey_obj, sample_seed=0, n=None):
        self.code_dict = {
            "age": {k: f"I am {2020 - int(k)} years old." for k in range(1900, 2020)},
            "gender": {
                1: "I am male.",
                2: "I am female.",
            },
            "party": {
                1: "I am a member of the Conservative Party.",
                2: "I am a member of the Constitution Party.",
                3: "I am a member of the Democratic Party.",
                5: "I am a member of the Green Party.",
                6: "I am an Independent, politically speaking.",
                7: "I am a member of the Libertarian Party.",
                8: "I am not a member of any political party.",
                10: "I am a member of the Reform Party.",
                11: "I am a member of the Republican Party.",
                12: "I am a member of the Socialist Party.",
                14: "I am a member of the Working Families Party.",
            },
            "education": {
                1: "I didn't graduate from high school.",
                2: "I graduated from high school.",
                3: "I went to some college.",
                4: "I got my associate's degree.",
                5: "I got my bachelor's degree.",
                6: "I went to graduate school.",
            },
            "ideology": {
                1: "Ideologically, I am very liberal.",
                2: "Ideologically, I am liberal.",
                3: "Ideologically, I am somewhat liberal.",
                4: "Ideologically, I am middle of the road.",
                5: "Ideologically, I am somewhat conservative.",
                6: "Ideologically, I am conservative.",
                7: "Ideologically, I am very conservative.",
            },
            "income": {
                1: "Over the last year, my family's income was less than $10,000.",
                2: "Over the last year, my family's income was $10,000 - $19,999.",
                3: "Over the last year, my family's income was $20,000 - $29,999.",
                4: "Over the last year, my family's income was $30,000 - $39,999.",
                5: "Over the last year, my family's income was $40,000 - $49,999.",
                6: "Over the last year, my family's income was $50,000 - $59,999.",
                7: "Over the last year, my family's income was $60,000 - $69,999.",
                8: "Over the last year, my family's income was $70,000 - $79,999.",
                9: "Over the last year, my family's income was $80,000 - $99,999.",
                10: "Over the last year, my family's income was $100,000 - $119,999.",
                11: "Over the last year, my family's income was $120,000 - $149,999.",
                12: "Over the last year, my family's income was $150,000 - $199,999.",
                13: "Over the last year, my family's income was $200,000 - $249,999.",
                14: "Over the last year, my family's income was $250,000 - $349,999.",
                15: "Over the last year, my family's income was $350,000 - $499,999.",
                16: "Over the last year, my family's income was $500,000 or more.",
            },
            "religion": {
                1: "I am Protestant.",
                2: "I am Roman Catholic.",
                3: "I am Mormon.",
                4: "I am Eastern or Greek Orthodox.",
                5: "I am Jewish.",
                6: "I am Muslim.",
                7: "I am Buddhist.",
                8: "I am Hindu.",
                9: "I am atheist.",
                10: "I am agnostic.",
                11: "I am not religious.",
            },
            "race_ethnicity": {
                1: "I am White.",
                2: "I am Black.",
                3: "I am Hispanic.",
                4: "I am Asian.",
                5: "I am Native American.",
                6: "I am bi-racial.",
                8: "I am Middle Eastern.",
            },
            "region": {
                1: "I live in the northeast of the United States.",
                2: "I live in the Midwest.",
                3: "I live in the South.",
                4: "I live in the western United States.",
            },
            "marital_status": {
                1: "I am married.",
                2: "I got married, but I am now separated from my partner.",
                3: "I am divorced.",
                4: "I am widowed.",
                5: "I have never been married.",
                6: "I am in a domestic/civil partnership.",
            },
        }
        super().__init__(survey_obj=survey_obj, sample_seed=sample_seed, n=n)

    def modify_data(self, df):
        return df

    def _make_backstory_paragraph(self, row):
        return (
            "\n".join([self.code_dict[dem][row[dem]] for dem in self.present_dems])
            + "\n\n"
        )

    def _make_backstory_qa(self, row):
        return (
            f"Q: What is your age?\nA: {row['age']}\n\n"
            f"Q: What is your gender?\nA: {row['gender']}\n\n"
            f"Q: What is your political affiliation?\nA: {row['party']}\n\n"
            f"Q: What is your education?\nA: {row['education']}\n\n"
            f"Q: What is your ideology?\nA: {row['ideology']}\n\n"
            f"Q: What is your income?\nA: {row['income']}\n\n"
            f"Q: What is your religion?\nA: {row['religion']}\n\n"
            f"Q: What is your race/ethnicity?\nA: {row['race_ethnicity']}\n\n"
            f"Q: What region of the country are your from?\nA: {row['region']}\n\n"
            f"Q: What is your marital status?\nA: {row['marital_status']}\n\n"
        )

    def _make_backstory_convo(self, row):
        return (
            f"P1: What is your age?\nP2: {row['age']}\n"
            f"P1: What is your gender?\nP2: {row['gender']}\n"
            f"P1: What is your political affiliation?\nP2: {row['party']}\n"
            f"P1: What is your education?\nP2: {row['education']}\n"
            f"P1: What is your ideology?\nP2: {row['ideology']}\n"
            f"P1: What is your income?\nP2: {row['income']}\n"
            f"P1: What is your religion?\nP2: {row['religion']}\n"
            f"P1: What is your race/ethnicity?\nP2: {row['race_ethnicity']}\n"
            f"P1: What region of the country are your from?\nP2: {row['region']}\n"
            f"P1: What is your marital status?\nP2: {row['marital_status']}\n"
        )

    def get_templates(self):
        QA_SHOTS = [
            "Q: What is your favorite color?\nA: Blue\n\n",
            "Q: Do you like music?\nA: Yes\n\n",
            "Q: How is your day going?\nA: Good\n\n",
        ]
        SUPPORT_OPPOSE_SHOTS = [
            "Tacos are the best food: Support\n",
            "Blue is a sad color: Oppose\n",
            "Taller people are smarter: Oppose\n",
        ]
        PROPOSAL_SHOTS = [
            "Proposal: Let's eat at Cheese Cake Factory tonight.\nResponse: Support\n\n",
            "Proposal: Eminem is the best rapper.\nResponse: Oppose\n\n",
            "Proposal: H&M has great clothing.\nResponse: Support\n\n",
        ]
        AGREE_DISAGREE_SHOTS = [
            "Tacos are the best food: Agree\n",
            "Blue is a sad color: Disagree\n",
            "Taller people are smarter: Neither\n",
        ]
        AGREEMENT_SHOTS = [
            "Statement: Cheese Cake Factory is delicious.\nResponse: Agree\n\n",
            "Statement: Eminem is the best rapper.\nResponse: Neither\n\n",
            "Statement: H&M has great clothing.\nResponse: Disagree\n\n",
        ]

        support_oppose_templates = lambda dv: {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {},
            ),
            "qa2": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + QA_SHOTS[0]
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "qa3": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:2])
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "qa4": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:3])
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "conversation1": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P1: Do you support or oppose the following proposal? {self.questions[dv]}.\n"
                    f"P2:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "conversation2": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P2: Ask me any question and I'll answer honestly.\n"
                    f"P1: Do you support or oppose the following proposal? {self.questions[dv]}.\n"
                    f"P2:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "support_oppose1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do I support or oppose the following proposals?\n\n"
                    f"{self.questions[dv]}:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "support_oppose2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do I support or oppose the following proposals?\n\n"
                    + SUPPORT_OPPOSE_SHOTS[0]
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "support_oppose3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do I support or oppose the following proposals?\n\n"
                    + "".join(SUPPORT_OPPOSE_SHOTS[:2])
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "support_oppose4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do I support or oppose the following proposals?\n\n"
                    + "".join(SUPPORT_OPPOSE_SHOTS[:3])
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "finish_sentence1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Proposal: {self.questions[dv]}.\n"
                    f"Choosing between support and oppose for the previous proposal, I"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "finish_sentence2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Proposal: {self.questions[dv]}.\n"
                    f"If I have to support or oppose this proposal I would"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "finish_sentence3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Someone asked me if I support or oppose the following proposal: {self.questions[dv]}."
                    f"I said I"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "finish_sentence4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f'Someone asked me if I support or oppose the following proposal: "{self.questions[dv]}."'
                    f"I said I"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "finish_sentence5": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f'Choosing between "support" or "oppose" for the following proposal: "{self.questions[dv]}."'
                    f'I choose "'
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "proposal1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following proposals.\n\n"
                    f"Proposal: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "proposal2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following proposals.\n\n"
                    + PROPOSAL_SHOTS[0]
                    + f"Proposal: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "proposal3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following proposals.\n\n"
                    + "".join(PROPOSAL_SHOTS[:2])
                    + f"Proposal: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "proposal4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following proposals.\n\n"
                    + "".join(PROPOSAL_SHOTS[:3])
                    + f"Proposal: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "yes_no1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f'Q: Do you support this proposal: "{self.questions[dv]}."\n'
                    f"A:"
                ),
                {
                    "Support": "yes",
                    "Oppose": "no",
                },
            ),
        }

        agreement_templates = lambda dv: {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "qa2": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + QA_SHOTS[0]
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "qa3": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:2])
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "qa4": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:3])
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"A:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "conversation1": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"P2:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "conversation2": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P2: Ask me any question and I'll answer honestly.\n"
                    f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"P2:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "agree_disagree1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do you agree or disagree with the following statements (or neither)?\n\n"
                    f"{self.questions[dv]}:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "agree_disagree2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do you agree or disagree with the following statements (or neither)?\n\n"
                    + AGREE_DISAGREE_SHOTS[0]
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "agree_disagree3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do you agree or disagree with the following statements (or neither)?\n\n"
                    + "".join(AGREE_DISAGREE_SHOTS[:2])
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "agree_disagree4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Do you agree or disagree with the following statements (or neither)?\n\n"
                    + "".join(AGREE_DISAGREE_SHOTS[:3])
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "finish_sentence1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Statement: {self.questions[dv]}.\n"
                    f'Choosing between "agree", "disagree", or "neither" for the previous statement, I'
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "finish_sentence2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Statement: {self.questions[dv]}.\n"
                    f"If I have to agree, disagree, or neither with this statement I would"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "finish_sentence3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f'Choosing "agree", "disagree", or "neither" for the following statement: "{self.questions[dv]}."'
                    f'I choose "'
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "statement1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    f"Statement: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "statement2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following statements.\n\n"
                    + AGREEMENT_SHOTS[0]
                    + f"Statement: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "statement3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:2])
                    + f"Statement: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "statement4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will now respond support or oppose to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:3])
                    + f"Statement: {self.questions[dv]}.\n"
                    f"Response:"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "qa_mapping1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + QA_SHOTS[0]
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"Answer (agree, disagree, neither):"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "qa_mapping2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + "".join(QA_SHOTS[:2])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"Answer (agree, disagree, neither):"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
            "qa_mapping3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + "".join(QA_SHOTS[:3])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}.\n"
                    f"Answer (agree, disagree, neither):"
                ),
                {
                    "Strongly agree": "agree",
                    "Somewhat agree": "agree",
                    "Neither agree nor disagree": "neither",
                    "Somewhat disagree": "disagree",
                    "Strongly disagree": "disagree",
                },
            ),
        }

        return {
            "co2_emissions": support_oppose_templates("co2_emissions"),
            "renewable_fuels": support_oppose_templates("renewable_fuels"),
            "clean_air": support_oppose_templates("clean_air"),
            "crime_victim": {},
            "police_feel": {},
            "body_cameras": support_oppose_templates("body_cameras"),
            "increase_police": support_oppose_templates("increase_police"),
            "decrease_police": support_oppose_templates("decrease_police"),
            "nations_economy": {},
            "income_change": {},
            "labor_union": {},
            "gender_change": {},
            "sexuality": {},
            "illegal_immigrants": support_oppose_templates("illegal_immigrants"),
            "border_patrols": support_oppose_templates("border_patrols"),
            "withhold_police_funds": support_oppose_templates("withhold_police_funds"),
            "reduce_immigration": support_oppose_templates("reduce_immigration"),
            "whites_understand_blacks": agreement_templates("whites_understand_blacks"),
            "slavery_influence": agreement_templates("slavery_influence"),
            "resent_white_denial": agreement_templates("resent_white_denial"),
        }


if __name__ == "__main__":
    survey = CcesSurvey()
    factory = CcesFactory(survey)
