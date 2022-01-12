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
            " ".join([self.code_dict[dem][row[dem]] for dem in self.present_dems])
            + "\n\n"
        )

    def _make_backstory_qa(self, row):
        return (
            f"Q: What is your age?\nA: {self.code_dict['age'][row['age']]}\n\n"
            f"Q: What is your gender?\nA: {self.code_dict['gender'][row['gender']]}\n\n"
            f"Q: What is your political affiliation?\nA: {self.code_dict['party'][row['party']]}\n\n"
            f"Q: What is your education?\nA: {self.code_dict['education'][row['education']]}\n\n"
            f"Q: What is your ideology?\nA: {self.code_dict['ideology'][row['ideology']]}\n\n"
            f"Q: What is your income?\nA: {self.code_dict['income'][row['income']]}\n\n"
            f"Q: What is your religion?\nA: {self.code_dict['religion'][row['religion']]}\n\n"
            f"Q: What is your race/ethnicity?\nA: {self.code_dict['race_ethnicity'][row['race_ethnicity']]}\n\n"
            f"Q: What region of the country are your from?\nA: {self.code_dict['region'][row['region']]}\n\n"
            f"Q: What is your marital status?\nA: {self.code_dict['marital_status'][row['marital_status']]}\n\n"
        )

    def _make_backstory_convo(self, row):
        return (
            f"P1: What is your age?\nP2: {self.code_dict['age'][row['age']]}\n"
            f"P1: What is your gender?\nP2: {self.code_dict['gender'][row['gender']]}\n"
            f"P1: What is your political affiliation?\nP2: {self.code_dict['party'][row['party']]}\n"
            f"P1: What is your education?\nP2: {self.code_dict['education'][row['education']]}\n"
            f"P1: What is your ideology?\nP2: {self.code_dict['ideology'][row['ideology']]}\n"
            f"P1: What is your income?\nP2: {self.code_dict['income'][row['income']]}\n"
            f"P1: What is your religion?\nP2: {self.code_dict['religion'][row['religion']]}\n"
            f"P1: What is your race/ethnicity?\nP2: {self.code_dict['race_ethnicity'][row['race_ethnicity']]}\n"
            f"P1: What region of the country are your from?\nP2: {self.code_dict['region'][row['region']]}\n"
            f"P1: What is your marital status?\nP2: {self.code_dict['marital_status'][row['marital_status']]}\n"
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
        QA_MAPPING_SHOTS = [
            "Question: What is your favorite color?\nAnswer (Blue, Green): Blue\n\n",
            "Question: Do you like music?\nAnswer (Yes, No): Yes\n\n",
            "Question: How is your day going?\nAnswer (Good, Bad): Good\n\n",
        ]
        YES_NO_SHOTS = [
            "Do you like tacos? Yes\n",
            "Are you a fan of Spiderman? No\n",
            "Do you like the winter time? Yes\n",
        ]

        YES_NO_REPHRASINGS = {
            "crime_victim": [
                "Have you been a crime victim this past year?",
                "Were you the victim of a crime in the past year?",
                "Would you consider yourself the victim of a crime this past year?",
                "Have you been the victim of a crime within the past year?",
            ],
            "gender_change": [
                "Have you ever tried to change your gender?",
                "Have you ever changed your clothes, changed your name, or undergone surgery to change your gender?",
                "Do you consider yourself a transvestite?",
                "Have you ever been a different gender than the gender you were assigned at birth?",
            ],
        }

        POLICE_FEEL_REPHRASINGS = {
            "enumerated": "Do the police make you feel safe or unsafe?",
            "safe_explicit": "Do the police make your feel safe?",
            "unsafe_explicit": "Do the police make you feel unsafe?",
            "quotations": 'Do the police make you feel "safe" or "unsafe"?',
            "implied": "How do the police make you feel?",
            "noun": "Police",
            "statment_safe_explicit": "the police make me feel safe.",
            "statment_unsafe_explicit": "the police make me feel unsafe.",
        }

        SAFE_UNSAFE_SHOTS = [
            "Tigers: Unsafe\n",
            "Home: Safe\n",
            "Heights: Unsafe\n",
        ]

        # Not done yet
        NATIONS_ECONOMY_REPHRASINGS = {
            "enumerated": "Would you say that over the past year the nation's economy has ?",
            "safe_explicit": "Do the police make your feel safe?",
            "unsafe_explicit": "Do the police make you feel unsafe?",
            "quotations": 'Do the police make you feel "safe" or "unsafe"?',
            "implied": "How do the police make you feel?",
            "noun": "Police",
            "statment_safe_explicit": "the police make me feel safe.",
            "statment_unsafe_explicit": "the police make me feel unsafe.",
        }

        INCOME_CHANGE_REPHRASINGS = {
            "enumerated": "Over the past year, has your household's annual income increased, decreased, or stayed the same?",
            "quotations": 'Over the past year, has your household\'s annual income "increased", "decreased", or "stayed the same"?',
            "implied": "Over the past year, how has your household's annual income changed?",
            "noun": "Income",
        }

        support_oppose_templates = lambda dv: {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}\n"
                    f"A: I"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "qa2": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + QA_SHOTS[0]
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}\n"
                    f"A: I"
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
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}\n"
                    f"A: I"
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
                    + f"Q: Do you support or oppose the following proposal? {self.questions[dv]}\n"
                    f"A: I"
                ),
                {
                    "Support": "support",
                    "Oppose": "oppose",
                },
            ),
            "conversation1": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P1: Do you support or oppose the following proposal? {self.questions[dv]}\n"
                    f"P2: I"
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
                    f"P1: Do you support or oppose the following proposal? {self.questions[dv]}\n"
                    f"P2: I"
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
                    f"{self.questions[dv]}: I"
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
                    + f"Proposal: {self.questions[dv]}\n"
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
                    + f"Proposal: {self.questions[dv]}\n"
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
                    + f"Someone asked me if I support or oppose the following proposal: {self.questions[dv]} "
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
                    + f'Someone asked me if I support or oppose the following proposal: "{self.questions[dv]} "'
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
                    + f'Choosing between "support" or "oppose" for the following proposal: "{self.questions[dv]} "'
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
                    f"Proposal: {self.questions[dv]}\n"
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
                    + f"Proposal: {self.questions[dv]}\n"
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
                    + f"Proposal: {self.questions[dv]}\n"
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
                    + f"Proposal: {self.questions[dv]}\n"
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
                    + f'Q: Do you support this proposal: "{self.questions[dv]}"\n'
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f'Choosing "agree", "disagree", or "neither" for the following statement: "{self.questions[dv]}"'
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
                    f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + AGREEMENT_SHOTS[0]
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:2])
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:3])
                    + f"Statement: {self.questions[dv]}\n"
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
                    + QA_MAPPING_SHOTS[0]
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + "".join(QA_MAPPING_SHOTS[:2])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + "".join(QA_MAPPING_SHOTS[:3])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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

        yes_no_templates = lambda dv: {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row) + f"Q: {self.questions[dv]}\n" f"A:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "qa2": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + QA_SHOTS[0]
                    + f"Q: {self.questions[dv]}\n"
                    f"A:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "qa3": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:2])
                    + f"Q: {self.questions[dv]}\n"
                    f"A:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "qa4": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:3])
                    + f"Q: {self.questions[dv]}\n"
                    f"A:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "conversation1": (
                lambda row: (
                    self._make_backstory_convo(row) + f"P1: {self.questions[dv]}\n"
                    f"P2:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "conversation2": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P2: Ask me any question and I'll answer honestly.\n"
                    f"P1: {self.questions[dv]}\n"
                    f"P2:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "yes_no1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer yes or no to the following questions?\n\n"
                    f"{self.questions[dv]}: I"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "yes_no2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer yes or no to the following questions?\n\n"
                    + YES_NO_SHOTS[0]
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "yes_no3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer yes or no to the following questions?\n\n"
                    + "".join(YES_NO_SHOTS[:2])
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "yes_no4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer yes or no to the following questions?\n\n"
                    + "".join(YES_NO_SHOTS[:3])
                    + f"{self.questions[dv]}:"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "finish_sentence1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {self.questions[dv]}\n"
                    f'Choosing between "yes" or "no" for the previous question, I would say "'
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "finish_sentence2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {self.questions[dv]}\n"
                    f"If I have to answer yes or no to this question I would say"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "finish_sentence3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f'Choosing "yes" or "no" for the following question: "{self.questions[dv]}"'
                    f'I choose "'
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "qa_mapping1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + QA_MAPPING_SHOTS[0]
                    + f"Question: {self.questions[dv]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "qa_mapping2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + "".join(QA_MAPPING_SHOTS[:2])
                    + f"Question: {self.questions[dv]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "qa_mapping3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + "".join(QA_MAPPING_SHOTS[:3])
                    + f"Question: {self.questions[dv]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "rephrase1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {YES_NO_REPHRASINGS[dv][0]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "rephrase2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {YES_NO_REPHRASINGS[dv][1]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "rephrase3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {YES_NO_REPHRASINGS[dv][2]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
            "rephrase4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {YES_NO_REPHRASINGS[dv][3]}\n"
                    f"Answer (yes, no):"
                ),
                {
                    "Yes": "yes",
                    "No": "no",
                },
            ),
        }

        safe_unsafe_templates = {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"A:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "qa2": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + QA_SHOTS[0]
                    + f"Q: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"A: I"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "qa3": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:2])
                    + f"Q: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"A: I"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "qa4": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + "".join(QA_SHOTS[:3])
                    + f"Q: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"A: I"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "conversation1": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P1: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"P2:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "conversation2": (
                lambda row: (
                    self._make_backstory_convo(row)
                    + f"P2: Ask me any question and I'll answer honestly.\n"
                    f"P1: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"P2:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "safe_unsafe1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer whether the following things make me feel safe or unsafe?\n\n"
                    f"{POLICE_FEEL_REPHRASINGS['noun']}:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "safe_unsafe2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer whether the following things make me feel safe or unsafe?\n\n"
                    + SAFE_UNSAFE_SHOTS[0]
                    + f"{POLICE_FEEL_REPHRASINGS['noun']}:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "safe_unsafe3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer whether the following things make me feel safe or unsafe?\n\n"
                    + "".join(SAFE_UNSAFE_SHOTS[:2])
                    + f"{POLICE_FEEL_REPHRASINGS['noun']}:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "safe_unsafe4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"I will answer whether the following things make me feel safe or unsafe?\n\n"
                    + "".join(SAFE_UNSAFE_SHOTS[:3])
                    + f"{POLICE_FEEL_REPHRASINGS['noun']}:"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "finish_sentence1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {POLICE_FEEL_REPHRASINGS['implied']}\n"
                    f"Choosing between safe and unsafe for the previous proposal, I feel"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "finish_sentence2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Question: {POLICE_FEEL_REPHRASINGS['enumerated']}\n"
                    f"If I have to safe or unsafe to answer this question I would say"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "finish_sentence3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Someone asked me the following question. {POLICE_FEEL_REPHRASINGS['quotations']} "
                    f'I said "'
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "finish_sentence4": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Someone asked me if {POLICE_FEEL_REPHRASINGS['statement_safe_explicit']}. "
                    f"I said"
                ),
                {
                    "Mostly safe": "yes",
                    "Somewhat safe": "yes",
                    "Somewhat unsafe": "no",
                    "Mostly unsafe": "no",
                },
            ),
            "finish_sentence5": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + f"Someone asked me if {POLICE_FEEL_REPHRASINGS['statement_unsafe_explicit']}. "
                    f"I said"
                ),
                {
                    "Mostly safe": "no",
                    "Somewhat safe": "no",
                    "Somewhat unsafe": "yes",
                    "Mostly unsafe": "yes",
                },
            ),
            "yes_no1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: {POLICE_FEEL_REPHRASINGS['safe_explicit']}\n"
                    f"A:"
                ),
                {
                    "Mostly safe": "yes",
                    "Somewhat safe": "yes",
                    "Somewhat unsafe": "no",
                    "Mostly unsafe": "no",
                },
            ),
            "yes_no2": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: {POLICE_FEEL_REPHRASINGS['unsafe_explicit']}\n"
                    f"A:"
                ),
                {
                    "Mostly safe": "no",
                    "Somewhat safe": "no",
                    "Somewhat unsafe": "yes",
                    "Mostly unsafe": "yes",
                },
            ),
            "qa_mapping1": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + QA_MAPPING_SHOTS[0]
                    + f"Question: {POLICE_FEEL_REPHRASINGS['implied']}\n"
                    f"Answer (safe, unsafe):"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "qa_mapping2": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + "".join(QA_MAPPING_SHOTS[:2])
                    + f"Question: {POLICE_FEEL_REPHRASINGS['implied']}\n"
                    f"Answer (safe, unsafe):"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
            "qa_mapping3": (
                lambda row: (
                    self._make_backstory_paragraph(row)
                    + "".join(QA_MAPPING_SHOTS[:3])
                    + f"Question: {POLICE_FEEL_REPHRASINGS['implied']}\n"
                    f"Answer (safe, unsafe):"
                ),
                {
                    "Mostly safe": "safe",
                    "Somewhat safe": "safe",
                    "Somewhat unsafe": "unsafe",
                    "Mostly unsafe": "unsafe",
                },
            ),
        }

        # Not done yet
        better_worse_templates = lambda dv: {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f'Choosing "agree", "disagree", or "neither" for the following statement: "{self.questions[dv]}"'
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
                    f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + AGREEMENT_SHOTS[0]
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:2])
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:3])
                    + f"Statement: {self.questions[dv]}\n"
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
                    + QA_MAPPING_SHOTS[0]
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + "".join(QA_MAPPING_SHOTS[:2])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + "".join(QA_MAPPING_SHOTS[:3])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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

        # Not done yet
        increased_decreased_templates = lambda dv: {
            "qa1": (
                lambda row: (
                    self._make_backstory_qa(row)
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Q: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    f"P1: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f'Choosing "agree", "disagree", or "neither" for the following statement: "{self.questions[dv]}"'
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
                    f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + AGREEMENT_SHOTS[0]
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:2])
                    + f"Statement: {self.questions[dv]}\n"
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
                    + f"I will now respond agree, disagree, or neither to the following statements.\n\n"
                    + "".join(AGREEMENT_SHOTS[:3])
                    + f"Statement: {self.questions[dv]}\n"
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
                    + QA_MAPPING_SHOTS[0]
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + "".join(QA_MAPPING_SHOTS[:2])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
                    + "".join(QA_MAPPING_SHOTS[:3])
                    + f"Question: Do you agree or disagree with the following statements (or neither)? {self.questions[dv]}\n"
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
            "crime_victim": yes_no_templates("crime_victim"),
            "police_feel": safe_unsafe_templates,
            "body_cameras": support_oppose_templates("body_cameras"),
            "increase_police": support_oppose_templates("increase_police"),
            "decrease_police": support_oppose_templates("decrease_police"),
            "nations_economy": {},
            "income_change": {},
            "labor_union": {},
            "gender_change": yes_no_templates("gender_change"),
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
