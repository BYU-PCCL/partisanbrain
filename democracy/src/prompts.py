from abc import ABC, abstractmethod

import openai
import os

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


FILLERS = {
    "[P]": {"republican": "Republican",
            "democrat": "Democratic"},
    "[Ps]": {"republican": "Republicans",
             "democrat": "Democrats"},
    "[Pp]": {"republican": "Republican",
             "democrat": "Democrat"}
}


class Question:

    def __init__(self, text_with_filler, possible_answers):
        self._text_with_filler = text_with_filler
        self._possible_answers = possible_answers

    def get_text(self, party):
        assert party in ["republican", "democrat"]

        text = self._text_with_filler

        for filler in FILLERS:
            text = text.replace(filler, FILLERS[filler][party])

        return text

    def get_possible_answers(self):
        return self._possible_answers


QUESTIONS = [
    Question(("If asked whether [Ps] should \"reduce\" "
              "or \"maintain\" the number of polling "
              "stations in areas that support [Ps], "
              "I would say that [Ps] should \""),
             ["reduce", "maintain"]),
    Question(("If asked whether [P] governors should "
              "\"ignore\" or \"heed\" unfavorable "
              "court rulings by [P]-appointed judges, "
              "I would say that [P] governors should \""),
             ["ignore", "heed"]),
    Question(("If asked whether [P] governors should "
              "\"prosecute\" or \"tolerate\" journalists who "
              "accuse [P] politicians of misconduct without "
              "revealing sources, I would say that [P] "
              "governors should \""),
             ["prosecute", "tolerate"]),
    Question(("If asked whether [Ps] should \"accept\" "
              "or \"deny\" the results of elections if "
              "they lose, I would say that [Ps] "
              "should \""),
             ["deny", "accept"]),
    Question(("If asked whether it is \"always\" or "
              "\"never\" OK for a [Pp] to send "
              "threatening and intimidating messages "
              "to [P] party leaders, I would answer by "
              "saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether it is \"always\" or \"never\" "
              "OK for an ordinary [Pp] in the public to harass "
              "an ordinary [Pp] on the Internet, in a way that "
              "makes the target feel frightened, I would "
              "answer by saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether it is \"always\" or \"never\" "
              "justified for [Ps] to use violence in advancing "
              "their political goals these days, I would answer "
              "by saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether it is \"always\" or \"never\" "
              "justified for [Ps] to use violence if the [P] "
              "party wins more races in the next election, "
              "I would answer by saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether I feel \"warm\" or \"cold\" "
              "towards [Ps], I would say that I feel \""),
             ["warm", "cold"]),
    Question(("If I were given a dollar and then asked "
              "whether I would share \"all of it\" or "
              "\"none of it\" with a [Pp], I would "
              "answer that I would share \""),
             ["none of it", "all of it"])
]


class PromptProcessor:

    def __init__(self, engine):
        possible_engines = ["davinci", "curie", "babbage", "ada"]
        assert engine in possible_engines, f"{engine} is not a valid engine"
        self._engine = engine

    def process_prompt(self, prompt, create_kwargs=None):
        # TODO: Actually make create_kwargs do something
        try:
            return openai.Completion.create(engine=self._engine,
                                            prompt=prompt,
                                            max_tokens=1,
                                            logprobs=100)
        # TODO: Catch more specific exception here
        except Exception as e:
            print(f"Exception in method process_prompt: {e}")
            return None


class Prompt(ABC):

    def __init__(self, respondent_data, party, question_idx):
        assert party in ["republican", "democrat"]
        self._row = respondent_data
        self._question_idx = question_idx
        self._party = party

    def _get_backstory(self):

        backstory = []

        # Gender
        gender_dict = {
            "male": "male",
            "female": "female"
        }
        backstory.append(f"I am {gender_dict[self._row.gender]}.")

        # Race
        race_dict = {
            "White": "White",
            "Black": "Black",
            "Hispanic": "Hispanic",
            "Asian/Native Hawaiian/Pacific Islander": None,
            "Native American/Alaskan Native": None
        }
        backstory.append(f"I am {race_dict[self._row.race]}.")

        # Age
        age_dict = {
            "18-24": "I am between 18 and 24 years old.",
            "25-34": "I am between 25 and 34 years old.",
            "35-44": "I am between 35 and 44 years old.",
            "45-54": "I am between 45 and 54 years old.",
            "55-64": "I am between 55 and 64 years old.",
            "65-75": "I am between 65 and 75 years old.",
            "75+": "I am older than 75 years old."
        }
        backstory.append(age_dict[self._row.age_range])

        # Education
        educ_dict = {
            "no high school degree": "I never graduated from high school.",
            "high school graduate": "I graduated from high school.",
            "some college": "I completed some college.",
            "bachelor's degree": "I earned a bachelor's degree.",
            "graduate degree": "I earned a graduate degree."
        }
        backstory.append(educ_dict[self._row.education])

        return " ".join(backstory)

    @abstractmethod
    def _get_treatment(self):
        pass

    def _get_question(self):
        return QUESTIONS[self._question_idx]

    def get_prompt(self):
        prompt = self._get_backstory() + " "
        prompt += self._get_treatment()
        prompt += self._get_question().get_text(self._party)
        return prompt


class PassivePrompt(Prompt):

    def __init__(self, respondent_data, party, question_idx):
        super().__init__(respondent_data, party, question_idx)

    def _get_treatment(self):
        return ""


class KalmoePrompt(Prompt):

    def __init__(self, respondent_data, party, question_idx, context_quote):
        super().__init__(respondent_data, party, question_idx)
        self._context_quote = context_quote

    def _get_treatment(self):
        return self._context_quote


if __name__ == "__main__":
    import pandas as pd
    row = pd.DataFrame({"age_range": "75+",
                        "gender": "male",
                        "race": "White",
                        "education": "high school graduate"}, index=[0]).iloc[0]
    passive_prompt = PassivePrompt(row, "republican", 0)
    print(passive_prompt.get_prompt())
