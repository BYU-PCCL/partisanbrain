from collections import defaultdict
from infra_modules import Dataset

import json
import pandas as pd


class CommonSenseQaDataset(Dataset):

    def __init__(self, sample_seed=0, n=None):

        # Token sets for Common Sense QA can vary question by
        # question
        self._token_set = lambda row: {
            k: row[k].split(" ")[0] for k in ["A", "B", "C", "D", "E"]
        }

        self._alphabet_token_set = {
            k: [k] for k in ["A", "B", "C", "D", "E"]
        }

        super().__init__(sample_seed=sample_seed,
                         n=n,
                         opening_func=self._load_data)

    def _load_data(self, fname):

        df_dict = defaultdict(list)

        with open(fname, "r") as f:
            json_questions = list(f)

        for json_question in json_questions:
            question = json.loads(json_question)
            df_dict["ground_truth"].append(question["answerKey"])
            df_dict["question"].append(question["question"]["stem"])

            # Drop questions if there are duplicates
            # across the first words of the answer options
            all_choices = question["question"]["choices"]
            first_words = [c["text"].split(" ")[0] for c in all_choices]
            if len(first_words) != len(set(first_words)):
                df_dict["ground_truth"].pop()
                df_dict["question"].pop()
                continue

            n_choices = 5
            for choice_idx in range(n_choices):
                choice_data = question["question"]["choices"][choice_idx]
                df_dict[choice_data["label"]].append(choice_data["text"])

        return pd.DataFrame(df_dict)

    def _modify_raw_data(self, df):
        return df

    def _get_templates(self):

        templates = {
            "qca": (lambda row: (f"Q: {row['question']}\n\n"
                                 f"Choices: {row['A']}, {row['B']}, "
                                 f"{row['C']}, {row['D']}, {row['E']}\n\n"
                                 "A:"), self._token_set),
            "fsqca": (lambda row: ("Q: What might a vegan eat for "
                                   "breakfast?\n\n"
                                   "Choices: oats, bacon, sausage, "
                                   "omelet, ham\n\n"
                                   "A: oats\n\n"
                                   f"Q: {row['question']}\n\n"
                                   f"Choices: {row['A']}, {row['B']}, "
                                   f"{row['C']}, {row['D']}, {row['E']}\n\n"
                                   "A:"), self._token_set),
            "ts": (lambda row: ("Teacher: I'm going to ask you a common "
                                "sense question.\n\n"
                                "Student: Alright.\n\n"
                                f"Teacher: {row['question']}\n\n"
                                "Student: What are the possible answers?\n\n"
                                "Teacher: The answer is either "
                                f"\"{row['A']},\" \"{row['B']},\" "
                                f"\"{row['C']},\" \"{row['D']},\" or "
                                f"\"{row['E']}.\"\n\n"
                                "Student: I know the right answer - it's \""),
                   self._token_set),
            "abfs": (lambda row: ("Instructions: For each question below, "
                                  "choose the answer from the answer "
                                  "bank corresponding to the question "
                                  "that best answers the question.\n\n"
                                  "Question 1 Answer Bank: "
                                  "ladybug, bunny, goldfish, leopard, "
                                  "caterpillar"
                                  "Question: What animal would "
                                  "be most dangerous for a human to encounter "
                                  "in the wild?\n\n"
                                  "Answer: leopard\n\n"
                                  "Question 2 Answer Bank: "
                                  f"{row['A']}, {row['B']}, {row['C']}, "
                                  f"{row['D']}, {row['E']}\n\n"
                                  f"Question: {row['question']}\n\n"
                                  "Answer:"), self._token_set),
            "tsfs": (lambda row: ("Teacher: I'm going to ask you a common "
                                  "sense question.\n\n"
                                  "Student: Alright.\n\n"
                                  "Teacher: What would you not expect to read "
                                  "about in a book on the founding of the "
                                  "United States?\n\n"
                                  "Student: What are the possible answers?\n\n"
                                  "Teacher: The answer is either \"george "
                                  "washington,\" \"declaration of "
                                  "independence,\" \"boston tea party,\" "
                                  "\"star spangled banner,\" or \"vampire "
                                  "assassins.\"\n\n"
                                  "Student: I know the right answer - "
                                  "it's \"vampire assassins.\"\n\n"
                                  "Teacher: That's right! Here's another "
                                  "common sense question "
                                  f"for you. {row['question']}\n\n"
                                  "Student: What are the possible answers?\n\n"
                                  "Teacher: The answer is either "
                                  f"\"{row['A']},\" \"{row['B']},\" "
                                  f"\"{row['C']},\" \"{row['D']},\" or "
                                  f"\"{row['E']}.\"\n\n"
                                  "Student: I know the right answer - "
                                  "it's \""),
                     self._token_set),
            "vb": (lambda row: (f"{row['question']}\n\n"
                                f"A: {row['A']}\n"
                                f"B: {row['B']}\n"
                                f"C: {row['C']}\n"
                                f"D: {row['D']}\n"
                                f"E: {row['E']}\n\n"
                                "Answer:"),
                   self._alphabet_token_set),
            "vbfs": (lambda row: ("What would you use to put out a fire?\n\n"
                                  "A: gasoline\n"
                                  "B: poison\n"
                                  "C: laundry detergent\n"
                                  "D: water\n"
                                  "E: pencil\n\n"
                                  "Answer: water\n\n"
                                  f"{row['question']}\n\n"
                                  f"A: {row['A']}\n"
                                  f"B: {row['B']}\n"
                                  f"C: {row['C']}\n"
                                  f"D: {row['D']}\n"
                                  f"E: {row['E']}\n\n"
                                  "Answer:"),
                     self._alphabet_token_set),
            "pyfs": (lambda row: ("# multiple choice quiz questions and "
                                  "answers\n\n"
                                  "qa = {'q': 'What is France?', "
                                  "'choices': ['state', 'city', 'country', "
                                  "'continent', 'mountain range'], "
                                  "'answer': 'country',"
                                  f"'q': '{row['question']}', "
                                  f"'choices': [{row['A']}, {row['B']}, "
                                  f"{row['C']}, {row['D']}, {row['E']}], "
                                  "'answer': '"),
                     self._token_set),
            "ps": (lambda row: ("Common Sense Quiz Answer Key\n\n"
                                f"Question 1: {row['question']}\n\n"
                                f"A: {row['A']}\n"
                                f"B: {row['B']}\n"
                                f"C: {row['C']}\n"
                                f"D: {row['D']}\n"
                                f"E: {row['E']}\n\n"
                                "Correct Answer:"),
                   self._alphabet_token_set),
            "psfs": (lambda row: ("Common Sense Quiz Answer Key\n\n"
                                  "Question 1: Where would people not "
                                  "typically go for fun?\n\n"
                                  "A: theme park\n"
                                  "B: movie theatre\n"
                                  "C: carnival\n"
                                  "D: waste management facility\n"
                                  "E: beach\n\n"
                                  "Correct Answer: D"
                                  f"Question 2: {row['question']}\n\n"
                                  f"A: {row['A']}\n"
                                  f"B: {row['B']}\n"
                                  f"C: {row['C']}\n"
                                  f"D: {row['D']}\n"
                                  f"E: {row['E']}\n\n"
                                  "Correct Answer:"),
                     self._alphabet_token_set),
        }

        return templates


if __name__ == "__main__":
    CommonSenseQaDataset()
