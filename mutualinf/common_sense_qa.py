from collections import defaultdict
from infra_modules import Dataset

import json
import pandas as pd

SHOTS = [
'''Question: I'm crossing the river, my feet are wet but my body is dry, where am I?
Choices: bridge, waterfall, valley, pebble, mountain
Answer: "valley" is the best answer. While "bridge" also seems to make sense at first, your feet would not be wet if you crossed over a river on a bridge. Meanwhile, if you crossed the river at a valley, the river would be shallow, only getting your feet wet.''',

'''Question: In what Spanish speaking North American country can you get a great cup of coffee?
Choices: mildred's coffee shop, mexico, diner, kitchen, canteen
Answer: "mexico" is the best answer. It's true that you can get a cup of coffee in a coffee shop or a diner, but the question specifically asks for a Spanish speaking North American country. Mexico is the only country listed, so that must be the correct answer.''',

'''Question: I'm crossing the river, my feet are wet but my body is dry, where am I?
Choices: bridge, waterfall, valley, pebble, mountain
Answers (in order of best to worst): valley, bridge, waterfall, mountain, pebble''',

'''Question: In what Spanish speaking North American country can you get a great cup of coffee?
Choices: mildred's coffee shop, mexico, diner, kitchen, canteen
Answers (in order of best to worst): mexico, mildred's coffee shop, diner, kitchen, canteen''',

'''"I'm crossing the river, my feet are wet but my body is dry, where am I?", "bridge, waterfall, valley, pebble, mountain", -> "valley"''',

'''"In what Spanish speaking North American country can you get a great cup of coffee?", "mexico, mildred's coffee shop, diner, kitchen, canteen", -> "mexico"''',
]


class CommonSenseQaDataset(Dataset):

    def __init__(self, sample_seed=0, n=None):

        # Token sets for Common Sense QA can vary question by
        # question
        self._token_set = lambda row: {
            k: row[k].split(" ")[0] for k in ["A", "B", "C", "D", "E"]
        }

        self._alphabet_token_set = ["A", "B", "C", "D", "E"]

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

            "vbfs1": (lambda row: ("What would you use to put out a fire?\n"
                                  "A: gasoline\n"
                                  "B: poison\n"
                                  "C: laundry detergent\n"
                                  "D: water\n"
                                  "E: pencil\n"
                                  "Answer: water\n\n"
                                  f"{row['question']}\n"
                                  f"A: {row['A']}\n"
                                  f"B: {row['B']}\n"
                                  f"C: {row['C']}\n"
                                  f"D: {row['D']}\n"
                                  f"E: {row['E']}\n"
                                  "Answer:"),
                     self._alphabet_token_set),

            "vbfs2": (lambda row: ("What would you use to put out a fire?\n"
                                  "A: gasoline\n"
                                  "B: poison\n"
                                  "C: laundry detergent\n"
                                  "D: water\n"
                                  "E: pencil\n"
                                  "Answer: D. water\n\n"
                                  f"{row['question']}\n"
                                  f"A: {row['A']}\n"
                                  f"B: {row['B']}\n"
                                  f"C: {row['C']}\n"
                                  f"D: {row['D']}\n"
                                  f"E: {row['E']}\n"
                                  "Answer:"),
                     self._alphabet_token_set),

            "pyfs": (lambda row: ("# multiple choice quiz questions and "
                                  "answers\n\n"
                                  "qa = ['q': 'What is France?', "
                                  "'choices': ['state', 'city', 'country', "
                                  "'continent', 'mountain range'], "
                                  "'answer': 'country', ], "
                                  f"'[q': '{row['question']}', "
                                  f"'choices': [{row['A']}, {row['B']}, "
                                  f"{row['C']}, {row['D']}, {row['E']}], "
                                  "'answer': '"),
                     self._token_set),

            "csv": (lambda row: ("questions,choices,answers\n"
                                  "\"What is France?\",\"[state,city,country,"
                                  "continent,mountain range]\",country\n"
                                  f"\"{row['question']}\",\"[{row['A']},{row['B']},{row['C']},{row['D']},{row['E']}]\","),
                     self._token_set),
            
            'explanation_1shot': (lambda row: (
                f"Choose the best single answer to the question, and explain your answer.\n\n" +
                SHOTS[0] + '\n\n' +
                f"Question: {row['question']}\n"
                f"Choices: {row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\n"
                f"Answer: \""), self._token_set),

            'explanation_2shot': (lambda row: (
                f"Choose the best single answer to the question, and explain your answer.\n\n" +
                SHOTS[0] + '\n\n' +
                SHOTS[1] + '\n\n' +
                f"Question: {row['question']}\n"
                f"Choices: {row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\n"
                f"Answer: \""), self._token_set),
            
            'best_worst_0shot': (lambda row: (
                f"Given the question, order the options from best answer to the question to worst answer to the question.\n\n"
                f"Question: {row['question']}\n"
                f"Choices: {row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\n"
                f"Answers (in order of best to worst):"), self._token_set),

            'best_worst_1shot': (lambda row: (
                f"Given the question, order the options from best answer to the question to worst answer to the question.\n\n" + 
                SHOTS[2] + '\n\n' +
                f"Question: {row['question']}\n"
                f"Choices: {row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\n"
                f"Answers (in order of best to worst):"), self._token_set),

            'best_worst_2shot': (lambda row: (
                f"Given the question, order the options from best answer to the question to worst answer to the question.\n\n" + 
                SHOTS[2] + '\n\n' +
                SHOTS[3] + '\n\n' +
                f"Question: {row['question']}\n"
                f"Choices: {row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\n"
                f"Answers (in order of best to worst):"), self._token_set),
            
            'open_ai_1shot': (lambda row: (
                f"Given the following questions and choices, pick the choice that corresponds best to the question.\n\n" + 
                SHOTS[4] + '\n' +
                f"\"{row['question']}\", \"{row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\" -> \""), self._token_set),

            'open_ai_2shot': (lambda row: (
                f"Given the following questions and choices, pick the choice that corresponds best to the question.\n\n" + 
                SHOTS[4] + '\n' +
                SHOTS[5] + '\n' +
                f"\"{row['question']}\", \"{row['A']}, {row['B']}, {row['C']}, {row['D']}, {row['E']}\" -> \""), self._token_set),
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
                                  "typically go for fun?\n"
                                  "A: theme park\n"
                                  "B: movie theatre\n"
                                  "C: carnival\n"
                                  "D: waste management facility\n"
                                  "E: beach\n"
                                  "Correct Answer: D\n\n"
                                  f"Question 2: {row['question']}\n"
                                  f"A: {row['A']}\n"
                                  f"B: {row['B']}\n"
                                  f"C: {row['C']}\n"
                                  f"D: {row['D']}\n"
                                  f"E: {row['E']}\n"
                                  "Correct Answer:"),
                     self._alphabet_token_set),
            "gs": (lambda row: ("Me: I watched the most recent episode "
                                "of the \"Is It Really Common Sense\" "
                                "game show yesterday night.\n"
                                "Friend: Oh, how was it?\n"
                                "Me: It was good. I remember one of the "
                                "questions.\n"
                                "Friend: What was the question?\n"
                                f"Me: {row['question']}\n"
                                "Friend: What were the options?\n"
                                f"Me: {row['A']}, {row['B']}, "
                                f"{row['C']}, {row['D']}, or {row['E']}\n"
                                "Friend: Did the contestant get the "
                                "answer right?\n"
                                "Me: Yep!\n"
                                "Friend: Which of the options was correct?\n"
                                "Me: The correct answer was"),
                   self._token_set)
        }

        return templates


if __name__ == "__main__":
    CommonSenseQaDataset()
