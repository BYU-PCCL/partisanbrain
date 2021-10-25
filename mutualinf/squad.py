from collections import defaultdict
from dataset import Dataset

import pandas as pd


SHOTS = [
"""CONTEXT:
BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.
QUESTIONS:
1) What high school GPA for BYU freshmen have on average?
Answer: "3.71"
""",

"""CHAPTER QUIZ
PASSAGE: BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.
QUESTIONS:
1) What high school GPA for BYU freshmen have on average?

ANSWER KEY: 
1) 3.71
""", 

"""P1: BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.
P2: What high school GPA for BYU freshmen have on average?
P1: 3.71
""",

"""\"BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.", "What high school GPA for BYU freshmen have on average?" -> "3.71\""""

"""\"BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.", "What high school GPA for BYU freshmen have on average?" -> "3.71\"
\"In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of precipitation include drizzle, rain, sleed, snow, graupel, and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. Short, intense periods of rain in scattered locations are called\"showers\".\", \"What causes precipitation to fall?" -> "gravity\""""
]

class SquadDataset(Dataset):

    def __init__(self, sample_seed=0, n=None):
        self._token_set = None
        super().__init__(sample_seed=sample_seed, n=n)
    
    '''
    def _open_json(self, fname):
        df = pd.read_json(fname)
        parsed_df = pd.DataFrame(columns = ['context', 'question', 'answer'])
        data_1 = df.data.values.tolist()
        for question_block in data_1:
            for qa in question_block['paragraphs']:
                context = qa['context']
                for question_answer in qa['qas']:
                    question = question_answer['question']
                    for data in question_answer['answers']:
                        answer = data['text']
                        if len((answer.split())) == 1:
                            parsed_df.loc[len(parsed_df.index)] = [context, question, answer]
        parsed_df.to_csv("parsed_data.csv")
        return parsed_df

    def _modify_raw_data(self, df):
        mod_df_dict = defaultdict(list)
        for _, row in df.iterrows():
            mod_df_dict['context'].append(row['context'])
            mod_df_dict["question"].append(row["question"])
            mod_df_dict["ground_truth"].append(row["text"])
        return pd.DataFrame(mod_df_dict, index=df.index)

    '''

    def _modify_raw_data(self, df):
        mod_df_dict = defaultdict(list)
        last_context = ''
        f_s_q = ''
        f_s_a = ''
        for _, row in df.iterrows():
            if len(str(row['text']).split(' ')) == 1:
                if last_context != row['context']:
                    last_context = row['context']
                    f_s_q = row['question']
                    f_s_a = row['text']
                else:
                    mod_df_dict['context'].append(str(row['context']))
                    mod_df_dict["question"].append(str(row["question"]))
                    mod_df_dict["ground_truth"].append(str(row["text"]))
                    mod_df_dict['few_shot_question'].append(str(f_s_q))
                    mod_df_dict['few_shot_answer'].append(str(f_s_a))

        return pd.DataFrame(mod_df_dict)


    def _get_templates(self):
        """
        template ideas:
        existing templates (5)
        chapter quiz/answer key (5)
        instruction (5)
        dialogue (5)
        few shot for each of the above categories
        """

        """
        OLD PROMPTS
        "basic_question": (lambda row: ("Context: "f"{row['context']}" "\n\nQ: "f"{row['question']}""\n\nA:"), self._token_set),
            
        "question_without_labels": (lambda row: (f"{row['context']}" "\n\n"f"{row['question']}"), self._token_set),
        
        "read_in_book": (lambda row: ("I read this in a book today:\n"f"{row['context']}" "\n"f"{row['question']}"), self._token_set),
        
        "read_with_question_prompt": (lambda row: ("I read this in a book today:\n"f"{row['context']}" "\nFrom that context, did you catch "f"{row['question']}"), self._token_set),
        
        "heard_from_friend": (lambda row: ("A friend of mine told me this:\n"f"{row['context']}" "\n"f"{row['question']}"), self._token_set),

        """

        t0 = ''


        templates = {

            'instruction_qa0' : (lambda row: (f"TASK: Using words from the CONTEXT, answer the below QUESTIONS.\n\n"
            f"CONTEXT:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['question']}\n"
            f"Answer: \""), self._token_set),

            'instruction_qa1' : (lambda row: (f"TASK: Answer the questions below using the phrasing from the context.\n\n"
            f"CONTEXT:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['question']}\n"
            f"Answer: \""), self._token_set),

            'instruction_qa2' : (lambda row: (f"TASK: Answer the questions below using the phrasing from the context.\n\n"
            f"CONTEXT:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['few_shot_question']}\nAnswer: \"{row['few_shot_answer']}\"\n\n"
            f"2) {row['question']}\nAnswer: \""), self._token_set),

            'instruction_qa3' : (lambda row: (f"TASK: Answer the questions below using the phrasing from the context.\n\n"
            f"{SHOTS[0]}\n\n"
            f"CONTEXT:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['question']}\n"
            f"Answer: \""), self._token_set),

            'answer_key0' : (lambda row: (f"CHAPTER QUIZ\n\n"
            f"PASSAGE:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['question']}\n\n"
            f"ANSWER KEY:\n1)"), self._token_set),

            'answer_key1' : (lambda row: (f"ANSWER KEY:\n\n"
            f"QUESTION1:\n\"{row['context']}\" {row['question']}\n"
            f"ANSWER1:"), self._token_set),

            'answer_key2' : (lambda row: (f"CHAPTER QUIZ\n\n"
            f"PASSAGE:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['few_shot_question']}\n"
            f"2) {row['question']}\n\n"
            f"ANSWER KEY:\n1) {row['few_shot_answer']}\n2)"), self._token_set),

            'answer_key3' : (lambda row: (SHOTS[1] + f"\nCHAPTER QUIZ\n\n"
            f"PASSAGE:\n{row['context']}\n\n"
            f"QUESTIONS:\n1) {row['question']}\n\n"
            f"ANSWER KEY:\n1)"), self._token_set),

            'dialogue0' : (lambda row: (f"P1: {row['context']}\n"
            f"P2: {row['question']}\n"
            f"P1: The answer is \""), self._token_set), 

            'dialogue1' : (lambda row: (f"P1 tells P2 some information, P2 asks comprehension questions, and P1 answers.\n\n"
            f"P1: {row['context']}\n"
            f"P2: {row['question']}\n"
            f"P1: The answer is \""), self._token_set), 

            'dialogue2' : (lambda row: (f"P1: {row['context']}\n"
            f"P2: {row['few_shot_question']}\n"
            f"P1: {row['few_shot_answer']}\n"
            f"P2: {row['question']}\n"
            f"P1:"), self._token_set),

            'dialogue3' : (lambda row: (SHOTS[2] + f"\n\nP1: {row['context']}\n"
            f"P2: {row['question']}\n"
            f"P1:"), self._token_set),

            "old0": (lambda row: ("Context: "f"{row['context']}" "\n\nQ: "f"{row['question']}""\n\nA:"), self._token_set),

            "old1": (lambda row: (f"{row['context']}" "\n\n"f"{row['question']}\n"
            f"The correct answer is:"), self._token_set),
        
            "old2": (lambda row: ("I read this in a book today:\n"f"{row['context']}" "\n"f"{row['question']}\nAnswer:"), self._token_set),
            
            "old3": (lambda row: ("I read this in a book today:\n"f"{row['context']}" "\nFrom that context, did you catch "f"{row['question']}\n"
            f"Yes, the answer is"), self._token_set),
            
            "old4": (lambda row: ("A friend of mine told me this:\n"f"{row['context']}\n"
            f"My friend then asked: {row['question']}\n"
            f"I answered:"), self._token_set),

            "openai0_shot": (lambda row: ("Given the following passages and questions, provide a brief, correct answer from the text.\n"
            f"\"{row['context']}\", \"{row['question']}\" -> \""), self._token_set),

            "openai1_shot": (lambda row: ("Given the following passages and questions, provide a brief, correct answer from the text.\n\n" +
            SHOTS[-1] + "\n" +
            f"\"{row['context']}\", \"{row['question']}\" -> \""), self._token_set),

            "openai2_shot": (lambda row: ("Given the following passages and questions, provide a brief, correct answer from the text.\n\n" +
            SHOTS[-1] + "\n" +
            f"\"{row['context']}\", \"{row['question']}\" -> \""), self._token_set),

        }
        return templates


if __name__ == "__main__":
    # Data should be at data/squad/raw.csv
    sd = SquadDataset()
