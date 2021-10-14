from collections import defaultdict
from dataset import Dataset

import pandas as pd

class SquadDataset(Dataset):

    def __init__(self):
        super().__init__()
    
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
            mod_df_dict["ground_truth"].append(row["answer"])
        return pd.DataFrame(mod_df_dict, index=df.index)

    def _get_templates(self):
        templates = {
            "basic_question": lambda row: ("Context: "f"{row['context']}" "\n\nQ: "f"{row['question']}""\n\nA:"),
            "question_without_labels": lambda row: (f"{row['context']}" "\n\n"f"{row['question']}"),
            "read_in_book": lambda row: ("I read this in a book today:\n"f"{row['context']}" "\n"f"{row['question']}"),
            "read_with_question_prompt": lambda row: ("I read this in a book today:\n"f"{row['context']}" "\nFrom that context, did you catch "f"{row['question']}"),
            "heard_from_friend": lambda row: ("A friend of mine told me this:\n"f"{row['context']}" "\n"f"{row['question']}"),
        }
        return templates


if __name__ == "__main__":
    # Data should be at data/example/raw.csv
    sd = SquadDataset()
