from collections import defaultdict
from dataset import Dataset

import sys
import pandas as pd


SHOTS = [
    """Word: bright
Usage 1: He is a bright child
Usage 2: The sun is very bright today
Meaning: different

""",
    """Word: air
Usage 1: Utah has too much air pollution.
Usage 2: Open a window and let in some air.
Meaning: same

""",
    """Word: cool
Usage 1: Her pants are cool.
Usage 2: Let your food cool.
Meaning: different

""",
    """Word: fight
Usage 1: My wife and I had a fight.
Usage 2: I fight for my freedom.
Meaning: same

""",
]

LOGIC_QUESTIONS = [
    """Q: What does 2 + 2 equal?
A: 4

""",
    """Q: If you are 60 inches tall how tall are you in feet?
A: 5 feet

""",
]

FACT_QUESTIONS = [
    """Q: What year did America first land on the moon?
A: 1969

""",
    """Q: What is the average height in America?
A: 5 feet 9 inches

""",
]

YES_NO_QUESTIONS = [
    """Q: Is the United States in South America?
A: No

""",
    """Q: Is the following sentence missing a comma? Before leaving I ate breakfast.
A: Yes

""",
]

WIC_CONTEXT = """Depending on its context, an ambiguous word can refer to multiple, potentially unrelated, meanings. Mainstream static word embeddings, such as Word2vec and GloVe, are unable to reflect this dynamic semantic nature. Contextualised word embeddings are an attempt at addressing this limitation by computing dynamic representations for words which can adapt based on context. A system's task on the WiC dataset is to identify the intended meaning of words. WiC is framed as a binary classification task. Each instance in WiC has a target word w, either a verb or a noun, for which two contexts are provided. Each of these contexts triggers a specific meaning of w. The task is to identify if the occurrences of w in the two contexts correspond to the same meaning or not. In fact, the dataset can also be viewed as an application of Word Sense Disambiguation in practise.
WiC features multiple interesting characteristics:

* It is suitable for evaluating a wide range of applications, including contextualized word and sense representation and Word Sense Disambiguation;
* It is framed asa binary classification dataset, in which, unlike Stanford Contextual Word Similarity (SCWS), identical words are paired with each other (in different contexts); hence, a context-insensitive word embedding model would perform similarly to a random baseline;
* It is constructed using high quality annotations curated by experts.

Examples from the dataset:
Context-1 // Context-2 // Target // Label
There's a lot of trash on the bed of the river // I keep a glass of water on my bed when I sleep // bed // Different
Air pollution // Open a window and let in some air // air // Same"""

WIKIPEDIA_CONTEXT = """In linguistics, a word sense is one of the meanings of a word. Words are in two sets: a large set with multiple meanings (word senses) and a small set with only one meaning (word sense). For example, a dictionary may have over 50 different senses of the word "play", each of these having a different meaning based on the context of the word's usage in a sentence, as follows:

"We went to see the play Romeo and Juliet at the theater."
"The coach devised a great play that put the visiting team on the defensive."
"The children went out to play in the park."
In each sentence we associate a different meaning of the word "play" based on hints the rest of the sentence gives us.

People and computers, as they read words, must use a process called word-sense disambiguation[1][2] to find the correct meaning of a word. This process uses context to narrow the possible senses down to the probable ones. The context includes such things as the ideas conveyed by adjacent words and nearby phrases, the known or probable purpose and register of the conversation or document, and the orientation (time and place) implied or expressed. The disambiguation is thus context-sensitive.

Advanced semantic analysis has resulted in a sub-distinction. A word sense corresponds either neatly to a seme (the smallest possible unit of meaning) or a sememe (larger unit of meaning), and polysemy of a word of phrase is the property of having multiple semes or sememes and thus multiple senses.

The following are examples of two sentences where the meaning of the word is either the same or different.

Examples:
There's a lot of trash on the bed of the river // I keep a glass of water on my bed when I sleep // bed // Different
Air pollution // Open a window and let in some air // air // Same"""


class WicDataset(Dataset):
    def __init__(self, sample_seed=0, n=None):
        self._token_set = {
            "question": {
                "True": ["yes"],
                "False": ["no"],
            },
            "true_false_classify": {
                "True": ["true"],
                "False": ["false"],
            },
            "few_shot": {
                "True": ["same"],
                "False": ["different"],
            },
        }
        super().__init__(sample_seed=sample_seed, n=n)

    def _modify_raw_data(self, df):
        # change "ground_truth" column from bool to str
        df["ground_truth"] = df["ground_truth"].astype(str)
        return df

    def _get_templates(self):
        templates = {
            "question0": (
                lambda row: (
                    f"{row.sentence1} // {row.sentence2}\n"
                    f'Choose "yes" or "no". Does the word {row.word} have the same meaning in the previous sentences? "'
                ),
                self._token_set["question"],
            ),
            "few_shot0": (
                lambda row: (
                    "Classify whether the following two sentences' use of the word has the same meaning or not.\n\n"
                    + SHOTS[0]
                    + f"Word: {row.word}\n"
                    f"Usage 1: {row.sentence1}\n"
                    f"Usage 2: {row.sentence2}\n"
                    f"Meaning:"
                ),
                self._token_set["few_shot"],
            ),
            "few_shot1": (
                lambda row: (
                    "Classify whether the following two sentences' use of the word has the same meaning or not.\n\n"
                    + "".join(SHOTS[:2])
                    + f"Word: {row.word}\n"
                    f"Usage 1: {row.sentence1}\n"
                    f"Usage 2: {row.sentence2}\n"
                    f"Meaning:"
                ),
                self._token_set["few_shot"],
            ),
            "few_shot2": (
                lambda row: (
                    "Classify whether the following two sentences' use of the word has the same meaning or not.\n\n"
                    + "".join(SHOTS[:3])
                    + f"Word: {row.word}\n"
                    f"Usage 1: {row.sentence1}\n"
                    f"Usage 2: {row.sentence2}\n"
                    f"Meaning:"
                ),
                self._token_set["few_shot"],
            ),
            "few_shot3": (
                lambda row: (
                    "Classify whether the following two sentences' use of the word has the same meaning or not.\n\n"
                    + "".join(SHOTS[:4])
                    + f"Word: {row.word}\n"
                    f"Usage 1: {row.sentence1}\n"
                    f"Usage 2: {row.sentence2}\n"
                    f"Meaning:"
                ),
                self._token_set["few_shot"],
            ),
            "question_answer_logic0": (
                lambda row: (
                    "".join(LOGIC_QUESTIONS[:1])
                    + f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'"{row.sentence1}"; "{row.sentence2}"\n'
                    f"A:"
                ),
                self._token_set["question"],
            ),
            "question_answer_logic1": (
                lambda row: (
                    "".join(LOGIC_QUESTIONS[:2])
                    + f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'"{row.sentence1}"; "{row.sentence2}"\n'
                    f"A:"
                ),
                self._token_set["question"],
            ),
            "question_answer_fact0": (
                lambda row: (
                    "".join(FACT_QUESTIONS[:1])
                    + f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'"{row.sentence1}"; "{row.sentence2}"\n'
                    f"A:"
                ),
                self._token_set["question"],
            ),
            "question_answer_fact1": (
                lambda row: (
                    "".join(FACT_QUESTIONS[:2])
                    + f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'"{row.sentence1}"; "{row.sentence2}"\n'
                    f"A:"
                ),
                self._token_set["question"],
            ),
            "question_answer_yes_no0": (
                lambda row: (
                    "".join(YES_NO_QUESTIONS[:1])
                    + f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'"{row.sentence1}"; "{row.sentence2}"\n'
                    f"A:"
                ),
                self._token_set["question"],
            ),
            "question_answer_yes_no1": (
                lambda row: (
                    "".join(YES_NO_QUESTIONS[:2])
                    + f'Q: Does the word "{row.word}" have the same meaning in the following sentences? '
                    f'"{row.sentence1}"; "{row.sentence2}"\n'
                    f"A:"
                ),
                self._token_set["question"],
            ),
            "true_false_classify1": (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}", '
                    f"true or false, "
                    f'the statement "the word {row.word} has the same meaning" is'
                ),
                self._token_set["true_false_classify"],
            ),
            "true_false_classify2": (
                lambda row: (
                    f'In the sentences "{row.sentence1}" and "{row.sentence2}" '
                    f'and choosing "true" or "false", '
                    f'the statement "the word {row.word} has the same meaning" is "'
                ),
                self._token_set["true_false_classify"],
            ),
            "true_false_classify3": (
                lambda row: (
                    f'"{row.sentence1}"\n'
                    f'"{row.sentence2}"\n\n'
                    f"True or false, the word {row.word} has the same meaning.\n"
                    f"Answer:"
                ),
                self._token_set["true_false_classify"],
            ),
            "true_false_classify4": (
                lambda row: (
                    f'"{row.sentence1}"\n'
                    f'"{row.sentence2}"\n\n'
                    f'"True" or "False", the word {row.word} has the same meaning.\n'
                    f'Answer: "'
                ),
                self._token_set["true_false_classify"],
            ),
            "true_false_classify5": (
                lambda row: (
                    f'"{row.sentence1}"\n'
                    f'"{row.sentence2}"\n\n'
                    f'True or False, the word "{row.word}" has the same meaning.\n'
                    f"Answer:"
                ),
                self._token_set["true_false_classify"],
            ),
            "true_false_classify6": (
                lambda row: (
                    f'True or False, the word "{row.word}" has the same meaning in the following sentences.\n\n'
                    f'Sentence 1: "{row.sentence1}"\n'
                    f'Sentence 2: "{row.sentence2}"\n\n'
                    f"Answer:"
                ),
                self._token_set["true_false_classify"],
            ),
            "true_false_classify7": (
                lambda row: (
                    "I am going to answer true or false questions about whether a word that appears in two sentences has the same meaning or not.\n\n"
                    f'True or False, the word "{row.word}" has the same meaning in the following sentences.\n\n'
                    f"Sentence 1: {row.sentence1}\n"
                    f"Sentence 2: {row.sentence2}\n"
                    f"Answer:"
                ),
                self._token_set["true_false_classify"],
            ),
            "wic_context": (
                lambda row: (
                    WIC_CONTEXT
                    + f"\n{row.sentence1} // {row.sentence2} // {row.word} //"
                ),
                self._token_set["few_shot"],
            ),
            "wikipedia_context": (
                lambda row: (
                    WIKIPEDIA_CONTEXT
                    + f"\n{row.sentence1} // {row.sentence2} // {row.word} //"
                ),
                self._token_set["few_shot"],
            ),
        }

        return templates


if __name__ == "__main__":
    # Data should be at data/wic/raw.csv
    wd = WicDataset()
