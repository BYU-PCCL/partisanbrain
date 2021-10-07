from collections import defaultdict

import numpy as np
import openai
import os

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


TEMPLATES = {
    "template_1": lambda row: f"I am {row['food']} and I am {row['age']}. Between carrots and bacon I prefer",
    "template_2": lambda row: f"I am {row['age']} and I am {row['food']}. Between carrots and bacon I prefer"
}


def make_prompt(row, template_name):
    return f"{TEMPLATES[template_name](row)}"


def process_prompt(prompt, gpt_3_engine):
    assert gpt_3_engine in ["ada", "babbage", "curie", "davinci"]
    return openai.Completion.create(engine=gpt_3_engine,
                                    prompt=prompt,
                                    max_tokens=1,
                                    logprobs=100)


def extract_category_probs(gpt_3_response, categories_key_sets):
    probs_dict = gpt_3_response["choices"][0]["logprobs"]["top_logprobs"][0]

    # Do softmax
    probs_dict_exp = {k: np.exp(v) for (k, v) in probs_dict.items()}

    score_dict = {k: 0 for k in categories_key_sets.keys()}

    # Only take the keys from probs_dict_exp that match allowed values
    # and aggregate
    for prob_key, prob in probs_dict_exp.items():
        prob_key = prob_key.strip().lower()
        for category_name in score_dict:
            for category_key in categories_key_sets[category_name]:
                if category_key.strip().lower().startswith(prob_key):
                    score_dict[category_name] += prob

    return score_dict


def process_df(df,
               gpt_3_engine,
               template_names,
               dv_col_name,
               categories_key_sets=None):

    new_df_dict = defaultdict(list)

    if categories_key_sets is None:
        categories = df[dv_col_name].unique().tolist()
        categories_key_sets = dict(zip(categories, [[c] for c in categories]))
    else:
        categories = list(categories_key_sets.keys())

    for _, row in df.iterrows():

        for template_name in template_names:

            new_df_dict["template_name"].append(template_name)

            prompt = make_prompt(row, template_name)
            new_df_dict["prompt"].append(prompt)
            new_df_dict["categories"].append(categories)
            new_df_dict["ground_truth"].append(row[dv_col_name])

            response = process_prompt(prompt, gpt_3_engine)
            category_probs = extract_category_probs(response,
                                                    categories_key_sets)

            coverage = sum(category_probs.values())

            # Normalize category_probs
            category_probs = {k: v / coverage
                              for (k, v) in category_probs.items()}

            try:
                response = process_prompt(prompt, gpt_3_engine)
                new_df_dict["response"].append(response)
            except Exception as e:
                print("Exception in process_df:", e)
                new_df_dict["response"].append(None)

            new_df_dict["coverage"].append(coverage)

            for category in categories:
                new_df_dict[category].append(category_probs[category])

    return pd.DataFrame(new_df_dict)


if __name__ == "__main__":
    import pandas as pd
    toy_df = pd.DataFrame(dict(
        age=["male", "male", "female"],
        food=["vegan", "a meat lover", "vegan"],
        dv=["carrots", "bacon", "carrots"]
    ))

    df = process_df(toy_df, "davinci", TEMPLATES.keys(), "dv", {"bacon": ["bacon", "meat"], "carrots": ["carrots"]})
    df.to_csv("temp.csv", index=False)
