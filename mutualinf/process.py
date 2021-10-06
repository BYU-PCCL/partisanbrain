from collections import defaultdict

import openai
import os

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


TEMPLATES = {
    "template_1": lambda row: f"I am {row['age']} and I am {row['party']}. Between cat and bird I like",
    "template_2": lambda row: f"I am {row['age']} and I am {row['party']}. Between cat and bird I love"
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
    allowed_vals = [item.strip().lower() for sublist in
                    list(categories_key_sets.values())
                    for item in sublist]
    print(allowed_vals)
    print(probs_dict)
    relevant = {k: v for (k, v) in probs_dict.items()
                if k.strip().lower() in allowed_vals}
    print(relevant)


def process_df(df,
               gpt_3_engine,
               template_names,
               dv_col_name,
               categories_key_sets=None):

    new_df_dict = defaultdict(list)

    categories = df[dv_col_name].unique().tolist()

    if categories_key_sets is None:
        categories_key_sets = dict(zip(categories, [[c] for c in categories]))

    for _, row in df.iterrows():

        for template_name in template_names:

            new_df_dict["template_name"].append(template_name)

            prompt = make_prompt(row, template_name)
            new_df_dict["prompt"].append(prompt)
            new_df_dict["ground_truth"].append(row[dv_col_name])
            new_df_dict["categories"].append(categories)

            response = process_prompt(prompt, gpt_3_engine)
            extract_category_probs(response, categories_key_sets)
            import sys  # DELETE
            sys.exit()  # DELETE

            try:
                response = process_prompt(prompt, gpt_3_engine)
                new_df_dict["response"].append(response)
            except Exception as e:
                print("Exception in process_df:", e)
                new_df_dict["response"].append(None)

    return pd.DataFrame(new_df_dict)


if __name__ == "__main__":
    import pandas as pd
    toy_df = pd.DataFrame(dict(
        age=[1, 2, 3],
        party=["r", "d", "r"],
        dv=["bird", "cat", "bird"]
    ))

    print(process_df(toy_df, "ada", TEMPLATES.keys(), "dv", {"bird": ["bird", "parrot"], "cat": ["cat"]}))
