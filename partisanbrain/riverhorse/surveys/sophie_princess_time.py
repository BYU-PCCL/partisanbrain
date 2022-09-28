from transformers import T5Tokenizer, T5ForConditionalGeneration
from .anes_survey_sophie import for_chris, fine_tuned_phrasing, AnesSurveySophie
from tqdm import tqdm

model_name = "allenai/unifiedqa-v2-t5-11b-1251000"  # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


if __name__ == "__main__":
    s = AnesSurveySophie()
    df = s.download_data()
    df_mod = s.modify_data(df)
    # df_mod = df_mod.sample(10)
    keeping_track = {'donald trump': {'y_hat': 0, 'y_true': 0, 'acc': 0}, 'hillary clinton': {
        'y_hat': 0, 'y_true': 0, 'acc': 0}, 'other \{specify\}': {'y_hat': 0, 'y_true': 0, 'acc': 0}}

    # prompt = (f"Four years ago, in 2016, Hillary Clinton ran on the Democratic ticket against Donald Trump for the Republicans. Which candidate did the following voter vote for? \\n"
    # f" (A) donald trump (B) hillary clinton (C) other \\n "
    # f"The voter is a 60-year-old christian white man and a republican.")
    # result = run_model(prompt)
    # print(result)
    for _, row in tqdm(df_mod.iterrows()):
        # prompt = for_chris(row)
        prompt = fine_tuned_phrasing(row)
        result = run_model(prompt, max_length=1000)
        y_hat = result[0]
        # print(prompt, result, row['vote_2016'])
        # print("\n\n")
        y_true = row['vote_2016'].lower()
        keeping_track[y_true]['y_true'] += 1
        keeping_track[y_hat]['y_hat'] += 1
        if y_hat == y_true:
            keeping_track[y_true]['acc'] += 1

    print(keeping_track)