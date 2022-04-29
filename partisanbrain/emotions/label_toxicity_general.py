import pandas as pd
from googleapiclient import discovery
from tqdm import tqdm
import time
import os
import pickle

"""Constants -- make sure to set the PERSPECTIVE_API_KEY in your terminal. Change N_DATASETS according to number of iterations per second."""

API_KEY = os.getenv("PERSPECTIVE_API_KEY")

if API_KEY is None:
    raise KeyError("Please set the PERSPECTIVE_API_KEY environment variable.")

N_DATASETS = 4

ATTRS = [
        "TOXICITY",
        "THREAT",
        "INSULT",
        "SEVERE_TOXICITY",
        "PROFANITY",
        "SEXUALLY_EXPLICIT",
        "IDENTITY_ATTACK",
        "FLIRTATION",
    ]

client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

def format_data(filename):
    """Abstract data from Real Toxicity Prompts by concatenating prompts and completions and 
       divide data for parallelizing scoring task when calling Perspective API"""

    df = pd.read_json(filename, lines=True)

    df["text"] = df["prompt"].apply(lambda text: text["text"]) + df["continuation"].apply(
        lambda text: text["text"]
        )

    df["text"].to_csv("texts.csv", index=False)

    n_texts = df.shape[0] // N_DATASETS
    for i in range(N_DATASETS):
        texts = df["text"].iloc[n_texts * i : n_texts * (i + 1)]
        texts.to_csv(f"texts_{i + 1}.csv", index=False)

def analyze_text(raw_text):
    """Calls API to score prompts"""

    tmp = {}
    for a in ATTRS:
        tmp[a] = {}
    analyze_request = {"comment": {"text": raw_text}, "requestedAttributes": tmp}
    response = client.comments().analyze(body=analyze_request).execute()
    return response
    
def process_texts(textfile):
    """Format text files into highest and lowest scores for each attribute."""

    prompts = pd.read_csv(textfile)

    responses = []
    score_dict = {a: [] for a in ATTRS}

    for i, prompt in tqdm(prompts.iterrows()):
        prompt = prompt["text"]

        try:
            resp = analyze_text(prompt)
            responses.append(resp)
            for a in ATTRS:
                score_dict[a].append(resp["attributeScores"][a]["summaryScore"]["value"])
            time.sleep(1 / (100 - 5))    # rate limit is 100 qps

        except Exception as e:
            print(f"Error processing [{prompt}]")
            responses.append(None)
            for a in ATTRS:
                score_dict[a].append(None)

    try:
        prompts["responses"] = responses
        for a in ATTRS:
            prompts[a] = score_dict[a]
        prompts.to_csv(f"processed_{textfile}", index=False)
        
    except:
        # Save responses as a pickle
        with open(f"responses_{textfile[:-3]}pkl", "wb") as f:
            pickle.dump(responses, f)

def feature_df():

    dfs = [pd.read_csv(f"processed_texts_{i}.csv") for i in range(1, N_DATASETS + 1)]
    df = pd.concat(dfs)

    for column in ATTRS:
        feature_df = df.sort_values(by=[column]).dropna()
        feature_df = feature_df[["text", column]]
        feature_df1 = feature_df.iloc[:3500]
        feature_df2 = feature_df.iloc[-3500:]
        feature_df = pd.concat([feature_df1, feature_df2])
        feature_df["label"] = feature_df[column].apply(round)
        feature_df.to_csv(f"{column.lower()}.csv", index=False)
    
if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    args = parser.parse_args()

    filename = args.data

    format_data(filename)

    with Pool(N_DATASETS) as p:
        p.map(process_texts, [f"texts_{i}.csv" for i in range(1, N_DATASETS + 1)])
    
    feature_df()