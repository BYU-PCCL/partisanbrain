import pandas as pd

import numpy as np
from tqdm import tqdm
from googleapiclient import discovery
import json
import time
import pickle
import sys
from pdb import set_trace as bp
import os


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

API_KEY = os.getenv("PERSPECTIVE_API_KEY")
if API_KEY == None:
    print("Please set the PERSPECTIVE_API_KEY environment variable")
    sys.exit(1)

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)


def analyze_text(raw_text):
    tmp = {}
    for a in ATTRS:
        tmp[a] = {}

    analyze_request = {"comment": {"text": raw_text}, "requestedAttributes": tmp}

    response = client.comments().analyze(body=analyze_request).execute()
    return response


#
# ---------------------------------------------------------------------------------
#

np.random.seed(42)

prompts_fn = sys.argv[1]
prompts = pd.read_csv(prompts_fn)

full_data = []
responses = []
score_dict = {a: [] for a in ATTRS}

for i, prompt in tqdm(prompts.iterrows()):
    prompt = prompt["text"]

    try:
        resp = analyze_text(prompt)
        responses.append(resp)
        for a in ATTRS:
            score_dict[a].append(resp["attributeScores"][a]["summaryScore"]["value"])
        time.sleep(1 / (100 - 5))  # rate limit is 100 qps
    except Exception as e:
        print(f"Error processing [{prompt}]")
        responses.append(None)
        for a in ATTRS:
            score_dict[a].append(None)
        # continue
        # prob = resp['attributeScores'][a]['summaryScore']['value']


try:
    prompts["responses"] = responses
    for a in ATTRS:
        prompts[a] = score_dict[a]
    prompts.to_csv(f"processed_{prompts_fn}", index=False)
except:
    # Save responses as a pickle
    with open(f"responses_{prompts_fn[:-3]}pkl", "wb") as f:
        pickle.dump(responses, f)


def score_df(df, attribute):
    scores = []
    responses = []

    for i, row in tqdm(df.iterrows()):
        sentence = row["sentence"]

        try:
            resp = analyze_text(sentence)
            responses.append(resp)
            scores.append(resp["attributeScores"][attribute]["summaryScore"]["value"])
            time.sleep(1 / (100 - 5))  # rate limit is 100 qps
        except Exception as e:
            print(f"Error processing [{prompt}]")
            responses.append(None)
            scores.append(None)

    df["responses"] = responses
    df[attribute] = scores
    df["label"] = df[attribute].apply(round)
    return df
