import sys
sys.path.append('src')
from time import time
import pandas as pd
import os
import openai
from pdb import set_trace as breakpoint
import pickle
from tqdm import tqdm
from templatize import Templatizer
import numpy as np
from datetime import date, timedelta
from readpickle import rpkl


dataset = 'nyt'
exp_name = 'gpt3_sizes'
exp_dir = f"experiments/{dataset}/07-30-2021/{exp_name}"

if os.path.isdir(exp_dir):
    pass
else:
    os.makedirs(exp_dir)

output = pd.DataFrame()

models = ['davinci', 'curie', 'babbage', 'ada', 'davinci-instruct-beta', 'curie-instruct-beta']
for model in models:
    templatizer = Templatizer(dataset_name='nytimes')
    o = templatizer.templatize(
        n_per_category=4,
        n_exemplars=3,
    )
    # add column 'model' to o
    o['model'] = model
    # append o to output
    output = output.append(o)

responses = []

print(f'Generating Completions for {len(output)} instances')
for i, row in tqdm(output.iterrows()):
    response = None
    try:
        response = openai.Completion.create(engine=row.model, prompt=row.prompt, max_tokens=1, logprobs=100)
    except Exception as exc:
        print(exc)
        pass
    responses.append(response)


output['responses'] = responses
pklpath = os.path.join(exp_dir,f'output.pickle')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)