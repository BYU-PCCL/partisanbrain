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
exp_name = 'ambiguity'
today = date.today()
exp_dir = f"experiments/{dataset}/{today.strftime('%m-%d-%Y')}/{exp_name}"
pccfs_dir = '/mnt/pccfs2/backed_up/crytting/partisanbrain/coding'
exp_dir = os.path.join(pccfs_dir, exp_dir)

if os.path.isdir(exp_dir):
    pass
else:
    os.makedirs(exp_dir)

templatizer = Templatizer(dataset_name='nytimes')
output = templatizer.ambiguity_candidates()
breakpoint()

responses = []

print('Generating Completions')
for i, row in tqdm(output.iterrows()):
    response = None
    try:
        response = openai.Completion.create(engine="davinci", prompt=row.prompt, max_tokens=1, logprobs=100)
    except Exception as exc:
        print(exc)
        pass
    responses.append(response)

output['response'] = responses
pklpath = os.path.join(exp_dir,'output.pickle')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)