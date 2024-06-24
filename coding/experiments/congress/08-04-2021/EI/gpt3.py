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


dataset = 'congress'
exp_name = 'EI'
exp_dir = f"experiments/{dataset}/08-04-2021/{exp_name}"

if os.path.isdir(exp_dir):
    pass
else:
    os.makedirs(exp_dir)

templatizer = Templatizer(dataset_name='nytimes')
output = templatizer.templatize_many(
    ns_per_category=[4],
    ns_exemplars=[0, 1, 2, 3, 4, 5],
    n_exemplar_runs=5,
    n_instance_runs=5,
)

responses = []

print('Generating Completions')
loop = tqdm(total=len(output))
for i, row in output.iterrows():
    response = None
    try:
        # print(i, row)
        response = openai.Completion.create(engine="davinci", prompt=row.prompt, max_tokens=1, logprobs=100)
    except Exception as exc:
        print(exc)
        pass
    responses.append(response)
    loop.update(1)

# Write output df to pickle
output['responses'] = responses
pklpath = os.path.join(exp_dir,'output.pickle')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)

breakpoint()
pass