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
exp_name = 'EI'
today = date.today()
yesterday = today - timedelta(days=1)
exp_dir = f"experiments/{dataset}/{yesterday.strftime('%m-%d-%Y')}/{exp_name}"
pccfs_dir = '/mnt/pccfs2/backed_up/crytting/partisanbrain/coding'
exp_dir = os.path.join(pccfs_dir, exp_dir)

if os.path.isdir(exp_dir):
    pass
else:
    os.makedirs(exp_dir)

templatizer = Templatizer(dataset_name='nytimes')
output = templatizer.templatize_many(
    ns_per_category=[1],
    ns_exemplars=[0],
    n_exemplar_runs=1,
    n_instance_runs=1,
)
    # ns_per_category=[1, 2, 3, 4],
    # n_exemplar_runs=5,
    # n_instance_runs=5,

responses = []

print('Generating Completions')
for i, row in tqdm(output.iterrows()):
    response = None
    try:
        print(i, row)
        breakpoint()
        # response = openai.Completion.create(engine="davinci", prompt=row.prompt, max_tokens=1, logprobs=100)
    except Exception as exc:
        print(exc)
        pass
    responses.append(response)
    # if i % 1000 == 0 or i == len(output) - 1:
    #     pklpath = os.path.join(exp_dir,f'responses.pickle')
    #     with open(pklpath, 'wb') as f:
    #         pickle.dump(responses, f)

# Write output df to pickle
output['responses'] = responses
pklpath = os.path.join(exp_dir,'EI.pickle')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)
