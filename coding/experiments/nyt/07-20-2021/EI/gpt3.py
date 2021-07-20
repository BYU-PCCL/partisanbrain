import pandas as pd
import os
import openai
from pdb import set_trace as breakpoint
import pickle
from tqdm import tqdm
from templatize import Templatizer
import numpy as np
from datetime import date
from readpickle import rpkl

dataset = 'nyt'
exp_name = 'EI'
today = date.today()
exp_dir = os.path.join(f'experiments/{dataset}/{today.strftime("%m-%d-%Y")}/{exp_name}')

if os.path.isdir(exp_dir):
    pass
else:
    os.makedirs(exp_dir)

templatizer = Templatizer(dataset_name='nytimes')
output = templatizer.templatize_many(
    ns_per_category=[1, 2, 3, 4],
    ns_exemplars=[1, 2, 3, 4, 5],
    n_exemplar_runs=5,
    n_instance_runs=5,
)
# output = templatizer.templatize_many()

responses = []

print('Generating Completions')
for i, row in tqdm(output.iterrows()):
    try:
        response = None
        response = openai.Completion.create(engine="davinci", prompt=row.prompt, max_tokens=1, logprobs=100)
        responses.append(response)

        pklpath = os.path.join(exp_dir,'responses.p')
        with open(pklpath, 'wb') as f:
            pickle.dump(responses, f)

    except Exception as exc:
        print(exc)
        pass

# Write output df to pickle
output['responses'] = responses
pklpath = os.path.join(exp_dir,'EI.p')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)