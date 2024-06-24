import pandas as pd
import os
import openai
from pdb import set_trace as breakpoint
import pickle
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('src')
from templatize import Templatizer

dataset = 'nyt'
exp_name = 'versions'
exp_dir = f"experiments/{dataset}/07-30-2021/{exp_name}"

if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

# Generate instructions for a task where a language model is supposed to categorize New York Times article headlines given a list of categories
prompts = {
    'v3.9-wrong_examples':{
        'suffix':'''\n"""\n\nThe world's worst article categorizer lost a lot of money by failing to categorize the following headlines with the above categories, making many mistakes in the process:
IRAN TURNS DOWN AMERICAN OFFER OF RELIEF MISSION -> Arts and Entertainment
 In Final Twist, Ill Pavarotti Falls Silent for Met Finale -> Macroeconomics
In Times Sq., a Dry Run for New Year's 2000 -> Energy
''',
    },
}


version = 'v3.9-wrong_examples'
templatizer = Templatizer('nytimes')
output = templatizer.templatize(n_per_category=4, n_exemplars=0, seed_instances=0, seed_exemplars=0, **prompts[version])
output['version'] = version + '-instruct'
output['model'] = 'davinci-instruct-beta'

responses = []

print(f'Generating Completions for {len(output)} instances')
for i, row in tqdm(output.iterrows()):
    response = None
    try:
        # take out '\n\n\n' from prompt
        prompt = row.prompt.replace('\n\n\n', '\n')
        response = openai.Completion.create(engine=row.model, prompt=prompt, max_tokens=1, logprobs=100)
    except Exception as exc:
        print(exc)
        pass
    responses.append(response)


output['responses'] = responses
pklpath = os.path.join(exp_dir,f'output7.pickle')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)