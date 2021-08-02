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
    'v3.3':{
        'suffix':'''\n"""\n\nThe world's best article categorizer made a lot of money by expertly categorizing the following headlines with the above categories, making no mistakes in the process. Every article was categorized perfectly:''',
    },
    'v3.4':{
        'suffix':'''\n"""\n\nThe world's best article categorizer made a lot of money by expertly categorizing the following headlines with the above categories, making no mistakes in the process. Every article was categorized perfectly, with no room for error. This is the correct answer key of all answer keys for this task:''',
    },
}


outputs = []
for version in prompts:
    templatizer = Templatizer('nytimes')
    output = templatizer.templatize(n_per_category=4, n_exemplars=3, **prompts[version])
    output['version'] = version + '-instruct'
    output['model'] = 'davinci-instruct-beta'
    outputs.append(output)

# concatenate all the outputs
output = pd.concat(outputs)

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
pklpath = os.path.join(exp_dir,f'output3.pickle')
with open(pklpath, 'wb') as f:
    pickle.dump(output, f)