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
    'v0': {
        'prefix': '''Classify the following New York Times article headlines.
Use ONLY the following categories: ''',
        'per_cat_lambda': lambda x: f'"{x}"',
        'join_cats': ', ',
        'suffix': '\n',
        'join_inputs': '\n###\n',
        'input_lambda': lambda x: f'Article Headline: {x}',
        'join_input_category': '\nCategory:',

    },
    'v1':{
        'prefix':'''The New York Times is a well-respected news media organization. 
It is a major source of news content for the United States and many other countries. 
Articles produced by this organization can be categorized with high-level categorical descriptions.
Below, a comprehensive list of these descriptions is provided.
"""\n''',
        'suffix':'''\n"""\nUsing only one of the categories listed above, categorize the following examples of New York Times headlines with that category:''',
        },
    'v2':{
        'prefix':"""Categories: """,
        'suffix': '',
        'per_cat_lambda': lambda x: f'"{x}"',
        'input_lambda': lambda x: x,
        'join_cats': ', ',
        'join_input_category':'''\n\nThis article is the first in a series of article on the category of "''',
        },
    'v3':{
        'suffix':"""The following is a list of categories:""",
        'suffix':'''\n"""\n\nThe world's best article categorizer made a lot of money by expertly categorizing the following headlines with the above categories, making no mistakes in the process:''',
        },
}


templatizer = Templatizer('nytimes')
output0 = templatizer.templatize(n_per_category=4, n_exemplars=3, **prompts['v0'])
output0['version'] = 'v0-instruct'
output0['model'] = 'davinci-instruct-beta'
# print(output0.prompt.iloc[0])

templatizer = Templatizer('nytimes')
output1 = templatizer.templatize(n_per_category=4, n_exemplars=3, **prompts['v1'])
output1['version'] = 'v1-instruct'
output1['model'] = 'davinci-instruct-beta'
# print(output1.prompt.iloc[0])

templatizer = Templatizer('nytimes-body')
output2 = templatizer.templatize(n_per_category=4, n_exemplars=0, **prompts['v2'])
output2['version'] = 'v2-instruct'
output2['model'] = 'davinci-instruct-beta'
# print(output2.prompt.iloc[1])

templatizer = Templatizer('nytimes-body')
output3 = templatizer.templatize(n_per_category=4, n_exemplars=0, **prompts['v2'])
output3['version'] = 'v2'
output3['model'] = 'davinci'

templatizer = Templatizer('nytimes')
output4 = templatizer.templatize(n_per_category=4, n_exemplars=3, **prompts['v3'])
output4['version'] = 'v3-instruct'
output4['model'] = 'davinci-instruct-beta'
# print(output3.prompt.iloc[1])

# concatenate all the outputs
output = pd.concat([output0, output1, output2, output3, output4], ignore_index=True)

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