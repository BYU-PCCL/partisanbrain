import pandas as pd
import os
import openai
from pdb import set_trace as breakpoint
import pickle
from tqdm import tqdm
#from framingtemplatize import gen_prompts 
#from templatize_nytimes import templatize
from templatize_congress import templatize
import numpy as np

# Generate instructions for a task where a language model is supposed to categorize New York Times article headlines given a list of categories
prompts = {
    'v1':{
        'prime':"""The New York Times is a well-respected news media organization. 
It is a major source of news content for the United States and many other countries. 
Articles produced by this organization can be categorized with high-level categorical descriptions.
Below, a comprehensive list of these descriptions is provided.
""",
        'instructions':"""Using only one of the categories listed above, categorize the following examples of New York Times headlines with that category:"""},
    'v2':{
        'prime':"""Categories:""",
        'instructions':"""This article is the first in a series of article on the category of"""},
    'v3':{
        'prime':"""The following is a list of categories:""",
        'instructions':"""The world's best article categorizer made a lot of money by expertly categorizing the following headlines with the above categories, making no mistakes in the process:"""},
}

for n in np.arange(21):
    exp_dir = os.path.join(f'experiments/congress/07-09-21/fewshot/{n}')
    if os.path.isdir(exp_dir):
        pass
    else:
        os.makedirs(exp_dir)

    prompts = templatize(n=5, n_examples=n)

    responses = []
    for prompt in tqdm(prompts):
        try:
            response = None
            response = openai.Completion.create(engine="davinci", prompt=prompt['text'], max_tokens=2, logprobs=100)
            prompt['response'] = response
            pklpath = os.path.join(exp_dir,'prompts.p')
            with open(pklpath, 'wb') as f:
                pickle.dump(prompts, f)
        except Exception as exc:
            print(exc)
            pass

