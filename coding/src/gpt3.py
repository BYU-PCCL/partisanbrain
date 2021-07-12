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

