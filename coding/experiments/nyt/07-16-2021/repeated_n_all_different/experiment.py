import pandas as pd
import os
import openai
from pdb import set_trace as breakpoint
import pickle
from tqdm import tqdm
import numpy as np
# add src to path
import sys
sys.path.append('src')
# import templatizing script
from templatize import Templatizer

# create templatizer
templatizer = Templatizer(dataset_name='nytimes')
# do for 10 runs
n_runs = 10
n_examples = 31

for run in range(n_runs):
    for n in range(31):
        exp_dir = os.path.join(f'experiments/nyt/07-16-2021/repeated_n_all_different/fewshot/{n}/run{run}')
        if os.path.isdir(exp_dir):
            pass
        else:
            os.makedirs(exp_dir)

        # different examples across instances
        prompts = templatizer.templatize(n_per_category=4, n_examples=n, seed_instances=run, seed_examples=lambda i: i*n_runs + run)

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
            with open(pklpath, 'wb') as f:
                pickle.dump(prompts, f)

