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
n_runs = 5
n_exemplars = 31

for i in range(n_runs):
    for j in range(n_runs):
        for n in range(n_exemplars):
            exp_dir = os.path.join(f'experiments/nyt/07-16-2021/repeated_n_all_same/n_{n}_instance_{i}_exemplar_{j}')
            if os.path.isdir(exp_dir):
                pass
            else:
                os.makedirs(exp_dir)

            # hold examples constant across instances
            prompts = templatizer.templatize(n_per_category=4, n_exemplars=n, seed_instances=i, seed_exemplars=j)

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

