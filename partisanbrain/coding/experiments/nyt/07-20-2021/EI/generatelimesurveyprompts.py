import sys
sys.path.append('src')
import pandas as pd
from pdb import set_trace as breakpoint
from datetime import date

dataset = 'nyt'
exp_name = 'EI'
# Create a date object for july 20th, 2021
day = date(2021, 7, 20)
exp_dir = f"experiments/{dataset}/{day.strftime('%m-%d-%Y')}/{exp_name}"

df = pd.read_pickle(f"{exp_dir}/EIwhole.pkl")
lsdf = df[(df.n_exemplars == 3) & (df.exemplar_set_ix == 0) & (df.n_per_category == 4)].set_index('level_0')