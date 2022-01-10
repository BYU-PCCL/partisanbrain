import pandas as pd
from pdb import set_trace as breakpoint

# plot_data = pd.read_csv('data/plot_data.pkl')

model = 'GPT3-175B'
file_path = 'data/rocstories_sample/exp_results_gpt3-davinci_09-11-2021_processed.pkl'
df = pd.read_pickle(file_path)
# group by template_name and take mean of accuracy
df = df.groupby(['template_name']).agg({'accuracy': 'mean'}).reset_index()
breakpoint()