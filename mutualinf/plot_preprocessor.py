import pandas as pd
import numpy as np
from analysis import get_sorted_templates, compare_per_template
from tqdm import tqdm
from ensemble import get_ensemble_acc
import os
from pdb import set_trace as breakpoint

datasets = ['anes', 'boolq', 'common_sense_qa', 'copa', 'imdb', 'rocstories', 'squad', 'wic']
# datasets = ['anes',  'copa', 'imdb', 'rocstories', 'wic', 'squad']
# models = ['gpt3-davinci', 'gpt3-curie', 'gpt3-babbage', 'gpt3-ada', 'gpt-j', 'gpt-neo-2.7B', 'gpt2-xl', 'gpt2']
models = ['gpt3-davinci', 'gpt3-curie', 'gpt3-babbage', 'gpt-j', 'gpt-neo-2.7B', 'gpt3-ada', 'gpt2-xl', 'gpt2']

def check_files_present():
    '''
    Check if all files are present.
    '''
    present = True
    for dataset in datasets:
        for model in models:
            file_name = get_file(dataset, model)
            if file_name is None:
                present = False
                print(f'No file found for {model} on {dataset}')
    if present:
        print('All files present')
    else:
        raise Exception('Some files missing')

def get_file(dataset, model):
    path = f'data/{dataset}'
    # get all filenames in path
    files = os.listdir(path)
    try:
        # get the file with the model name in it AND '_processed.pkl' in it
        file_path = [f for f in files if model in f and '_processed.pkl' in f][0]
    except:
        # if no file found, return None and raise warning
        print(f'No file found for {model} on {dataset}')
        return None
    return os.path.join(path, file_path)

def prep_scatter():
    '''
    For each dataset and model, get the file and read in the df. Then, aggregate by 'template_name' and take the mean of 'accuracy' and 'mutual_inf' columns.
    '''
    print('Prepping data file for plots')
    loop = tqdm(total=len(datasets) * len(models))
    # make empty df with points. Columns are models, rows are datasets
    dataset_dicts = []
    for dataset in datasets:
        model_dicts = []
        for model in models:
            file_name = get_file(dataset, model)
            exp_df = pd.read_pickle(file_name)
            # aggregate by 'template_name' and take the mean of 'accuracy' and 'mutual_inf' columns
            exp_df = exp_df.groupby('template_name').agg({'accuracy': np.mean, 'mutual_inf': np.mean})
            # make 'template_name' (index) a column
            # exp_df.reset_index(inplace=True)
            # add to df
            model_dicts.append(exp_df)
            # increment loop
            loop.update(1)
        # add
        dataset_dicts.append(model_dicts)
    # make df with datasets as rows and models as columns
    df = pd.DataFrame(dataset_dicts, index=datasets, columns=models)
    # save to data/plot_data.pkl
    df.to_pickle('data/plot_data.pkl')
    print('Saved to data/plot_data.pkl')

# TODO - add ensemble prep here
def prep_ensemble(Ks=[1, 5, 20]):
    '''
    For each dataset and model, get the file and read in the df. Then, calculate the top k ensembles for all k.
    '''
    print('Prepping data file for ensemble')
    loop = tqdm(total=len(datasets) * len(models))
    # make empty df with points. Columns are models, rows are datasets
    dataset_dicts = []
    for dataset in datasets:
        model_dicts = []
        for model in models:
            file_name = get_file(dataset, model)
            exp_df = pd.read_pickle(file_name)
            n_ensembles = len(exp_df['template_name'].unique())
            acc_dict = {f'{k}_acc': get_ensemble_acc(exp_df, k) for k in Ks}
            # add avg accuracy to dict
            acc_dict['avg'] = exp_df['accuracy'].mean()
            model_dicts.append(acc_dict)
            # increment loop
            loop.update(1)
        # add
        dataset_dicts.append(model_dicts)
    # make df with datasets as rows and models as columns
    df = pd.DataFrame(dataset_dicts, index=datasets, columns=models)
    # save to data/ensemle_data.pkl
    df.to_pickle('data/ensemble_data.pkl')
    print('Saved to data/ensemble_data.pkl')


if __name__ == '__main__':
    check_files_present()
    prep_scatter()
    prep_ensemble()