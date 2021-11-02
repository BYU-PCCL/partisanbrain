import pandas as pd
import numpy as np
from analysis import get_sorted_templates, compare_per_template
from tqdm import tqdm
from ensemble import get_ensemble_acc
import os
from pdb import set_trace as breakpoint

datasets = ['squad', 'rocstories', 'common_sense_qa', 'anes', 'boolq', 'imdb', 'copa', 'wic']
models = ['gpt3-davinci', 'gpt3-curie', 'gpt3-babbage', 'gpt-j', 'gpt-neo-2.7B', 'gpt3-ada', 'gpt2-xl', 'gpt2']

model_map = {
    'gpt3-davinci': 'GPT-3: 175B',
    'gpt3-curie': 'GPT-3: 13B',
    'gpt3-babbage': 'GPT-3: 6.7B',
    'gpt3-ada': 'GPT-3: 2.7B',
    'gpt-j': 'GPT-J: 6B',
    'gpt-neo-2.7B': 'GPT-Neo: 2.7B',
    'gpt2-xl': 'GPT-2: 1.5B',
    'gpt2': 'GPT-2: 124M',
}

dataset_map = {
    'anes': 'ANES',
    'boolq': 'BoolQ',
    'common_sense_qa': 'CommonsenseQA',
    'copa': 'COPA',
    'imdb': 'IMDB',
    'rocstories': 'ROCStories',
    'squad': 'SQuAD',
    'wic': 'WiC',
}

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
        files = [f for f in files if model in f and '_processed.pkl' in f]
        # if model is gpt2, filter so 'gpt2_' is only thing included
        if model == 'gpt2':
            files = [f for f in files if 'gpt2_' in f]
        file_path = files[0]
        # if length of files is more than 1, print the files and raise a warning
        if len(files) > 1:
            print(f'Multiple files found for {model} on {dataset}')
            print(files)
            print()
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
            print(file_name)
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
    dataset_names = [dataset_map[d] for d in datasets]
    model_names = [model_map[m] for m in models]
    # make df with datasets as rows and models as columns
    df = pd.DataFrame(dataset_dicts, index=dataset_names, columns=model_names)
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
    dataset_names = [dataset_map[d] for d in datasets]
    model_names = [model_map[m] for m in models]
    # make df with datasets as rows and models as columns
    df = pd.DataFrame(dataset_dicts, index=dataset_names, columns=model_names)
    # save to data/ensemle_data.pkl
    df.to_pickle('data/ensemble_data.pkl')
    print('Saved to data/ensemble_data.pkl')


if __name__ == '__main__':
    check_files_present()
    prep_scatter()
    # prep_ensemble()
