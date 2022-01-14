import pandas as pd
import numpy as np
from analysis import get_sorted_templates, compare_per_template
from tqdm import tqdm
from ensemble import get_ensemble_acc
import os
from pdb import set_trace as breakpoint

# datasets = ['squad', 'rocstories', 'common_sense_qa', 'anes', 'boolq', 'imdb', 'copa', 'wic']
datasets = ['squad', 'lambada', 'rocstories', 'common_sense_qa', 'imdb', 'boolq', 'copa', 'wic']
# models = ['gpt3-davinci', 'gpt3-curie', 'gpt3-babbage', 'gpt-j', 'gpt-neo-2.7B', 'gpt3-ada', 'gpt2-xl', 'gpt2']
models = ['gpt3-davinci']

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
    'lambada': 'LAMBADA',
    'boolq': 'BoolQ',
    'common_sense_qa': 'CoQA',
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

def prob_dict_to_arr(d):
    '''
    Converts a probability dictionary into an array of probabilities.
    Args:
        d (dict: str->float): dictionary of probabilities for each category/token.
    Returns:
        arr (np.array): array of probabilities.
    '''
    arr = np.array([d[k] for k in d])
    return arr

def agg_prob_dicts(dicts):
    '''
    Given a list of probability dictionaries, aggregate them.
    '''
    n = len(dicts)
    agg_dict = {}
    for d in dicts:
        for k, v in d.items():
            if k not in agg_dict:
                agg_dict[k] = v / n
            else:
                agg_dict[k] += v / n
    return agg_dict

def entropy(arr):
    '''
    Given an array of probabilities, calculate the entropy.
    '''
    return -sum(arr * np.log(arr))

def calculate_conditional_entropy(df):
    '''
    Calculates the conditional entropy, up to a constant. Adds a column called 'conditional_entropy' to df.
    df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

    Returns modified df.
    '''
    df = df.copy()

    entropy_lambda = lambda row: entropy(prob_dict_to_arr(row['probs']))

    # Calculate entropy for each row
    df['conditional_entropy'] = df.apply(entropy_lambda, axis=1)

    return df

def get_marginal_distribution(df, groupby='template_name'):
    '''
    Calculates the marginal distribution over categories.
    '''
    marginal_df = df.groupby(by=groupby)['probs'].agg(agg_prob_dicts)
    # series to df
    marginal_df = pd.DataFrame(marginal_df)
    return marginal_df

def calculate_mutual_information(df, groupby='template_name'):
    '''
    Calculate the mutual information between the template and the output distribution.
    '''
    # H(Y) - H(Y|X) method
    # first, calculate conditional entropy
    df = calculate_conditional_entropy(df)
    # get marginal distributions
    marginal_df = get_marginal_distribution(df, groupby)
    # get entropy
    entropy_lambda = lambda row: entropy(prob_dict_to_arr(row['probs']))
    marginal_df['entropy'] = marginal_df.apply(entropy_lambda, axis=1)
    # function to apply per row
    def mutual_inf(row):
        index = row[groupby]
        mutual_info = marginal_df.loc[index]['entropy'] - row['conditional_entropy']
        return mutual_info

    # apply function to each row
    df['mutual_inf'] = df.apply(mutual_inf, axis=1)

    return df

def run_baseline(df, n, random_seed):
    # get argmax template of mutual information
    full_mi_choice = df.groupby('template_name').mutual_inf.mean().idxmax()
    full_mi_acc = df.loc[df['template_name'] == full_mi_choice]['accuracy'].mean()
    # partiton into train and test, with n in train set
    indices = df.raw_idx.unique()
    # set random seed
    np.random.seed(random_seed)
    train_indices = np.random.choice(indices, n, replace=False)
    test_indices = [i for i in indices if i not in train_indices]
    train_df = df[df.raw_idx.isin(train_indices)]
    # calculate mutual information on train_df
    train_df = calculate_mutual_information(train_df)
    test_df = df[df.raw_idx.isin(test_indices)]

    # get template with highest accuracy in train_df
    # aggregate by 'template_name' and take the mean of 'accuracy'
    train_acc = train_df.groupby('template_name').accuracy.mean()
    test_acc = test_df.groupby('template_name').accuracy.mean()
    # get the template with the highest accuracy
    max_acc = train_acc.max()
    # filter to only those with max accuracy
    max_acc_df = train_acc[train_acc == max_acc]
    # pick random
    max_acc_df = max_acc_df.sample(n=1, random_state=random_seed)
    baseline_choice = max_acc_df.index[0]

    # get the template with the highest mutual information
    mi_choice = train_df.groupby('template_name').mutual_inf.mean().idxmax()

    # baseline_acc, mi_acc, full_mi_acc = test_acc[baseline_choice], test_acc[mi_choice], test_acc[full_mi_choice]
    baseline_acc, mi_acc = test_acc[baseline_choice], test_acc[mi_choice]
    return baseline_acc, mi_acc, full_mi_acc


def baseline(N = [2, 4, 8, 16, 32, 64, 128, 256], K=10):
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
            all_baselines, all_mis, all_full_mis = [], [], []
            for n in N:
                baselines, mis, full_mis = [], [], []
                for k in range(K):
                    # run baseline
                    baseline_acc, mi_acc, full_mi_acc = run_baseline(exp_df, n=n, random_seed=(k+K)*n)
                    # append
                    baselines.append(baseline_acc)
                    mis.append(mi_acc)
                    full_mis.append(full_mi_acc)
                all_baselines.append(baselines)
                all_mis.append(mis)
                all_full_mis.append(full_mis)
            # put into dataframe, with rows as N, columns baseline, mi, full_mi
            df = pd.DataFrame({'baseline': all_baselines, 'mi': all_mis, 'full_mi': all_full_mis})
            # change index to N
            df.index = N
            # add to df
            model_dicts.append(df)
            # increment loop
            loop.update(1)
        # add
        dataset_dicts.append(model_dicts)
    dataset_names = [dataset_map[d] for d in datasets]
    model_names = [model_map[m] for m in models]
    # make df with datasets as rows and models as columns
    df = pd.DataFrame(dataset_dicts, index=dataset_names, columns=model_names)
    # save to data/baseline_data.pkl
    df.to_pickle('data/baseline_data.pkl')
    print('Saved to data/baseline_data.pkl')


if __name__ == '__main__':
    check_files_present()
    baseline(N = [2, 4, 8, 16, 32, 64, 128, 256], K=100)
    # prep_ensemble()
