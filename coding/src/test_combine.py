import pandas as pd
import pickle
import numpy as np
import os
import re

def get_files(path, ends_with='.pkl'):
    '''
    Get all files in a directory and all subdirectories that end with ends_with.
    Returns: list of paths to file
    '''
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(ends_with):
                files.append(os.path.join(dirpath, filename))
    return files

def read_pickle(path):
    '''
    Read pickle file at path.
    Returns: data from pickle file
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_values(s):
    '''
    Given a string of a file path, extract n, instance, and exemplar with a regex.
    Returns: n, instance, and exemplar
    Regex: *n_{n}_instance_{instance}_exemplar_{exemplar}*
    >>> extract_values('./data/n_100_instance_1_exemplar_1.pkl')
    (100, 1, 1)
    '''
    regex = re.compile(r'n_(\d+)_instance_(\d+)_exemplar_(\d+)')
    n, instance, exemplar = regex.search(s).groups()
    return int(n), int(instance), int(exemplar)



if __name__ == '__main__':
    files = get_files('experiments/nyt/07-06-21/slash/n_4_instance_0_exemplar_0/', ends_with='.p')
    # print(files)
    # iterate through files and get n, instance, and exemplar
    n_instances_exemplars = []
    dataframes = []
    breakpoint()
    for f in files:
        n, instance, exemplar = extract_values(f)
        n_instances_exemplars.append((n, instance, exemplar))
        df = pd.DataFrame(read_pickle(f))
        df['n'] = n
        df['instance'] = instance
        df['exemplar'] = exemplar
        dataframes.append(df)
    
    # combine dataframes
    if len(dataframes) > 1:
        df = pd.concat(dataframes)
    else:
        df = dataframes[0]
    # pickle df as 'experiments/nyt/nyt_combined.pkl'
    breakpoint()
    df.to_pickle('experiments/nyt/nyt_combined.pkl')

    # iterate through files, read file, and append to df
    n_instances_exemplars = []
    for f in files:
        n, instance, exemplar = extract_values(f)
        n_instances_exemplars.append((n, instance, exemplar))
    
    print(files[0])
    print(extract_values(files[0]))
    # read in a pickle file
    data = read_pickle(files[0])
    pass