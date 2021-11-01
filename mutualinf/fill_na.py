import pandas as pd
import numpy as np
from tqdm import tqdm
from postprocessor import get_files_to_process, Postprocessor
from lmsampler import LMSampler
import os
from pdb import set_trace as breakpoint

model_dict = {
    'EleutherAI-gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'EleutherAI-gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
}


def fill_na(file_name):
    '''
    Read in the pickle, check if there are any na values in 'resp', and fill them in accordingly.
    Finally, save the updated file.
    '''
    # read in file
    df = pd.read_pickle(file_name)
    # check if any na
    if not df['resp'].isna().sum():
        print('No Na values in resp')
        return
    # dataset is <dataset> in data/<dataset>/*.pkl in file_name
    dataset = file_name.split('/')[-2]
    # model is <model> in data/<dataset>/exp_results_<model>_*.pkl
    model_name = file_name.split('/')[-1].split('_')[-2]
    if model_name in model_dict:
        model_name = model_dict[model_name]

    model = LMSampler(model_name)
    
    def fill_row(row):
        # if 'resp' is not na, return resp
        if row['resp']:
            return row['resp']
        # else, run through model
        try:
            resp = model.send_prompt(
                row["prompt"],
                n_probs=100, # TODO - make this not magic number
            )
            return resp
        except Exception as e:
            print(e)
            breakpoint()
            return None
    
    # fill na
    df['resp'] = df.apply(fill_row, axis=1)

    # save
    df.to_pickle(file_name)

    # post process

    input_fname = file_name
    # save_fname is same name, but replace .pkl with _processed.pkl
    save_fname = input_fname.replace('.pkl', '_processed.pkl')

    # process
    Postprocessor(input_fname, save_fname)
    


def fill_all_na():
    files = get_files_to_process()
    for file_name in tqdm(files):
        print(file_name)
        fill_na(file_name)

files_to_fill = [
    # 'data/common_sense_qa/exp_results_gpt3-ada_25-10-2021.pkl',
    'data/imdb/exp_results_gpt2-xl_21-10-2021.pkl',
    'data/imdb/exp_results_gpt2-xl_18-10-2021.pkl',
    'data/imdb/exp_results_gpt2_22-10-2021.pkl',

]

if __name__ == '__main__':
    import sys
    # check if first arg
    if len(sys.argv) > 1:
        fill_na(sys.argv[1])
    else:
        # fill_all_na()
        # fill in all files_to_fill
        for file_name in files_to_fill:
            print(file_name)
            fill_na(file_name)