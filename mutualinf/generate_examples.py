import pandas as pd
import os

def generate_examples(ds_name, shuffle=False, out_file=None):
    # read in dataframe
    ds_fname = f"data/{ds_name}/ds.pkl"
    df = pd.read_pickle(ds_fname)
    # groupby 'template_name' and keep first for column 'prompt'
    # shuffle df
    if shuffle:
        df = df.sample(frac=1)
    prompts = df.groupby('template_name').agg({'prompt': 'first', 'token_sets': 'first'})
    # print the number of prompts
    print(f"{ds_name}: {len(prompts)} prompts")

    # check if 'examples' directory exists, if not, create it
    if not os.path.exists('examples'):
        os.makedirs('examples')
   
    # write out examples to examples/{ds_name}.txt
    out_file = f"examples/{ds_name}.txt"
    with open(out_file, 'w') as f:
        for _, row in prompts.iterrows():
            f.write(f'"{row.name}":\n')
            f.write(f'<<<{row.prompt}>>>\n')
            # write token sets
            f.write(f"{row.token_sets}\n")
            f.write('\n\n')
    print(f'Saved to {out_file}')
    
if __name__ == '__main__':
    import sys
    # first argument is name of dataset
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        # second argument is shuffle
        shuffle = sys.argv[2]
    else:
        shuffle = False
    
    # first, build dataset
    if dataset == 'rocstories':
        from rocstories import RocstoriesDataset
        ds = RocstoriesDataset(n=500)
    elif dataset == 'boolq':
        from boolq import BoolqDataset
        ds = BoolqDataset(n=500)
    elif dataset == 'copa':
        from copa import CopaDataset
        ds = CopaDataset(n=500)
    elif dataset == 'wic':
        from wic import WicDataset
        ds = WicDataset(n=500)
    elif dataset == 'anes':
        from anes import AnesDataset
        ds = AnesDataset(n=500)
    elif dataset == 'squad':
        from squad import SquadDataset
        ds = SquadDataset(n=500)
    elif dataset == 'common_sense_qa':
        from common_sense_qa import CommonSenseQaDataset
        ds = CommonSenseQaDataset(n=500)
    elif dataset == 'lambada':
        from lambada import LambadaDataset
        ds = LambadaDataset(n=500)
    else:
        raise ValueError(f'Dataset {dataset} not supported. Modify generate_examples.py to add support (should be easy).')

    generate_examples(dataset, shuffle=shuffle)