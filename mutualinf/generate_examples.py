from rocstories import RocstoriesDataset
from boolq import BoolqDataset
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
    prompts = df.groupby('template_name').agg({'prompt': 'first'})

    # check if 'examples' directory exists, if not, create it
    if not os.path.exists('examples'):
        os.makedirs('examples')
   
    # write out examples to examples/{ds_name}.txt
    out_file = f"examples/{ds_name}.txt"
    with open(out_file, 'w') as f:
        for _, row in prompts.iterrows():
            f.write(f'"{row.name}":\n')
            f.write(f'<<<{row.prompt}>>>\n')
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
        ds = RocstoriesDataset(n=500)
    elif dataset == 'boolq':
        ds = BoolqDataset(n=500)
    else:
        raise ValueError(f'Dataset {dataset} not supported. Modify generate_examples.py to add support (should be easy).')

    generate_examples(dataset, shuffle=shuffle)