from rocstories import RocstoriesDataset
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
    # example usage
    print("Building dataset...")
    # build dataset
    RocstoriesDataset(n=500)
    # generate examples
    generate_examples('rocstories')