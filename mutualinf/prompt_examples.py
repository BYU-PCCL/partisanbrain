import pandas as pd
from plot_preprocessor import get_file
import os

datasets = ['squad', 'rocstories', 'common_sense_qa', 'anes', 'boolq', 'imdb', 'copa', 'wic']
models = ['gpt3-davinci']

def get_str(templates):
    '''
    Make a string with the mutual inf, accuracy, and prompt
    '''
    s = ''
    for i, (index, row) in enumerate(templates.iterrows()):
        s += 'Prompt {} (Mutual Information: {:.3f}, Accuracy: {:.3f}):\n'.format(i+1, row['mutual_inf'], row['accuracy'])
        s += row['prompt'] + '\n\n'
    # convert to ascii
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s

def save_prompts(dataset, model, output_file=''):
    file_name = get_file(dataset, model)
    df = pd.read_pickle(file_name)
    # group by 'template_name', agg accuracy, mutualinf, and prompt
    templates = df.groupby('template_name').agg({'accuracy': 'mean', 'mutual_inf': 'mean', 'prompt': 'first'})
    # sort descending by mutualinf
    templates = templates.sort_values('mutual_inf', ascending=False)
    
    # get str
    s = get_str(templates)
    # if output_file is none, make prompts/dataset.txt
    if output_file == '':
        output_file = os.path.join('prompts', dataset + '.txt')
    # write s to output file
    with open(output_file, 'w') as f:
        f.write(s)

def save_all_prompts():
    # check if folder 'prompts' exists, else create it
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
    for dataset in datasets:
        for model in models:
            save_prompts(dataset, model)


if __name__ == '__main__':
    save_all_prompts()