import pandas as pd
from plot_preprocessor import get_file, dataset_map
import os

datasets = ['squad', 'rocstories', 'common_sense_qa', 'anes', 'boolq', 'imdb', 'copa', 'wic']
models = ['gpt3-davinci']

def get_str(dataset, model):
    '''
    Make a string with the mutual inf, accuracy, and prompt
    '''
    file_name = get_file(dataset, model)
    df = pd.read_pickle(file_name)
    # group by 'template_name', agg accuracy, mutualinf, and prompt
    templates = df.groupby('template_name').agg({'accuracy': 'mean', 'mutual_inf': 'mean', 'prompt': 'first'})
    # sort descending by mutualinf
    templates = templates.sort_values('mutual_inf', ascending=False)

    s = ''
    for i, (index, row) in enumerate(templates.iterrows()):
        s += '\\textbf{'
        s += 'Prompt {} (Mutual Information: {:.3f}, Accuracy: {:.3f}):'.format(i+1, row['mutual_inf'], row['accuracy'])
        s += '}\n'
        s += row['prompt'] + '\n\n'
    # convert to ascii
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s

def save_prompts(dataset, model, output_file=''):
    # get str
    s = get_str(dataset, model)
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
    
def combine_all_prompts():
    # check if folder 'prompts' exists, else create it
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
    s = ''
    for dataset in datasets:
        for model in models:
            s += '\subsubsection{' + dataset_map[dataset] + '}\n'
            s += get_str(dataset, model)
    # convert to ascii
    s = s.encode('ascii', 'ignore').decode('ascii')
    # write s to output file
    with open('prompts/all.txt', 'w') as f:
        f.write(s)



if __name__ == '__main__':
    # save_all_prompts()
    combine_all_prompts()