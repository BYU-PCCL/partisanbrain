import pandas as pd
from plot_preprocessor import get_file, dataset_map
import os

datasets = ['squad', 'lambada', 'rocstories', 'common_sense_qa', 'imdb', 'boolq', 'copa', 'wic']
models = ['gpt3-davinci']

# for all strings below, wrap in \\hlx{ and }
hlx_strings = [# '''As of the census of 2000, there were 197,790 people, 84,549 households, and 43,627 families residing in the city. The population density was 3,292.6 people per square mile (1,271.3/kmÂ²). There were 92,282 housing units at an average density of 1,536.2 per square mile (593.1/km). The racial makeup of the city was 38.3% White, 57.2% African American, 0.2% Native American, 1.3% Asian, 0.1% Pacific Islander, 1.5% from other races, and 1.5% from two or more races. Hispanic or Latino of any race were 2.6% of the population.''',
# '''As of the census of 2000, there were 197,790 people, 84,549 households, and 43,627 families residing in the city. The population density was 3,292.6 people per square mile (1,271.3/km). There were 92,282 housing units at an average density of 1,536.2 per square mile (593.1/km). The racial makeup of the city was 38.3\\% White, 57.2\\% African American, 0.2\\% Native American, 1.3\\% Asian, 0.1\\% Pacific Islander, 1.5\\% from other races, and 1.5\\% from two or more races. Hispanic or Latino of any race were 2.6\\% of the population.'''
'''As of the census of 2000, there were 197,790 people, 84,549 households, and 43,627 families residing in the city. The population density was 3,292.6 people per square mile (1,271.3/km). There were 92,282 housing units at an average density of 1,536.2 per square mile (593.1/km). The racial makeup of the city was 38.3% White, 57.2% African American, 0.2% Native American, 1.3% Asian, 0.1% Pacific Islander, 1.5% from other races, and 1.5% from two or more races. Hispanic or Latino of any race were 2.6% of the population.''',
'''In 2000, how many families lived in Richmond?''',
'''What percentage of the Richmond population of 2000 was Pacific Islander?''',
'''43,627''',
# LAMBADA
'''"I would speak to you privately," Bowen said, casting a glance around at the others milling about.\\\\\\\\The worry in her eyes deepened, but she nodded hesitantly and awaited Bowen's directive.\\\\\\\\He led her through the great hall, annoyance biting at him when he saw no place where people weren't congregated. He stepped outside the back of the keep, where, finally, he spied an area near the bathhouses, where it was quiet and''',
# ROCSTORIES
'''Marissa loved _____ pokemon go game. It is the biggest thing right now. She had done so much more walking since she started playing it. She walked all day and evening sometimes. She walked almost 10 miles in two days.''',
'''"Marissa loved''',
'''\\\\Marissa loved''',
'''Poke GO!''',
# COQA
'''If you're still in love and end up stopping being married to your partner, what emotion are you likely to experience?''',
'''wrong, pleasure, encouragement, depression, relief''',
'''A: wrong\\\\B: pleasure\\\\C: encouragement\\\\D: depression\\\\E: relief''',
# IMDB
'''John Cassavetes is on the run from the law. He is at the bottom of the heap. He sees Negro Sidney Poitier as his equal and they quickly become friends, forming a sort of alliance against a bully of a foreman played by Jack Warden.\\\\\\\\As someone who has worked in a warehouse myself when I was younger, I can tell you that the warehouse fights, complete with tumbling packing cases and flailing grappling hooks are as realistic as it gets. I've been in fights like these myself, although no one got killed.\\\\\\\\The introduction of Sidney Poitier's widow is a variation on Shakespeare's Shylock "Do I not bleed?" This is an anti racist film, which, at the time, was much needed.\\\\\\\\All the three principle characters - Warden, Cassavetes and Poitier - are superb, with Warden the most outstanding of the three.''',
# BOOLQ
'''Pyruvic acid -- Pyruvic acid (CHCOCOOH) is the simplest of the alpha-keto acids, with a carboxylic acid and a ketone functional group. Pyruvate (/paruvet/), the conjugate base, CHCOCOO, is a key intermediate in several metabolic pathways.''',
'''Is pyruvic acid and pyruvate the same thing?''',
# COPA
'''My foot went numb.''',
'''I put my shoes on.''',
'''I shook my foot.''',
# WIC
'''The didacticism expected in books for the young.''',
'''The didacticism of the 19th century gave birth to many great museums.''',
'''didacticism''',
]

other_replacements = {
    '\\hlx{43,627} families': '43,627 families',
    '\\hlx{didacticism} of': 'didacticism of',
    '\\hlx{didacticism} expected': 'didacticism expected',
}

def get_str(dataset, model):
    '''
    Make a string with the mutual inf, accuracy, and prompt
    '''
    file_name = get_file(dataset, model)
    df = pd.read_pickle(file_name)
    # group by 'template_name', agg accuracy, mutualinf, and prompt
    templates = df.groupby('template_name').agg({'accuracy': 'mean', 'mutual_inf': 'mean', 'prompt': 'first', 'token_sets': 'first'})
    # sort descending by accuracies
    templates = templates.sort_values('accuracy', ascending=False)

    # s = '\\subsection{' + dataset_map[dataset] + '}\n'
    s = ''
    for i, (index, row) in enumerate(templates.iterrows()):
        s += ' \\textbf{'
        s += 'Prompt {} (MI: {:.3f}, Acc: {:.3f}):'.format(i+1, row['mutual_inf'], row['accuracy'])
        s += '}'
        s += '''\n\\begin{minipage}{1\linewidth} \\fbox{ \parbox{\\textwidth}{ \\fontsize{\\figurefont}{\\figurefont}\\selectfont\n'''
        prompt = row['prompt']
        # replace \n with \n\\\\
        prompt = prompt.encode('ascii', 'ignore').decode('ascii')
        prompt = prompt.replace('\n', '\\\\')
        s += prompt + '\\hly{ }\n'
        s += '}}\\end{minipage}'
        token_sets = row['token_sets']
        s += ' \\textbf{Collapsing token sets:} '
        if not token_sets:
            s += 'None, all tokens are considered'
        else:
            token_str =  str(token_sets)
            token_str = token_str.replace('{', '\\{')
            token_str = token_str.replace('}', '\\}')
            s += token_str
        s += '\n\\\\ \\\\'
    for hlx_string in hlx_strings:
        # replace hlx str with hlx{ + hlx_string + }
        s = s.replace(hlx_string, '\\hlx{' + hlx_string + '}')
    for key, value in other_replacements.items():
        # replace other str with other_replacement
        s = s.replace(key, value)
    # convert to ascii
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s

def save_prompts(dataset, model, output_file='prompts/latex_prompts.txt'):
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
            s += '\subsection{' + dataset_map[dataset] + '}\n'
            s += get_str(dataset, model)
    # convert to ascii
    s = s.encode('ascii', 'ignore').decode('ascii')
    # replace all % with \%
    s = s.replace('%', '\\%')
    # replace all _ with \_
    s = s.replace('_', '\\_')
    # write s to output file
    with open('prompts/latex_prompts.txt', 'w') as f:
        f.write(s)



if __name__ == '__main__':
    # save_all_prompts()
    combine_all_prompts()