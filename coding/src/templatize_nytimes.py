import pandas as pd
from nyt_categories import categories

catintro = "Using only the following categories"

all_categories = list(categories.values())
category_string = '\n'.join(all_categories)

instructions = 'Assign the following headlines to one of the categories:'

exheadlines = '''U.N. Says North Korea Will Face Famine as Early as This Summer
Apple Computer Ousts Chief In Response to Poor Results
THE SARS EPIDEMIC: THE AMERICAN RESPONSE: Aggressive Steps, and Luck, Help U.S. Avoid SARS Brunt
Talks Are Pressed to End Strike By Phone Workers on East Coast'''.split('\n')

excats = '''International Affairs and Foreign Aid
Banking, Finance, and Domestic Commerce
Health
Labor'''.split('\n')


nl = '\n'

func = lambda x: lambda example_headlines, example_categories: f'''{catintro}
"""
{category_string}
"""
{instructions}

{nl.join([f" : ".join((ex[0], ex[1])) for ex in zip(example_headlines, example_categories)])}
{x} :'''

# REVERSE
category_reverse = '\n'.join(all_categories[::-1])
func_reverse= lambda x: lambda example_headlines, example_categories: f'''{catintro}
"""
{category_reverse}
"""
{instructions}

{nl.join([f" : ".join((ex[0], ex[1])) for ex in zip(example_headlines, example_categories)])}
{x} :'''

# quotes
quote_categories = [f'"{cat}"' for cat in all_categories]
category_quotes = '\n'.join(quote_categories)
func_quotes= lambda x: lambda example_headlines, example_categories: f'''{catintro}
"""
{category_quotes}
"""
{instructions}

{nl.join([f" : ".join((ex[0], ex[1])) for ex in zip(example_headlines, [f'"{excat}"' for excat in example_categories])])}
{x} : "'''

# forward slash
category_string = '\n'.join(all_categories)
func_slash = lambda x: lambda example_headlines, example_categories: f'''{catintro}
"""
{category_string}
"""
{instructions}

{nl.join([f" / ".join((ex[0], ex[1])) for ex in zip(example_headlines, example_categories)])}
{x} /'''

# arrow
category_string = '\n'.join(all_categories)
func_arrow = lambda x: lambda example_headlines, example_categories: f'''{catintro}
"""
{category_string}
"""
{instructions}

{(nl+nl).join([f" -> ".join((ex[0], ex[1])) for ex in zip(example_headlines, example_categories)])}

{x} ->'''

example_data = pd.read_csv('data/nyt/nytimes.csv', encoding='unicode_escape')
example_data['category'] = example_data.topic_2digit.map(categories)
example_data = example_data[['title', 'category']]
example_data = example_data.dropna()

def generate_examples(n, seed=0):
    global example_data
    # TODO - add functionality for sampling "ambiguous" or "prototypical" examples
    examples = example_data.sample(n=n, random_state=seed)
    return examples.title.tolist(), examples.category.tolist()


def templatize(
    n=30,
    n_examples=4,
    variation = 'slash',
    surround_triple_quotes=False,
    bodies_file=''):
    '''
    Templatize the nytimes dataset.
    Arguments:
        n: number of headlines to be templatized per category
        n_examples: number of examples per sample
        variation: 'slash', 'quotes', 'reverse', or 'arrow'
        surround_triple_quotes: surround the body with triple quotes
        bodies_file: path to file containing article bodies. Adds bodies to headline if not empty.
    '''
    df = pd.read_csv('data/nyt/nytimes.csv', encoding='unicode_escape')
    df['category'] = df.topic_2digit.map(categories)

    global example_data

    if bodies_file:
        bodies = pd.read_csv(bodies_file, encoding='unicode_escape')
        # merge df and bodies by title
        df = df.merge(bodies, on='title')
        # concat bodies to titles
        df['title'] = df.title + '\n\n' + df.body

        # do same for example data
        example_data = example_data.merge(bodies, on='title')
        example_data['title'] = example_data.title + '\n\n' + example_data.body

    
    if surround_triple_quotes:
        func_quotes = lambda x: f'"""{x} """'
        df['title'] = df.title.map(func_quotes)
        example_data['title'] = example_data.title.map(func_quotes)



    # shuffle
    df = df.sample(frac=1, random_state=1)
    if variation == 'novar':
        df['prompt'] = df.title.map(func)
    if variation == 'reverse':
        df['prompt'] = df.title.map(func_reverse)
    if variation == 'quotes':
        df['prompt'] = df.title.map(func_quotes)
    if variation == 'slash':
        df['prompt'] = df.title.map(func_slash)
    if variation == 'arrow':
        df['prompt'] = df.title.map(func_arrow)


    # filter to just prompt and category columns
    df = df[['prompt', 'category']]

    prompts = []
    for category in categories.values():
        df_cat = df[df.category == category]
        # take n from each category
        df_cat = df_cat.iloc[:n]
        for i, instance in df_cat.iterrows():
            example_headlines, example_categories = generate_examples(n_examples, seed=i)
            prompts.append({'text': instance.prompt(example_headlines, example_categories), 'target': instance.category})

    return prompts 

 

if __name__ == '__main__':
    # variations = ['novar', 'reverse', 'quotes', 'slash']
    # for variation in variations:
    variation = 'arrow'
    print(variation)

    # excluding body text
    # prompts = templatize(variation=variation, n_examples=3)

    # including body text
    prompts = templatize(variation=variation, n_examples=3, surround_triple_quotes=True, bodies_file='data/nyt/bodies-small.csv')
    print(prompts[20]['text'])
    # print(prompts[101]['text'])
    print(len(prompts))
    pass
