import pandas as pd
from nyt_categories import categories

catintro = "Using only the following categories"

all_categories = list(categories.values())
category_string = '\n'.join(all_categories)

instructions = 'assign the following headlines to one of the categories:'

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
func_slash= lambda x: lambda example_headlines, example_categories: f'''{catintro}
"""
{category_string}
"""
{instructions}

{nl.join([f" / ".join((ex[0], ex[1])) for ex in zip(example_headlines, example_categories)])}
{x} /'''

data = pd.read_csv('data/nyt/nytimes.csv', encoding='unicode_escape')
data['category'] = data.topic_2digit.map(categories)
breakpoint()
data = data[['title', 'category']]
data = data.dropna()

def generate_examples(n, seed=0):
    # TODO - add functionality for sampling "ambiguous" or "prototypical" examples
    examples = data.sample(n=n, random_state=seed)
    return examples.title.tolist(), examples.category.tolist()


def templatize(n=30, n_examples=4, variation = 'slash'):
    df = pd.read_csv('data/nyt/nytimes.csv', encoding='unicode_escape')
    df['category'] = df.topic_2digit.map(categories)
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
    variation = 'slash'
    print(variation)
    prompts = templatize(variation=variation, n_examples=3)
    print(prompts[100]['text'])
    print(prompts[101]['text'])
    print(len(prompts))
    pass
