import numpy as np
import pandas as pd
from nyt_categories import categories as nyt_categories
from congress_categories import categories as congress_categories

arguments = {
    'nytimes': {
        'data_path': 'data/nyt/nytimes.csv', # path to data
        'int_to_cat': nyt_categories, # dictionary mapping integer to category
        'cat_to_int': {cat: i for i, cat in nyt_categories.items()}, # dictionary mapping category to integer
        'categories': list(nyt_categories.values()), # list of categories
        'input_column': 'title', # name of column containing text to be input
        'category_column': 'topic_2digit', # name of column containing category
        'prefix': 'Using only the following categories\n"""\n', # prefix to add to front of instructions
        'per_cat_lambda': lambda x: x, # function to modify category in instructions
        'join_cats': '\n', # string to join categories in instructions
        'suffix': '\n"""\nAssign the following headlines to one of the categories:', # suffix to add to end of instructions
        'input_lambda': lambda x: x, # function to modify input text in prompt
        'category_lambda': lambda x: x, # function to modify category in prompt
        'join_input_category': '->', # string to join input and category in prompt
        'join_inputs': '\n', # string to join exemplars in prompt
    },
    'nytimes-body': {
        'data_path': 'data/nyt/nytimes.csv',
        'body_path': 'data/nyt/bodies-small.csv', # path to body data
        'body_column': 'body', # name of column containing body text
        'join_input_body': '\n\n', # string to join input and body in prompt
        'int_to_cat': nyt_categories,
        'cat_to_int': {cat: i for i, cat in nyt_categories.items()},
        'categories': list(nyt_categories.values()),
        'input_column': 'title',
        'category_column': 'topic_2digit',
        'prefix': 'Using only the following categories\n"""\n',
        'per_cat_lambda': lambda x: x,
        'join_cats': '\n',
        'suffix': '\n"""\nAssign the following headlines to one of the categories:',
        'input_lambda': lambda x: f'"""{x} """',
        'category_lambda': lambda x: x,
        'join_input_category': '->',
        'join_inputs': '\n\n',
    },
    'congress': {
        'data_path': 'data/congressional_hearings/hearings.csv',
        'int_to_cat': congress_categories,
        'cat_to_int': {cat: i for i, cat in congress_categories.items()},
        'categories': list(congress_categories.values()),
        'input_column': 'description',
        'category_column': 'majortopic',
        'prefix': 'Using only the following categories\n"""\n',
        'per_cat_lambda': lambda x: x,
        'join_cats': '\n',
        'suffix': '\n"""\nAssign the following congressional hearing summaries to one of the categories:',
        'input_lambda': lambda x: x,
        'category_lambda': lambda x: x,
        'join_input_category': '->',
        'join_inputs': '\n',
    },
}


class Templatizer:
    def __init__(self, dataset_name, **kwargs):
        '''
        Arguments:
            dataset_name (string): name of dataset. supported: 'nytimes', 'nytimes-body', congress'
            **kwargs (dictionary): arguments to override default arguments, as shown in arguments dictionary
        '''
        # check if dataset_name is supported
        if dataset_name not in arguments:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

        self.dataset_name = dataset_name
        self.args = arguments[self.dataset_name]
        # update args with kwargs
        self.args.update(kwargs)

        self.load_dataset(self.dataset_name)

        if self.dataset_name == 'nytimes-body':
            self.add_body_to_input()

    
    def load_pandas(self, path):
        '''
        Load in a pandas array from a filepath.
        Supported file types: csv, pkl
        '''
        if path.endswith('.csv'):
            return pd.read_csv(path, encoding='unicode_escape')
        if path.endswith('.pkl'):
            return pd.read_pickle(path)
        raise ValueError(f'Unsupported file type for load_pandas: {path}')

    
    def add_body_to_input(self):
        '''
        Adds body to input. supported dataset_name: 'nytimes-body'
        '''
        # make sure supported
        if self.dataset_name != 'nytimes-body':
            raise ValueError(f'Unsupported dataset for add_body_to_input: {self.dataset_name}')
        body = self.load_pandas(self.args['body_path'])
        # merge body on input_column with dataset
        self.dataset = pd.merge(self.dataset, body, on=self.args['input_column'])
        # dataset drop where body_column is na
        self.dataset = self.dataset.dropna(subset=[self.args['body_column']])
        # add body_column to input
        self.dataset.input = self.dataset.input + self.args['join_input_body'] + self.dataset[self.args['body_column']]

    
    def load_dataset(self, dataset_name):
        '''
        Loads dataset.
        '''
        self.dataset = self.load_pandas(self.args['data_path'])

        # create input column
        self.dataset['input'] = self.dataset[self.args['input_column']]
        # create category column
        self.dataset['category'] = self.dataset[self.args['category_column']]
        # map category from int to string
        self.dataset.category = self.dataset.category.map(self.args['int_to_cat'])

        # dropna for dataset for columns input and category
        self.dataset = self.dataset.dropna(subset=['input', 'category'])
    
    
    # def generate_instructions(self, prefix = None, suffix = None, per_cat_lambda = None, join_cats = None):
    def generate_instructions(self):
        '''
        Arguments:
            prefix (string): prefix before categories
            suffix (string): suffix after categories
            per_cat_lambda (lambda function): lambda function to apply to each category
            join_cats (string): string to join categories
        '''
        # # if argument is missing, fill with default
        # if prefix is None:
        #     prefix = self.args['prefix_default']
        # if suffix is None:
        #     suffix = self.args['suffix_default']
        # if per_cat_lambda is None:
        #     per_cat_lambda = self.args['per_cat_lambda_default']
        # if join_cats is None:
        #     join_cats = self.args['join_cats_default']

        # run category lambda function
        cats = [self.args['per_cat_lambda'](cat) for cat in self.args['categories']]
        # join categories
        cat_string = self.args['join_cats'].join(cats)
        # add prefix and suffix
        instructions = self.args['prefix'] + cat_string + self.args['suffix']
        return instructions
    
    def templatize_instance(self, input, category):
        '''
        Templatize a single instance of input and category.
        If category is none, keep empty.
        '''
        input = self.args['input_lambda'](input)
        # if not category, make sure to add any prefix if neccessary
        if not category:
            test_str = 'TEST TEST'
            category = self.args['category_lambda'](test_str)
            category = category.split(test_str)[0]
        else:
            category = self.args['category_lambda'](category)

        # if not category, don't include trailing space
        if not category:
            return f'{input} {self.args["join_input_category"]}'
        else:
            return f'{input} {self.args["join_input_category"]} {category}'
    
    def generate_exemplars(self, n_exemplars, seed_exemplars=-1):
        '''
        Select n randomly generated exemplars.
        '''
        # TODO - add functionality for sampling "ambiguous" or "prototypical"
        if seed_exemplars != -1:
            seed_offset = 42
            exemplars = self.dataset.sample(n=n_exemplars, random_state=seed_exemplars+seed_offset)
        else:
            exemplars = self.dataset.sample(n=n_exemplars)
        return exemplars
    
    def get_subset(self, n_per_category=30, seed_instances=-1):
        '''
        Select a subset of the dataset. Draw n from each category.
        '''
        category_counts = self.dataset.category.value_counts()
        if seed_instances != -1:
            subset = pd.concat([self.dataset[self.dataset.category == cat].sample(n=n_per_category, random_state=seed_instances, replace=False if category_counts[cat] > n_per_category else True) for cat in self.args['categories']], axis=0)
        else:
            subset = pd.concat([self.dataset[self.dataset.category == cat].sample(n=n_per_category, replace=False if category_counts[cat] > n_per_category else True) for cat in self.args['categories']], axis=0)
            # subset = pd.concat([self.dataset[self.dataset.category == cat].sample(n=n_per_category) for cat in self.args['categories']], axis=0)
        return subset
    
    def templatize_row(self, row, n_exemplars=3, seed_exemplars=-1):
        '''
        Templatize a single row of input and category.
        '''
        # TODO - I could probably just generate this somewhere else
        instructions = self.generate_instructions()
        
        exemplars = self.generate_exemplars(n_exemplars=n_exemplars, seed_exemplars=seed_exemplars)
        # generate per instance
        example_texts = [self.templatize_instance(input=example.input, category=example.category) for example in exemplars.itertuples()]
        # join
        example_text = self.args['join_inputs'].join(example_texts)

        prompt = self.templatize_instance(input=row.input, category='')

        return instructions + self.args['join_inputs'] + example_text + self.args['join_inputs'] + prompt
    
    def templatize(self, n_per_category=30, n_exemplars=3, seed_instances=-1, seed_exemplars=-1, **kwargs):
        '''
        Templatize the dataset.
        Arguments:
            n_per_category (int): number of instances to draw from each category
            n_exemplars (int): number of exemplars to draw
            seed_instances (int): random see for drawing instances
            seed_exemplars (int or lambda function): random see for drawing exemplars per instance.
                If lambda function, applies function to each instance row (1, 2, 3, ..., n_per_category*len(categories))
            **kwargs: any necessary updates to args
        '''
        # update args with kwargs
        self.args.update(kwargs)

        # if seed_exemplars is an int, turn into lambda function
        if isinstance(seed_exemplars, int):
            f = lambda x: seed_exemplars
        else:
            f = seed_exemplars

        subset = self.get_subset(n_per_category=n_per_category, seed_instances=seed_instances)
        prompts = [self.templatize_row(row, n_exemplars=n_exemplars, seed_exemplars=f(i)) for i, row in subset.iterrows()]
        categories = subset.category.tolist()
        # make a list of dictionaries with keys- 'text': prompt, 'target': category
        return [{'text': prompt.strip(), 'target': category} for prompt, category in zip(prompts, categories)]

if __name__ == '__main__':
    # instantiate templatizer
    templatizer = Templatizer(dataset_name='nytimes')
    # do for 10 runs
    n_runs = 10
    # hold exemplars constant across instances
    for k in range(n_runs):
        output = templatizer.templatize(n_per_category=10, n_exemplars=3, seed_instances=k, seed_exemplars=k)
        print(output[0]['text'])
        print(output[1]['text'])
    # sample different exemplars across instances
    for k in range(n_runs):
        output = templatizer.templatize(n_per_category=10, n_exemplars=3, seed_instances=k, seed_exemplars=lambda i: n_runs*i + k)
        print(output[0]['text'])
        print(output[1]['text'])

    # # nytimes example
    # print('nytimes')
    # templatizer = Templatizer(dataset_name='nytimes')
    # output = templatizer.templatize(n_per_category=10, seed=0, n_exemplars=6, category_lambda=lambda x: f'"{x}"')
    # print(output[0]['text'])
    # print()

    # # nytimes-body example
    # print('nytimes-body')
    # templatizer = Templatizer(dataset_name='nytimes-body')
    # output = templatizer.templatize(n_per_category=10, seed=0, n_exemplars=3)
    # print(output[0]['text'])
    # print()

    # # congress example
    # print('congress')
    # templatizer = Templatizer(dataset_name='congress')
    # output = templatizer.templatize(n_per_category=10, seed=0, n_exemplars=3)
    # print(output[0]['text'])
    # print()
