import numpy as np
from pdb import set_trace as breakpoint
from numpy.lib.function_base import extract
import pandas as pd
from pandas.core.frame import DataFrame
from nyt_categories import categories as nyt_categories
from nyt_categories import category_descriptions as nyt_descriptions
from congress_categories import categories as congress_categories
from tqdm import tqdm

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
        'join_category_description': lambda cat, desc: f'{cat} ({desc})', # function to join category and description in prompt
        'use_description': False, # whether to use description in prompt
        'category_to_description': nyt_descriptions, # dictionary mapping category to descriptio
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
        'join_category_description': lambda cat, desc: f'{cat} ({desc})',
        'use_description': False,
        'category_to_description': nyt_descriptions,
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
        self.args = arguments[self.dataset_name].copy()
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
        cats = self.args['categories']

        # if we want to include the description in the instructions
        if 'use_description' in self.args:
            if self.args['use_description']:
                # apply join_category_description
                d = self.args['category_to_description']
                cats = [self.args['join_category_description'](cat, d[cat]) for cat in cats]

        # run category lambda function
        cats = [self.args['per_cat_lambda'](cat) for cat in cats]

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
    
    def generate_exemplars(self, n_exemplars, seed_exemplars=0):
        '''
        Select n randomly generated exemplars.
        '''
        # TODO - add functionality for sampling "ambiguous" or "prototypical"
        # offset seed so that exemplars are different than instances
        seed_offset = 42
        exemplars = self.dataset.sample(n=n_exemplars, random_state=seed_exemplars+seed_offset)
        return exemplars
    
    def get_subset(self, n_per_category=30, seed_instances=0):
        '''
        Select a subset of the dataset. Draw n from each category.
        '''
        category_counts = self.dataset.category.value_counts()
        subset = pd.concat([self.dataset[self.dataset.category == cat].sample(n=n_per_category, random_state=seed_instances, replace=False if category_counts[cat] > n_per_category else True) for cat in self.args['categories']], axis=0)
        return subset
    
    def ambiguity_candidates(self):
        """Create a df with 90 instances in each category to pass a constant set
        of exemplars and score for ambiguity/prototypicality"""
        instance_set = self.get_subset(n_per_category=90)
        for i, row in instance_set.iterrows():
            prompt = self.templatize_row(
                row, 
                n_exemplars = 3, 
                seed_exemplars = 0)
            instance_set.at[i, "prompt"] = prompt
        return instance_set

    def extract_exemplars(self, n_exemplars, seed_exemplars=0):
        """Extract exemplars in a dataset into a list of exemplars

        Args:
            n_exemplars (int): Number of exemplars to extract
            seed_exemplars (int): Random seed to extract exemplars

        Returns:
            example_texts: List of exemplars joined by self.args.join_input_category
        """
        exemplars = self.generate_exemplars(n_exemplars=n_exemplars, seed_exemplars=seed_exemplars)
        example_texts = [self.templatize_instance(input=example.input, category=example.category) for example in exemplars.itertuples()]
        return example_texts

    def ambiguity_candidates(self):
        """Create a df with 90 instances in each category to pass a constant set of exemplars and score for ambiguity/prototypicality"""
        instance_set = self.get_subset(n_per_category=90)
        for i, row in instance_set.iterrows():
            prompt = self.templatize_row(row, n_exemplars = 50, seed_exemplars = 0)
            instance_set.at[i, "prompt"] = prompt
        return instance_set

    def extract_exemplars(self, n_exemplars, seed_exemplars=0):
        """Extract exemplars in a dataset into a list of exemplars
        Args:
            n_exemplars (int): Number of exemplars to extract
            seed_exemplars (int): Random seed to extract exemplars
        Returns:
            example_texts: List of exemplars joined by self.args.join_input_category
        """
        exemplars = self.generate_exemplars(n_exemplars=n_exemplars, seed_exemplars=seed_exemplars)
        example_texts = [self.templatize_instance(input=example.input, category=example.category) for example in exemplars.itertuples()]
        return example_texts

    
    def templatize_row(self, row, n_exemplars=3, seed_exemplars=0):
        '''
        Templatize a single row of input and category.

        Arguments:
            row (row of a pandas dataframe)
        '''
        # TODO - I could probably just generate this somewhere else
        instructions = self.generate_instructions()
        
        example_texts = self.extract_exemplars(n_exemplars, seed_exemplars)
        example_text = self.args['join_inputs'].join(example_texts)

        prompt = self.templatize_instance(input=row.input, category='')

        return instructions + self.args['join_inputs'] + example_text + self.args['join_inputs'] + prompt
    
    def templatize(self,
        n_per_category=1,
        n_exemplars=1,
        seed_instances=0,
        seed_exemplars=0,
        **kwargs):
        '''
        Templatize the dataset.

        Arguments:
            ns_per_category (int): number of instances to draw from each category
            ns_exemplars (int): number of exemplars to draw
            seed_instances (int): random see for drawing instances
            seed_exemplars (int): random see for drawing exemplars per instance.
            **kwargs: any necessary updates to args
        '''
        # throw an exception if any kwargs are not in self.args
        for key in kwargs:
            if key not in self.args:
                raise Exception(f'Keyword argument {key} not in args and is not supported.')

        # update args with kwargs
        self.args.update(kwargs)

        instance_set = self.get_subset(n_per_category=n_per_category, seed_instances=seed_instances)
        for i, row in instance_set.iterrows():
            exemplars = self.extract_exemplars(n_exemplars, seed_exemplars)
            instance_set.at[i, "exemplars"] = "|||".join(exemplars)
            prompt = self.templatize_row(row, n_exemplars=n_exemplars, seed_exemplars=seed_exemplars)
            instance_set.at[i, "prompt"] = prompt
            instance_set.at[i, "n_per_category"] = int(n_per_category)
            instance_set.at[i, "instance_set_ix"] = int(seed_instances)
            instance_set.at[i, "exemplar_set_ix"] = int(seed_exemplars)
            instance_set.at[i, "n_exemplars"] = int(n_exemplars)
            instance_set.at[i, "prompt_length"] = len(prompt.split())
        return instance_set

    def templatize_many(self,
        ns_per_category=[1],
        ns_exemplars=[1],
        n_exemplar_runs=1,
        n_instance_runs=1,
        **kwargs):
        '''
        Templatizes over a cartesian product of all possible combinations of the input parameters.
        Arguments:
            ns_per_category (list(int)): range of number of instances to draw from each category
            ns_exemplars (list(int)): range of number of exemplars to draw
            ns_exemplar_runs (int): number of distinct exemplar set trials
            ns_instance_runs (int): number of distinct instance set trials
            **kwargs: any necessary updates to args
        '''
        # if ns_per_category is not iterable, make a list
        if not hasattr(ns_per_category, '__iter__'):
            ns_per_category = [ns_per_category]
        # if ns_exemplars is not iterable, make a list
        if not hasattr(ns_exemplars, '__iter__'):
            ns_exemplars = [ns_exemplars]
        
        df = pd.DataFrame()
        
        print('Templatizing')
        # For every n_per_category
        for n_per_category in tqdm(ns_per_category):
            # Grab a distinct instance set n_instance runs times
            for instance_set_ix in range(n_instance_runs):
                # Then for every instance set, seed for sampling exemplars
                for exemplar_set_ix in range(n_exemplar_runs):
                    # Grow the set of exemplars according to ns_exemplars
                    for n_exemplars in ns_exemplars:
                        instance_set = self.templatize(
                            n_per_category=n_per_category,
                            n_exemplars=n_exemplars,
                            seed_exemplars=exemplar_set_ix,
                            seed_instances=instance_set_ix,
                            **kwargs,
                        )
                        df = df.append(instance_set)
        return df

if __name__ == '__main__':
    templatizer = Templatizer(dataset_name='nytimes')
    output = templatizer.templatize(use_description=True)
    breakpoint()
    pass
    # templatizer = Templatizer(dataset_name='nytimes')
    # output = templatizer.templatize_many(
    #     ns_per_category=[1, 2, 3, 4],
    #     ns_exemplars=[1, 2, 3, 4, 5],
    #     n_exemplar_runs=5,
    #     n_instance_runs=5,
    # )

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