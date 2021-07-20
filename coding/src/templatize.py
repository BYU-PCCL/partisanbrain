import numpy as np
from pdb import set_trace as breakpoint
from numpy.lib.function_base import extract
import pandas as pd
from pandas.core.frame import DataFrame
from nyt_categories import categories as nyt_categories
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

    def extract_exemplars(self, n_exemplars, seed_exemplars):
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

    
    def templatize_row(self, row, n_exemplars=3, seed_exemplars=-1):
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
        ns_per_category=[1],
        ns_exemplars=[1],
        n_exemplar_runs=1,
        n_instance_runs=1,
        seed_instances=-1,
        seed_exemplars=-1,
        **kwargs):
        '''
        Templatize the dataset and put it into a df.
        Arguments:
            ns_per_category (list): range of number of instances to draw from each category
            ns_exemplars (list): range of number of exemplars to draw
            n_exemplar_runs (int): number of distinct exemplar set trials
            n_instance_runs (int): number of distinct instance set trials
            seed_instances (int): random see for drawing instances
            seed_exemplars (int or lambda function): random see for drawing exemplars per instance.
                If lambda function, applies function to each instance row (1, 2, 3, ..., ns_per_category*len(categories))
            **kwargs: any necessary updates to args
        '''
        # update args with kwargs
        self.args.update(kwargs)

        df = DataFrame()

        # For every n_per_category
        for n_per_category in tqdm(ns_per_category):
            # Grab a distinct instance set n_instance runs times
            for instance_set_ix in range(n_instance_runs):
                instance_set_orig = self.get_subset(n_per_category=n_per_category, seed_instances=instance_set_ix)

                #Then for every instance set, seed for sampling exemplars
                for exemplar_set_ix in range(n_exemplar_runs):
                    #Grow the set of exemplars according to ns_exemplars
                    for n_exemplars in ns_exemplars:
                        instance_set = instance_set_orig.copy()
                        for i, row in instance_set.iterrows():
                            exemplars = self.extract_exemplars(n_exemplars, exemplar_set_ix)
                            instance_set.at[i, 'exemplars'] = "|||".join(exemplars)
                            prompt = self.templatize_row(row, n_exemplars=n_exemplars, seed_exemplars=exemplar_set_ix)
                            instance_set.at[i, 'prompt'] = prompt
                            instance_set.at[i, "n_per_category"] = int(n_per_category)
                            instance_set.at[i, "instance_set_ix"] = int(instance_set_ix)
                            instance_set.at[i, "exemplar_set_ix"] = int(exemplar_set_ix)
                            instance_set.at[i, "n_exemplars"] = int(n_exemplars)
                            instance_set.at[i, 'prompt_length'] = len(prompt.split())
                        df = df.append(instance_set)
        return df


def test_ns_exemplars():
    """Tests whether each n_exemplar in n_exemplars shows up 28 times, and that there are exactly 2"""
    templatizer = Templatizer(dataset_name='nytimes')
    ns_exemplars = [1, 3]
    output = templatizer.templatize(
        ns_exemplars=ns_exemplars,
        )
    counts = output.n_exemplars.value_counts()
    assert len(counts) == 2
    assert output.n_exemplars.value_counts().loc[3.0] == 28
    assert output.n_exemplars.value_counts().loc[1.0] == 28

                
def test_ns_per_category():
    """Tests whether each code appears np.sum(ns_per_category) times"""
    templatizer = Templatizer(dataset_name='nytimes')
    ns_per_category = [1, 2]
    output = templatizer.templatize(
        ns_per_category=ns_per_category,
        )
    counts = output.topic_2digit.value_counts().unique()
    assert len(counts) == 1
    assert counts[0] == np.sum(ns_per_category)


    ns_per_category = [1, 2, 4]
    output = templatizer.templatize(
        ns_per_category=ns_per_category,
        )
    counts = output.topic_2digit.value_counts().unique()
    assert len(counts) == 1
    assert counts[0] == np.sum(ns_per_category)

def test_exemplar_constancy():
    """Test whether marginal exemplar was the only one that changed"""
    templatizer = Templatizer(dataset_name='nytimes')
    ns_exemplars = [1, 2, 3]
    output = templatizer.templatize(
        ns_exemplars=ns_exemplars,
        )
    articledf = output[output.article_id==4262] 
    article1shotexemplars = articledf.iloc[0].exemplars.split('|||')
    article2shotexemplars  = articledf.iloc[1].exemplars.split('|||')
    article3shotexemplars  = articledf.iloc[2].exemplars.split('|||')
    assert article3shotexemplars[0] == article2shotexemplars[0] == article1shotexemplars[0]
    assert article3shotexemplars[1] == article2shotexemplars[1]

def tests():
    test_exemplar_constancy()
    test_ns_exemplars()
    test_ns_per_category()

if __name__ == '__main__':
    # instantiate templatizer
    templatizer = Templatizer(dataset_name='nytimes')
    output = templatizer.templatize(
        ns_per_category=[1, 2, 3, 4],
        ns_exemplars=[1, 2, 3, 4, 5],
        n_exemplar_runs=5,
        n_instance_runs=5,
        )
    breakpoint()

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
