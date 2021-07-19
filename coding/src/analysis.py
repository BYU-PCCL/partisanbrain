import numpy as np
import pandas as pd
import os

class ExperimentResults():
    def __init__(self, results_path):
        '''
        Initialize results object. Read in all results from results_path.
        Attributes:
            results_path: path to directory containing results
            results: pandas dataframe containing all results
                Columns:
                    'text': text of the prompt
                    'response': response to the prompt (type: OpenAI response)
                    'I': instances index
                    'E': exemplars index
                    'N_exemplars': number of exemplars
                    'N_per_cat': number of instances per category
                    'category': true category of the instance
        '''
        self.results_path = results_path
        # supported matching strategies
        # self.matching_strategies = ['exact', 'starts_with', 'substr']
        self.matching_strategies = ['starts_with', 'substr']
        self.results = None
        self.load_results()
        # TODO - put somewhere else
        self.results['category'] = self.results.target
        self.populate_category_probs()
        self.populate_scores()

    def read_df(self, file_name):
        '''
        Reads a file into a pandas dataframe. Supported filetypes: csv, pkl
        '''
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        elif file_name.endswith('.pkl'):
            df = pd.read_pickle(file_name)
        else:
            raise ValueError('File type not supported. Supported filetypes: csv, pkl')
        return df

    def read_and_combine_dfs(self, file_names):
        '''
        Reads a list of files into a pandas dataframe. Supported filetypes: csv, pkl
        '''
        dfs = []
        for file_name in file_names:
            dfs.append(self.read_df(file_name))
        df = pd.concat(dfs, ignore_index=True)
        return df

    def get_all_files(self, path, ends_with='.pkl'):
        '''
        Searches subdirectories in path and returns a list of all files ending with
        ends_with.
        '''
        matched_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(ends_with):
                    matched_files.append(os.path.join(root, file))
        return matched_files
    
    def load_results(self):
        '''
        Loads all files in results_path into self.results
        '''
        files = self.get_all_files(self.results_path)
        self.results = self.read_and_combine_dfs(files)
        # TODO - remove
        self.results = self.results.dropna()
        # self.results = self.results.sample(n=200)
    
    def get_logprobs(self, response):
        '''
        Given an instance of gpt3 response, return the logprobs of the first sampled token.
        Returns a sorted list of tuples of (token, logprob)
        '''
        logprobs = response['choices'][0].logprobs.top_logprobs[1]
        # sort by logprob
        logprobs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
        return logprobs
    
    def matches(self, token, category, matching_strategy='substr'):
        '''
        Returns True if token matches category according to the matching strategy.
        Arguments:
            matching_strategy (str): matching strategy for logprobs to categories. Supported: 'exact', 'starts_with', 'substr'. Defaults to 'substr'.
                'exact': exact match of token and category. Doesn't work for all categories since some are longer than 1 token.
                'starts_with': category starts with token
                'substr': token is a substring of category
        '''
        # strip token and category and change to lowercase
        token = token.lower().strip()
        category = category.lower().strip()

        # check matching strategy
        if matching_strategy == 'exact':
            return token == category
        elif matching_strategy == 'starts_with':
            return category.startswith(token)
        elif matching_strategy == 'substr':
            return token in category
    
    def get_category_probs(self, response, categories, matching_strategy='substr'):
        '''
        Iterate through responses and return the relative probability for each category.
        Arguments:
            response (OpenAI GPT-3 response): gpt3 output results
            categories (list): list of categories
            matching_strategy (str): matching strategy for logprobs to categories. Supported: 'exact', 'starts_with', 'substr'. Defaults to 'substr'.
                'exact': exact match of token and category. Doesn't work for all categories since some are longer than 1 token.
                'starts_with': category starts with token
                'substr': token is a substring of category
        Returns:
            category_probs (dict): dictionary of category probabilities
        '''
        # get logprobs
        logprobs = self.get_logprobs(response)
        # get category probabilities
        category_probs = {cat: 0 for cat in categories}

        for token, logprob in logprobs:
            for category in categories:
                if self.matches(token, category, matching_strategy):
                    category_probs[category] += np.exp(logprob)
        
        # normalize so sums to 1
        category_probs = {cat: prob/sum(category_probs.values()) for cat, prob in category_probs.items()}

        return category_probs
    
    def populate_category_probs(self):
        '''
        Populate category probs in self.results for each matching strategy. 
        After running, self.results should contain columns 'probs_exact', 'probs_starts_with', and 'probs_substr'.
        '''
        # get all categories
        categories = self.results['category'].unique()
        # populate category probs
        for matching_strategy in self.matching_strategies:
            column_name = 'probs_' + matching_strategy
            self.results[column_name] = self.results.apply(lambda x: self.get_category_probs(x['response'], categories, matching_strategy), axis=1)
    
    # TODO - add a function to normalize category_probs by marginal probability of each category
    
    def argmax(self, category_probs):
        '''
        Given a dictionary of category_probs, return the category with the highest probability.
        '''
        return max(category_probs, key=category_probs.get)
        
    def populate_scores(self):
        '''
        Populate score (0 if match, 1 otherwise) in self.results for each matching strategy.
        After running, self.results should contain columns 'score_exact', 'score_starts_with', 'score_substr'.
        '''
        # populate score
        for matching_strategy in self.matching_strategies:
            column_name = 'score_' + matching_strategy
            self.results[column_name] = self.results.apply(lambda x: 1 if self.argmax(x['probs_' + matching_strategy]) == x['category'] else 0, axis=1)
    
    def group_by_columns(self, columns, df=None):
        '''
        Groups df by columns and returns a dataframe of the aggregated scores.
        Arguments:
            columns (list): list of columns to group by
            df (pandas dataframe): dataframe to group. Default is self.results
        '''
        if df is None:
            df = self.results
        # df = df.groupby(columns).agg({'score_exact': 'mean', 'score_starts_with': 'mean', 'score_substr': 'mean'})
        df = df.groupby(columns).agg({'score_starts_with': 'mean', 'score_substr': 'mean'})
        # drop all columns except [*columns, 'score_exact', 'score_starts_with', 'score_substr']
        # keep_columns = columns + ['score_exact', 'score_starts_with', 'score_substr']
        # keep_columns = columns + ['score_starts_with', 'score_substr']
        # df = df[keep_columns]
        # # set index to columns
        # df = df.set_index(columns)
        return df

if __name__ == '__main__':
    er = ExperimentResults('experiments/nyt/07-16-2021/examples_instances_runs/combined/')
    df = er.group_by_columns(['n'])
    breakpoint()
    pass
