import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import product

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
        print(file_names)
        dfs = []
        for file_name in file_names:
            dfs.append(self.read_df(file_name))
        if len(dfs) == 1:
            df = dfs[0]
        else:
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
        # dropna
        self.results = self.results.dropna()
        # TODO - remove
        # self.results = self.results.sample(n=1000)
        # reset index
        self.results = self.results.reset_index()
    
    def get_logprobs(self, response):
        '''
        Given an instance of gpt3 response, return the logprobs of the first sampled token.
        Returns a sorted list of tuples of (token, logprob)
        '''
        logprobs = response['choices'][0].logprobs.top_logprobs[0]
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
    
    def get_category_probs(self, response, matching_strategy='substr'):
        '''
        Iterate through responses and return the relative probability for each category.
        Arguments:
            response (OpenAI GPT-3 response): gpt3 output results
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
        category_probs = {cat: 0 for cat in self.categories}

        for token, logprob in logprobs:
            for category in self.categories:
                if self.matches(token, category, matching_strategy):
                    category_probs[category] += np.exp(logprob)
        
        # normalize so sums to 1
        category_probs = {cat: prob/sum(category_probs.values()) for cat, prob in category_probs.items()}

        return category_probs
    
    def populate_category_probs(self, matching_strategy='substr', normalize_marginal=True):
        '''
        Populate category probs in self.results for each matching strategy. 
        Arguments:
            matching_strategy (str): matching strategy for logprobs to categories. Supported: 'exact', 'starts_with', 'substr'. Defaults to 'substr'.
            normalize_marginal (bool): whether to normalize the marginal probability of each category. Defaults to True.
        Afterwards, there should be a column for each category with total weight.
        '''
        # get all categories
        self.categories = self.results['category'].unique().tolist()
        # populate category probs
        category_probs = self.results.apply(lambda x: self.get_category_probs(x['response'], matching_strategy), axis=1)
        # to dataframe
        category_probs = pd.DataFrame(category_probs.tolist())
        # normalize by marginal probability
        if normalize_marginal:
            category_probs = category_probs.divide(category_probs.mean(axis=0), axis=1)
        # normalize to sum to 1
        category_probs = category_probs.divide(category_probs.sum(axis=1), axis=0)
        # add to results
        self.results = pd.concat([self.results, category_probs], axis=1)
    
    def populate_scores(self):
        '''
        Populate score (0 if match, 1 otherwise) in self.results for each matching strategy.
        After running, self.results should contain columns 'score_exact', 'score_starts_with', 'score_substr'.
        '''
        # populate score
        self.results['guess'] = self.results[self.categories].idxmax(axis=1)
        self.results['score'] = 1 * (self.results['guess'] == self.results['category'])
    
    def group_by_columns(self, columns, df=None):
        '''
        Groups df by columns and returns a dataframe of the aggregated scores.
        Arguments:
            columns (list): list of columns to group by
            df (pandas dataframe): dataframe to group. Default is self.results
        '''
        if df is None:
            df = self.results
        df = df.groupby(columns).agg({'score': 'mean'})
        return df
    
    def plot_n(self, split_by=[], average_with_error_bars=False):
        '''
        Plots the n column, and splits runs by the colums in split_by.
        Arguments:
            split_by (list): list of columns to split by
            average_with_error_bars (bool): whether to plot average and error bars. Defaults to False.
        '''
        # check if split_by is a list, if not, make it a list
        if not isinstance(split_by, list):
            split_by = [split_by]
        df = self.group_by_columns(split_by + ['n'])

        for col in df.columns:
            df_col = df[col]
            if len(split_by) > 0:
                if average_with_error_bars:
                    means = df_col.groupby('n').agg('mean')
                    std = df_col.groupby('n').agg('std')
                    # plot with shaded error bars
                    plt.plot(means.index, means, label=col)
                    plt.fill_between(means.index, means - 2*std, means + 2*std, alpha=0.2)
                else:
                    n_levels = len(split_by)
                    runs = list(product(*[df_col.index.get_level_values(i).unique() for i in range(n_levels)]))
                    for run in runs:
                        if n_levels == 1:
                            run = df_col.loc[run[0]]
                        elif n_levels == 2:
                            run = df_col.loc[run[0], run[1]]
                        elif n_levels == 3:
                            run = df_col.loc[run[0], run[1], run[2]]
                        else:
                            raise ValueError('split_by must be of length <= 3')
                        plt.plot(run.index, run, alpha=.5)
            else:
                plt.plot(df_col)
            plt.title(col)
            plt.xlabel('n')
            plt.show()
        pass
    
    def plot_confusion_matrix(self):
        '''
        Plot a confusion matrix for all categories. Guesses located in 'guess' and target in 'categories' in self.results.
        '''
        # TODO - sort categories in an intelligent way
        # get categories
        categories = self.categories
        # get guesses
        guesses = self.results['guess'].unique().tolist()
        # get confusion matrix
        cm = confusion_matrix(self.results['category'], self.results['guess'])
        # plot
        # make big figure
        fig, ax = plt.subplots(figsize=(10, 10))
        # title
        plt.title('Confusion Matrix')
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=90)
        plt.yticks(tick_marks, guesses)
        plt.ylabel('Guess')
        plt.xlabel('Target')
        plt.show()
    
    def plot_category_accuracies(self):
        '''
        Plot a bar chart of accuracies, in order of accuracy.
        '''
        # get accuracies
        accuracies = self.results.groupby('category').agg({'score': 'mean'})
        # sort accuracies
        accuracies = accuracies.sort_values(by='score', ascending=False)
        # plot
        sns.barplot(accuracies.index, accuracies['score'])
        plt.ylim(0,1)
        plt.title('Accuracies')
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.subplots_adjust(bottom=0.5)
        plt.show()



if __name__ == '__main__':
    er = ExperimentResults('experiments/nyt/07-16-2021/examples_instances_runs/combined/')
    # er = ExperimentResults('experiments/nyt/07-06-21/slash/')
    # df = er.group_by_columns(['n'])
    er.plot_category_accuracies()
    er.plot_confusion_matrix()
    er.plot_n(['exemplar', 'instance'], average_with_error_bars=True)
    er.plot_n(['exemplar', 'instance'], average_with_error_bars=False)
    breakpoint()
    pass
