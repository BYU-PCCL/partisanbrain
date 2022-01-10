import numpy as np
from pdb import set_trace as breakpoint
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import product
from confusion_matrix_utils import get_best_confusion_matrix

class ExperimentResults():
    def __init__(self, results_path, ends_with='.pickle', matching_strategy='substr', normalize_marginal=True):
        '''
        Initialize results object. Read in all results from results_path.
        Arguments:
            results_path (str): path to results
            ends_with (str): file extension of results. Defaults to 'pickle'
            matching_strategy (str): matching strategy for logprobs to categories. Supported: 'starts_with', 'substr'. Defaults to 'substr'.
                'starts_with' - category must start with token
                'substr' - token must be a substring of category
            normalize_marignal (bool): whether to normalize marginal probabilities. Defaults to True.
        Attributes:
            results_path: path to directory containing results
            results: pandas dataframe containing all results
                Columns:
                    'text': text of the prompt
                    'responses': response to the prompt (type: OpenAI response)
                    'I': instances index
                    'E': exemplars index
                    'N_exemplars': number of exemplars
                    'N_per_cat': number of instances per category
                    'category': true category of the instance
        '''
        # save arguments
        self.results_path = results_path
        # supported matching strategies
        self.ends_with = ends_with
        self.matching_strategies = ['starts_with', 'substr']
        self.matching_strategy = matching_strategy
        self.normalize_marginal = normalize_marginal

        # load results
        self.results = None
        self.load_results()
        # populate probs
        self.populate_category_probs()
        # populate scores
        self.populate_scores()
        #populate margin
        self.populate_margin()
    
    def populate_margin(self, picklepath='experiments/nyt/07-27-2021/ambiguity/ambiguitycandidates/ambiguity_candidates_w_margin.pickle'):
        '''Finds the margin between the probability of the correct category
        and the probability of the next category of highest weight'''
        
        if not os.path.exists(picklepath):
            df = self.results[self.categories + ['category']]
            # get logprobs of correct category
            for i, row in df.iterrows():
                correctcol = row.category
                correctlogprob = row[correctcol]
                leftovers = row.copy().drop([correctcol, 'category'])
                otherprobs = sorted(leftovers.items(), key = lambda x: x[1], reverse=True)
                # get logprobs of next highest weight
                
                runnerupcol, nextlogprob = otherprobs[0]
                # get margin
                margin = correctlogprob - nextlogprob
                df.at[i, 'margin'] = margin
                df.at[i, 'runnerup'] = runnerupcol
            self.results['margin'] = df['margin']
            self.results['runnerup'] = df['runnerup']
            self.results.to_pickle(picklepath)
    
    def read_df(self, file_name):
        '''
        Reads a file into a pandas dataframe. Supported filetypes: csv, pkl, p, pickle
        '''
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        elif file_name.endswith('.pkl') or file_name.endswith('.p') or file_name.endswith('.pickle'):
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
        if len(dfs) == 1:
            df = dfs[0]
        else:
            df = pd.concat(dfs, ignore_index=True)
        return df

    def get_all_files(self, path):
        '''
        Searches subdirectories in path and returns a list of all files ending with
        ends_with.
        '''
        matched_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                # if ends_with is list
                if isinstance(self.ends_with, list):
                    for end in self.ends_with:
                        if file.endswith(end):
                            matched_files.append(os.path.join(root, file))
                else:
                    if file.endswith(self.ends_with):
                        matched_files.append(os.path.join(root, file))
        return matched_files
    
    def load_results(self):
        '''
        Loads all files in results_path into self.results
        '''
        files = self.get_all_files(self.results_path)
        self.results = self.read_and_combine_dfs(files)
        # dropna in row 'responses'
        self.results = self.results.dropna(subset=['responses'])
        # reset index
        self.results = self.results.reset_index()
    
    def get_logprobs(self, response):
        '''
        Given an instance of gpt3 response, return the logprobs of the first sampled token.
        Returns a sorted list of tuples of (token, logprob)
        '''
        # if response is tuple, need to extract second element
        if isinstance(response, tuple):
            response = response[1]
        logprobs = response['choices'][0].logprobs.top_logprobs[0]
        # sort by logprob
        logprobs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
        return logprobs
    
    def matches(self, token, category):
        '''
        Returns True if token matches category according to the matching strategy.
        Arguments:
        '''
        # strip token and category and change to lowercase
        token = token.lower().strip()
        category = category.lower().strip()

        # check matching strategy
        if self.matching_strategy == 'starts_with':
            return category.startswith(token)
        elif self.matching_strategy == 'substr':
            return token in category
    
    def get_category_probs(self, response):
        '''
        Iterate through responses and return the relative probability for each category.
        Arguments:
            response (OpenAI GPT-3 response): gpt3 output results
        Returns:
            category_probs (dict): dictionary of category probabilities
        '''
        # get logprobs
        logprobs = self.get_logprobs(response)
        # get category probabilities
        category_probs = {cat: 0 for cat in self.categories}

        for token, logprob in logprobs:
            for category in self.categories:
                if self.matches(token, category):
                    category_probs[category] += np.exp(logprob)

        return category_probs
    
    def populate_category_probs(self):
        '''
        Populate category probs in self.results for each matching strategy. 
        Afterwards, there should be a column for each category with total weight.
        '''
        # get all categories
        self.categories = self.results['category'].unique().tolist()
        # populate category probs
        category_probs = self.results.apply(lambda x: self.get_category_probs(x['responses']), axis=1)
        # to dataframe
        category_probs = pd.DataFrame(category_probs.tolist())
        # normalize by marginal probability
        if self.normalize_marginal:
            # TODO - marginalize over just each unique run, where run is a unique set of 'instance_set_ix', 'example_set_ix', 'n_exemplars',
            category_probs = category_probs.divide(category_probs.mean(axis=0), axis=1)
        # normalize to sum to 1
        category_probs = category_probs.divide(category_probs.sum(axis=1), axis=0)
        # add to results
        self.results = pd.concat([self.results, category_probs], axis=1)
    
    def top_k_match(self, probs, labels, k):
        '''
        Arguments:
            probs (np.array): 2d probability vector array (instances x categories)
            labels (np.array): label vector array of ints. Corresponds to the index of the probs vector.
            k (int): top k
        Returns:
            top_k_match (list): list of 1 if match, 0 otherwise
        '''
        # get top k labels
        top_k_labels = np.argsort(probs)[:, -k:]
        # reshape labels to be same shape as top_k_labels
        labels = labels.repeat(k).reshape(-1, k)
        return (labels == top_k_labels).sum(axis=1)
    
    
    def populate_scores(self, max_k=10):
        '''
        Populate score (0 if match, 1 otherwise) in self.results for each matching strategy.
        Arguments:
            max_k (int): maximum top_k_accuracy to calculate. Default is 3.
        '''
        # populate score
        self.results['guess'] = self.results[self.categories].idxmax(axis=1)
        self.results['score'] = 1 * (self.results['guess'] == self.results['category'])
        # calculate top_k_accuracy
        probs = self.results[self.categories].values
        # get category as int
        labels = np.array([self.categories.index(cat) for cat in self.results['category'].tolist()])
        for k in range(1, max_k + 1):
            self.results['score_' + str(k)] = self.top_k_match(probs, labels, k)
    
    def get_average_score(self, df=None, top_k=1):
        '''
        Returns the average score.
        Arguments:
            top_k (int): top_k_accuracy to calculate. Default is 0.
        '''
        if df is None:
            df = self.results

        # get average score
        if top_k == 0:
            return df['score'].mean()
        else:
            column = 'score_' + str(top_k)
            if column not in df.columns:
                # compute populate_scores again with top_k
                self.populate_scores(top_k)
            return df[column].mean()
    
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
    
    def average_predictions(self, additional_columns=[]):
        '''
        Average predictions for a given title and category to look at an "ensembled" prediction.
        Arguments:
            additional_columns (list): list of additional columns to split over. Default is [].
        '''
        columns = ['title', 'category']
        # add aditional columns
        columns += additional_columns
        # aggregate columns = self.categories
        agg_dict = {cat: 'mean' for cat in self.categories}
        df = self.results.groupby(columns).agg(agg_dict)
        df = df.reset_index()

        df['guess'] = df[self.categories].idxmax(axis=1)
        df['score'] = 1 * (df['guess'] == df['category'])
        return df
    
    def plot_top_k_accuracies(self, df=None, max_k=3, save_path=None):
        '''
        Plots the top_k accuracy from 1 to max_k.
        '''
        if df is None:
            df = self.results
        
        # get K
        K = np.arange(1, max_k + 1)
        # get scores
        scores = [self.get_average_score(df, k) for k in K]
        # plot
        plt.plot(K, scores)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title('Top-k Accuracy')
        if save_path is not None:
            plt.savefig(save_path + '_top_k.pdf')
        else:
            plt.show()
        # clear plt
        plt.clf()
    
    def plot(self, df=None, x_variable='n_exemplars', y_variable='score', split_by=[], color_by='', average=False, save_path=''):
        '''
        Plots the results of x_variable vs. y_variable given some other variables (split_by).
        Arguments:
            x_variable (str): column to plot on x-axis
            y_variable (str): column to plot on y-axis
            split_by (list, str): list of columns to split by. Variables that you want to iterate over, keeping x and y constant. If single string, turn into list. Default is []
            color_by (str): column to color by. Must be included in split_by. Default is None.
            save_path (str): path to save figure. Default is ''
        '''
        if df is None:
            df = self.results

        # check if split_by is a list, if not, make it a list
        if not isinstance(split_by, list):
            split_by = [split_by]
        
        if color_by:
            # check if color_by is in split_by, if not, add it
            if color_by not in split_by:
                split_by.append(color_by)
            unique = df[color_by].unique()
            colors = sns.color_palette("Set1", len(unique))
            color_dict = {cat: color for cat, color in zip(unique, colors)}
        
        # add x_variable to split_by
        if x_variable not in split_by:
            split_by.append(x_variable)

        df_agg = df.groupby(split_by).agg({y_variable: 'mean'})
        df_agg = df_agg.reset_index()

        # iterate through unique tuples of columns, excluding x_variable and y_variable
        other_columns = list(set(df_agg.columns) - set([x_variable, y_variable]))
        if not average:
            # if other columns, split by other columns and then plot
            if len(other_columns) > 0:
                runs = df_agg[other_columns].drop_duplicates()
                # iterate through runs and plot x_variable vs. y_variable
                labels = []
                for run in runs.itertuples():
                    df_run = df_agg.copy()
                    for i, column in enumerate(other_columns):
                        df_run = df_run[df_run[column] == run[i+1]].copy()
                    # plot
                    if color_by:
                        i = other_columns.index(color_by)
                        label = run[i+1]
                        color = color_dict[label]
                        if label not in labels:
                            plt.plot(df_run[x_variable], df_run[y_variable], label=f'{color_by}: {label}', color=color, alpha=.6)
                            labels.append(label)
                        else:
                            plt.plot(df_run[x_variable], df_run[y_variable], color=color, alpha=.6)
                    else:
                        label = y_variable
                        if label not in labels:
                            plt.plot(df_run[x_variable], df_run[y_variable], alpha=.6, label=label)
                        else:
                            plt.plot(df_run[x_variable], df_run[y_variable], alpha=.6)
            # otherwise, just plot x_variable vs. y_variable
            else:
                label = y_variable
                plt.plot(df_agg[x_variable], df_agg[y_variable], label=label)
        else:
            means = df_agg.groupby(x_variable).agg({y_variable: 'mean'}).reset_index()
            std = df_agg.groupby(x_variable).agg({y_variable: 'std'}).reset_index()
            # plot with shaded error bars
            plt.plot(means[x_variable], means[y_variable], label=f'{y_variable}')
            plt.fill_between(means[x_variable], means[y_variable] - 2*std[y_variable], means[y_variable] + 2*std[y_variable], alpha=0.2)
        
        plt.legend()
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title(f'{y_variable} vs. {x_variable}')
        if save_path is not None:
            if average:
                plt.savefig(save_path + '_' + x_variable + '_vs_' + y_variable + '_average.pdf')
            else:
                plt.savefig(save_path + '_' + x_variable + '_vs_' + y_variable + '.pdf')
        else:
            plt.show()
        # clear plt
        plt.clf()

    def plot_confusion_matrix(self, df=None, save_path=None):
        '''
        Plot a confusion matrix for all categories. Guesses located in 'guess' and target in 'categories' in self.results.
        '''
        if df is None:
            df = self.results

        # get categories
        categories = sorted(self.categories)
        # get confusion matrix
        cm = confusion_matrix(df['category'], df['guess'], normalize='true')
        # find best matrix
        cm, categories = get_best_confusion_matrix(cm, categories)
        # plot
        # make big figure
        fig, ax = plt.subplots(figsize=(20, 20))
        # title
        plt.title('Confusion Matrix')
        plt.imshow(cm, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=90)
        plt.yticks(tick_marks, categories)
        plt.ylabel('Target')
        plt.xlabel('Guess')
        # plt.show()
        if save_path is not None:
            plt.savefig(save_path + '_confusion_matrix.pdf')
            # clear plt
            plt.clf()
        
    
    def plot_category_accuracies(self, df=None, save_path=None):
        '''
        Plot a bar chart of accuracies, in order of accuracy.
        '''
        if df is None:
            df = self.results

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
        # plt.show()
        if save_path is not None:
            plt.savefig(save_path + '_accuracies.pdf')
            # clear plt
            plt.clf()
    
if __name__ == '__main__':
    experiment_dir = 'experiments/nyt/07-27-2021/ambiguity/ambiguitytiers'
    er = ExperimentResults(experiment_dir, ends_with = 'output.pickle', normalize_marginal=True)
    # er.plot(
    #     split_by=['exemplar_method'],
    #     save_path=experiment_dir+'/plots'
    # )
    er = ExperimentResults('experiments/nyt/07-20-2021/EI', ends_with = 'EInex0.pickle'.split())
    breakpoint()

    # er.plot_category_accuracies(save_path='plots/')
    # er.plot_confusion_matrix(save_path='plots/')
    # er.plot_top_k_accuracies(max_k=10, save_path='plots/')
    er.plot(split_by='exemplar_method')

    # plot by different colors
    er.plot(
        split_by=['exemplar_set_ix', 'instance_set_ix'],
        color_by='instance_set_ix',
        save_path='plots/'
    )
    # plot means and std
    er.plot(
        split_by=['exemplar_set_ix', 'instance_set_ix'],
        average=True,
        save_path='plots/'
    )
    # plot instances
    er.plot(
        split_by = ['instance_set_ix'],
        color_by='instance_set_ix',
        save_path='plots/instance'
    )
    # plot exemplars
    er.plot(
        split_by = ['exemplar_set_ix'],
        color_by='exemplar_set_ix',
        save_path='plots/exemplar'
    )

    # with aggregate predictions
    df = er.average_predictions()
    df_n = er.average_predictions(['n_exemplars'])

    er.plot_category_accuracies(df, save_path='plots/ensemble/')
    er.plot_confusion_matrix(df, save_path='plots/ensemble/')

    # split by instance set and exemplar set
    er.plot(df_n, save_path='plots/ensemble/')
    breakpoint()
    pass
