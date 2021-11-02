import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def exponentiate(d):
    '''
    Exponentiates a dictionary's probabilities.
    '''
    return {k: np.exp(v) for k, v in d.items()}

def normalize(d):
    '''
    Normalizes a dictionary to sum to one.
    '''
    return {k: v/sum(d.values()) for k, v in d.items()}

def collapse_lower_strip(d):
    '''
    Collapses a dictionary's probabilities after doing lower case and strip, combining where needed.
    '''
    new_d = {}
    for k, v in d.items():
        new_k = k.lower().strip()
        # check if empty
        if new_k:
            # if already present, add to value
            if new_k in new_d:
                new_d[new_k] += v
            # if not present, add to dictionary
            else:
                new_d[new_k] = v
    return new_d

def collapse_token_sets(d, token_sets, matching_strategy='startswith'):
    '''
    Collapses a dictionary's probabilities combining into token sets.
    Args:
        d (dict): Dictionary of probabilities.
        token_sets (dict:str->list, list): Dictionary of token sets, where keys are categories and
            values are lists of tokens. If token_sets is a list, it is assumed that the keys are
            the lists of tokens.
        matching_strategy (str): Strategy for matching tokens. Can be 'startswith' or 'exact'.
    Returns:
        new_d (dict): Dictionary of probabilities after collapsing.
    '''
    # if token_sets is a list, convert to dictionary
    if isinstance(token_sets, list):
        token_sets = {t: [t] for t in token_sets}
    # make sure all items in dictionary are lists
    for k, v in token_sets.items():
        if not isinstance(v, list):
            token_sets[k] = [v]
    # create new dictionary
    new_d = {cat: 1e-10 for cat in token_sets.keys()}
    # iterate over tokens and probs in d
    for token, prob in d.items():
        # iterate over token sets
        for category, tokens in token_sets.items():
            # if token is in the token set, add to new dictionary
            match = False
            for t in tokens:
                if matching_strategy == 'startswith':
                    if t.lower().strip().startswith(token):
                        match = True
                elif matching_strategy == 'exact':
                    if token == t:
                        match = True
            if match:
                new_d[category] += prob
    return new_d

def prob_function(row):
    '''
    Collapses a row of probabilities.
    Args:
        row (pandas.Series): Series of probabilities.
    Returns:
        d (dict: str->float): Dictionary of probabilities after collapsing.
    '''
    d = row['resp']
    # logprobs to probs
    d = exponentiate(d)
    # lower strip collapse
    d = collapse_lower_strip(d)
    # if 'token_sets' in row, collapse token sets
    if 'token_sets' in row:
        # make sure 'token_sets' isn't None or an empty list
        if row['token_sets']:
            # check if matching strategy exists
            if 'matching_strategy' in row:
                d = collapse_token_sets(d, row['token_sets'], row['matching_strategy'])
            else:
                d = collapse_token_sets(d, row['token_sets'])
    return d

def coverage(d):
    '''
    Returns the sum of the values of d.
    '''
    return sum(d.values())


class Postprocessor:

    def __init__(self, results_fname, save_fname, matching_strategy=None):
        '''
        Instantiates a Postprocessor object.
        matching_strategy (str): Strategy for matching tokens. Can be 'startswith', 'exact', or None.
        '''

        # Read in dataframe specified by results_fname
        self.df = pd.read_pickle(results_fname)

        self.df['ground_truth'] = self.df['ground_truth'].astype(str)
        # # TODO - remove? to fix earlier bug
        # if 'imdb' in results_fname:
        #     # make 'token_sets' ['positive', 'negative']
        #     self.df['token_sets'] = [['positive', 'negative']] * len(self.df)
        # if 'boolq' in results_fname:
        #     # 'ground_truth' to string
        #     self.df['ground_truth'] = self.df['ground_truth'].astype(str)
        #     # where 'template_name' is 'few-shot3', change token set
        #     token_set = {'True': ['yes'], 'False': ['no']}
        #     def f(row):
        #         if row['template_name'] == 'few-shot3':
        #             return token_set
        #         else:
        #             return row['token_sets']
        #     self.df['token_sets'] = self.df.apply(f, axis=1)

        # get number of instances where 'resp' is missing
        num_missing = self.df.loc[self.df.resp.isnull()].shape[0]
        # print Dropping {} instances with missing responses
        print(f'Dropping {num_missing} instances with missing responses from', results_fname)
        # drop na where 'resp' is missing
        self.df = self.df.dropna(subset=['resp'])

        if matching_strategy:
            if matching_strategy not in ['startswith', 'exact']:
                msg = f"{matching_strategy} is not a valid matching strategy"
                raise RuntimeError(msg)
            self.df['matching_strategy'] = matching_strategy
        self.calculate_probs()
        self.calculate_coverage()
        self.normalize_probs()

        # calculate mutual information
        self.df = self.calculate_mutual_information(self.df)

        # calculate accuracy
        self.df = self.calculate_accuracy(self.df)

        # calculate correct weight
        self.df = self.calculate_correct_weight(self.df)

        # save df
        self.df.to_pickle(save_fname)

    def prob_dict_to_arr(self, d):
        '''
        Converts a probability dictionary into an array of probabilities.
        Args:
            d (dict: str->float): dictionary of probabilities for each category/token.
        Returns:
            arr (np.array): array of probabilities.
        '''
        arr = np.array([d[k] for k in d])
        return arr

    def agg_prob_dicts(self, dicts):
        '''
        Given a list of probability dictionaries, aggregate them.
        '''
        n = len(dicts)
        agg_dict = {}
        for d in dicts:
            for k, v in d.items():
                if k not in agg_dict:
                    agg_dict[k] = v / n
                else:
                    agg_dict[k] += v / n
        return agg_dict

    def get_marginal_distribution(self, df, groupby='template_name'):
        '''
        Calculates the marginal distribution over categories.
        '''
        marginal_df = df.groupby(by=groupby)['probs'].agg(self.agg_prob_dicts)
        # series to df
        marginal_df = pd.DataFrame(marginal_df)
        return marginal_df

    def calculate_correct_weight(self, df):
        '''
        Calculates the correct_weight. Adds a column called 'correct_weight' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        # Our function for calculating weight on ground truth
        get_correct_weight = lambda row: row['probs'].get(row['ground_truth'], 0)

        # Calculate conditional entropy for each row
        df['correct_weight'] = df.apply(get_correct_weight, axis=1)

        return df

    def entropy(self, arr):
        '''
        Given an array of probabilities, calculate the entropy.
        '''
        return -sum(arr * np.log(arr))

    def calculate_conditional_entropy(self, df):
        '''
        Calculates the conditional entropy, up to a constant. Adds a column called 'conditional_entropy' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        entropy_lambda = lambda row: self.entropy(self.prob_dict_to_arr(row['probs']))

        # Calculate entropy for each row
        df['conditional_entropy'] = df.apply(entropy_lambda, axis=1)

        return df

    def calculate_mutual_information(self, df, groupby='template_name'):
        '''
        Calculate the mutual information between the template and the output distribution.
        '''
        # H(Y) - H(Y|X) method
        # first, calculate conditional entropy
        df = self.calculate_conditional_entropy(df)
        # get marginal distributions
        marginal_df = self.get_marginal_distribution(df, groupby)
        # get entropy
        entropy_lambda = lambda row: self.entropy(self.prob_dict_to_arr(row['probs']))
        marginal_df['entropy'] = marginal_df.apply(entropy_lambda, axis=1)
        # function to apply per row
        def mutual_inf(row):
            index = row[groupby]
            mutual_info = marginal_df.loc[index]['entropy'] - row['conditional_entropy']
            return mutual_info

        # apply function to each row
        df['mutual_inf'] = df.apply(mutual_inf, axis=1)

        return df

    def calculate_probs(self):
        '''
        Population the 'probs' column in the dataframe.
        '''
        self.df['probs'] = self.df.apply(prob_function, axis=1)

    def calculate_coverage(self):
        coverage_lambda = lambda row: coverage(row['probs'])
        self.df['coverage'] = self.df.apply(coverage_lambda, axis=1)

    def normalize_probs(self):
        '''
        Normalize the 'probs' column to sum to 1.
        '''
        self.df['probs'] = self.df['probs'].apply(normalize)

    def calculate_accuracy(self, df):
        '''
        Calculates the accuracy of the model. Adds a column called 'accuracy' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        # if row['ground_truth'] starts with argmax(row['probs']) stripped and lowercase, then it's correct
        def accuracy_lambda(row):
            # guess is argmax of row['probs'] dict
            guess = max(row['probs'], key=row['probs'].get)
            # lower and strip
            guess = guess.lower().strip()
            if row['ground_truth'].lower().strip().startswith(guess):
                return 1
            else:
                return 0
        df['accuracy'] = df.apply(accuracy_lambda, axis=1)

        return df

def get_files_to_process():
    '''
    Step down into the data subdirectory, and get all files in all subdirectories that have
    'exp_results' in them, end with 'pkl', and don't include 'processed'.
    '''
    files_to_process = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if 'exp_results' in file and file.endswith('pkl') and 'processed' not in file:
                # if processed file already exists, don't process it again
                if file.replace('.pkl', '_processed.pkl') not in files:
                    files_to_process.append(os.path.join(root, file))
    return files_to_process

def process(files):
    for input_fname in tqdm(files):
        try:
            # save_fname is same name, but replace .pkl with _processed.pkl
            save_fname = input_fname.replace('.pkl', '_processed.pkl')
            # process
            Postprocessor(input_fname, save_fname)
        except Exception as e:
            print('Error processing {}'.format(input_fname))
            print(e)

def process_all():
    files_to_process = get_files_to_process()
    file_string = '\n'.join(files_to_process)
    print(f'Processing: {file_string}')
    process(files_to_process)

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='file with results to use')
    # get dataset arg
    args = parser.parse_args()
    input_fname = args.input

    if input_fname == 'all':
        process_all()
    else:
        # save_fname is same name, but replace .pkl with _processed.pkl
        save_fname = input_fname.replace('.pkl', '_processed.pkl')

        # process
        Postprocessor(input_fname, save_fname)