import pandas as pd
import numpy as np

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
                    if t.startswith(token):
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

    def __init__(self, df):
        '''
        Instantiates a Postprocessor object.
        df (pd.DataFrame): DataFrame containing the experiment results.
            columns:
                - 'resp' (dict: str->float): dictionary with the top n logprobs
        '''
        self.df = df.copy()
        self.calculate_probs()
        self.calculate_coverage()
        self.normalize_probs()
    
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

# if __name__ == '__main__':
#     df = pd.read_pickle('infra/lms/example.pkl')
#     df['resp'] = df['responses']
#     postprocessor = Postprocessor(df)
#     df = postprocessor.df
#     breakpoint()
#     pass
# 