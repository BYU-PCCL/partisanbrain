import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from pdb import set_trace as breakpoint
import pickle
import openai
import os

def read_data(path):
    '''
    Given a path to a csv, read the data into a pandas dataframe.
    Columns:
        prompt (str): the prompt
        possible_answers (list of strings): potential responses
        api_resp (dict): the response from the api
    '''
    data = pd.read_csv(path)
    # convert possible_answers to a list
    data['possible_answers'] = data['possible_answers'].apply(lambda x: eval(x))
    # convert api_resp to a dict
    data['api_resp'] = data['api_resp'].apply(lambda x: eval(x))
    return data

def get_logprobs(response):
    '''
    Given an instance of gpt3 response, return the logprobs of the first sampled token.
    Returns a sorted list of tuples of (token, logprob)
    '''
    # logprobs = response['choices'][0].logprobs.top_logprobs[0]
    logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
    # sort by logprob
    logprobs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
    return logprobs

def matches(token, category, matching_strategy):
    '''
    Returns True if token matches category according to the matching strategy.
    Arguments:
        token: the token to match
        category: the category to match
        matching_strategy: the matching strategy to use
            Accepted values:
                'starts_with': token starts with category
                'substr': token is a substring of category
                'exact': token is the exact category
    '''
    # strip token and category and change to lowercase
    token = token.lower().strip()
    category = category.lower().strip()

    # check matching strategy
    if matching_strategy == 'starts_with':
        return category.startswith(token)
    elif matching_strategy == 'substr':
        return token in category
    elif matching_strategy == 'exact':
        return category == token
    else:
        raise ValueError('Invalid matching strategy')

def parse_response(response, candidates, matching_strategy='starts_with'):
    '''
    Given the response, measure the total probability mass
    on each candidate.
    '''
    # strip and lowercase all candidates
    candidates = [c.lower().strip() for c in candidates]
    logprobs = get_logprobs(response)
    # get category probabilities
    cand_probs = {cand: 0 for cand in candidates}

    for token, logprob in logprobs:
        # see if lower and strip is a candidate
        token = token.lower().strip()
        for cand in candidates:
            if matches(token, cand, matching_strategy):
                cand_probs[cand] += np.exp(logprob)
    # normalize, and store coverage
    coverage = sum(cand_probs.values())
    cand_probs = {cand: prob/coverage for cand, prob in cand_probs.items()}
    cand_probs['coverage'] = coverage
    return cand_probs

def parse_responses(df):
    '''
    Given a dataframe of responses, parse the responses and add columns to df.
    Get the response from the 'api_resp' column, and the candidates from the 'possible_answers' column.
    '''
    df['parsed_resp'] = df.apply(lambda x: parse_response(x['api_resp'], x['possible_answers']), axis=1)
    return df

def plot_treatments(treatment_dict, y_label='', x_label='Treatments', save_path=''):
    '''
    Given a dictionary of treatment results, plot the results.
    Arguments:
        treatment_dict: a dictionary of the treatment results. The keys are the treatment names,
            and the values are the treatment results (array-like).
    '''
    treatments = list(treatment_dict.keys())
    means, stds = [], []
    for treatment in treatments:
        results = treatment_dict[treatment]
        mean = np.mean(results)
        std = np.std(results)
        means.append(mean)
        stds.append(std)
    plt.bar(treatments, means, yerr=stds)
    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def t_test(treatment_dict):
    '''
    Given a dictionary of treatment results, perform a t-test on each treatment.
    Arguments:
        treatment_dict: a dictionary of the treatment results. The keys are the treatment names,
            and the values are the treatment results (array-like).
    '''
    treatments = list(treatment_dict.keys())
    p_values = pd.DataFrame(index=treatments, columns=treatments)
    for treatment1 in treatments:
        for treatment2 in treatments:
            if treatment1 == treatment2:
                continue
            else:
                p_value = ttest_ind(treatment_dict[treatment1], treatment_dict[treatment2])[1]
                p_values.loc[treatment1, treatment2] = p_value
    return p_values


if __name__ == '__main__':
    # prompt = ''' Would you say that you agree or disagree that mexican food is good?
    # I would say that I'''

    # response = openai.Completion.create(
        # prompt=prompt,
        # max_tokens=1,
        # logprobs=100,
        # engine='ada',
    # )

    # candidates = ['agree', 'disagree']
    # print(parse_response(response, candidates))

    # treatments = {
        # 'a': 3 * np.random.rand(100) + 1,
        # 'b': 2 * np.random.rand(100) + 1.4,
    # }
    # # plot_treatments(treatments, y_label='Mean', x_label='Treatments', save_path='test.png')
    # print(t_test(treatments))

    # read in data from data/kalmoe_republican.csv and data/passive_republican.csv
    # merge in both into one dataframe
    data1, data2 = read_data('data/kalmoe_republican.csv'), read_data('data/passive_republican.csv')
    data1, data2 = parse_responses(data1), parse_responses(data2)
    breakpoint()
    pass
