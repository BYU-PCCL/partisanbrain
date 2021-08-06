import numpy as np
import pandas as pd
import openai
from pdb import set_trace as breakpoint
from tqdm import tqdm
from matplotlib import pyplot as plt

class LikertSampler:
    def __init__(self, possible_responses, engine='davinci'):
        '''
        Arguments:
            possible_responses (list): list of possible responses
            engine (str): which GPT-3 model to use
        '''
        self.possible_responses = possible_responses
        self.engine = engine
        # generate tree
        self.base_tree = self.generate_tree(possible_responses)
        # count the number of nodes that are not leaf nodes (i.e. have children)
        non_leaf_count = 1 + sum([1 for word in self.base_tree if len(self.base_tree[word]) > 0])
        # number of times we will have to ping the API per prompt
        self.n_per_prompt = non_leaf_count
        # print so the user knows how many API calls will be made
        print(possible_responses)
        print('{} API calls will be made per prompt'.format(self.n_per_prompt))

    def generate_tree(self, possible_responses):
        '''
        Given a list of possible responses, generate a branching tree of all possible responses, split by word.
        Arguments:
            possible_responses (list): list of possible responses
        Returns:
            tree (dict): tree of possible responses, split by word
        '''
        tree = {}
        for response in possible_responses:
            words = response.split()
            current_node = tree
            for word in words:
                if word not in current_node:
                    current_node[word] = {}
                current_node = current_node[word]
        return tree

    def get_response(self, prompt):
        '''
        Gets a response from the OpenAI API.
        '''
        # get only one token
        max_tokens = 1
        # return maximum logprobs
        logprobs = 100
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=logprobs
        )
        return response

    def get_logprobs(self, response):
        '''
        Given a response, extract the logprobs.
        '''
        logprobs = response['choices'][0].logprobs.top_logprobs[0]
        # sort by logprob
        logprobs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
        return logprobs


    def get_probs(self, words, response):
        '''
        Get the probability of each word in the response.
        Arguments:
            words (list): list of words
            response (dict): response from the OpenAI API
        Returns:
            probs (dict): probability of each word
        '''
        logprobs = self.get_logprobs(response)
        probs = {word: 0 for word in words}
        for token, logprob in logprobs:
            for word in words:
                if token.strip().lower() == word.strip().lower():
                    # exponentiate to make absolute probability
                    probs[word] += np.exp(logprob)
        
        # normalize
        probs = {word: prob / sum(probs.values()) for word, prob in probs.items()}
        return probs

    def populate_probs(self, tree):
        '''
        Given a tree, calculate the raw probability of each response.
        Arguments:
            tree (dict): tree of possible responses, split by word
        Returns:
            probs (dict): probability of each possible response
        '''
        probs = {}
        for option in self.possible_responses:
            words = option.split()
            prob = 1
            # multiply probability of each word in tree
            current_node = tree
            for word in words:
                current_node = current_node[word]
                prob *= current_node['prob']
            probs[option] = prob
        return probs


    def sample_likert(self, prompt, tree=None):
        '''
        Given a prompt, run through the OpenAI API and get the absolute probability of each possible response.
        Recursively search through the tree in a recursive way.
        Arguments:
            prompt (str): prompt to respond to
            tree (dict): tree of possible responses, split by word
        Returns:
            probs (dict): probability of each possible response
        '''
        # if tree is None, use the base tree
        if tree is None:
            tree = self.base_tree.copy()
        
        # base case - tree is empty
        if len(tree) == 0:
            return {}
        
        # copy tree
        tree = tree.copy()

        # recursively step through the tree, adding word to prompt at each step
        for word in tree:
            tree[word] = self.sample_likert(prompt + ' ' + word, tree[word])

        # get probability of each word in tree
        response = self.get_response(prompt)
        probs = self.get_probs(tree.keys(), response)
        for word, prob in probs.items():
            tree[word]['prob'] = prob

        # add response to tree
        tree['response'] = response
        
        return tree
    
    def __call__(self, prompt):
        '''
        Given a prompt, run through the OpenAI API and get the absolute probability of each possible response.
        Return the full tree, along with the probability of each response.
        Arguments:
            prompt (str): prompt to respond to
        '''
        tree = self.sample_likert(prompt)
        probs = self.populate_probs(tree)
        return probs, tree



if __name__ == '__main__':
    five_point_likert = [
        'strongly disagree',
        'disagree',
        'neither',
        'agree',
        'strongly agree',
    ]

    likert = LikertSampler(five_point_likert, 'ada')
    text = 'Dogs are cute'
    prompt = f'''State how much you agree or disagree with the following statements. Respond with only 'strongly agree', 'agree', 'neither', 'disagree', or 'strongly disagree'.

    I like dogs: strongly agree
    I think that taxes should be higher: strongly disagree
    Connecticut is a good state to raise a child: neither
    I like the idea of a global warming solution: agree
    {text}:'''

    probs, tree = likert(prompt)

    # big plot
    plt.figure(figsize=(10,8))
    # plot relative probabilities
    plt.bar(range(len(probs)), probs.values(), align='center')
    plt.xticks(range(len(probs)), probs.keys())
    # rotate xticks
    plt.xticks(rotation=30)
    plt.title(text)
    # save as pdf
    name = text.replace(' ', '').lower()
    plt.savefig(f'likert-{name}.pdf')
    plt.show()