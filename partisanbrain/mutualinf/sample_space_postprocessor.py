import pandas as pd
import numpy as np
from tqdm import tqdm
import os


class Postprocessor:

    def __init__(self, results_fname, save_fname, matching_strategy=None):
        '''
        Instantiates a Postprocessor object.
        matching_strategy (str): Strategy for matching tokens. Can be 'startswith', 'exact', or None.
        '''

        # Read in dataframe specified by results_fname
        self.df = pd.read_pickle(results_fname)

        self.df['ground_truth'] = self.df['ground_truth'].astype(str)

        # get number of instances where 'resp' is missing
        num_missing = self.df.loc[self.df.resp.isnull()].shape[0]
        # print Dropping {} instances with missing responses
        print(f'Dropping {num_missing} instances with missing responses from', results_fname)
        # drop na where 'resp' is missing
        self.df = self.df.dropna(subset=['resp'])

        # calculate guess
        self.df = self.calculate_guess(self.df)

        # calculate accuracy
        self.df = self.calculate_accuracy(self.df)

        # save df
        self.df.to_pickle(save_fname)
    
    def calculate_guess(self, df):
        '''
        Gets the guess for each instance. Trim 'resp' by whitespace or punctuation and take first word.
        '''
        df = df.copy()
        df['guess'] = df['resp'].str.strip().str.split().str[0]
        # also remove all punctuation: . , ; : - _ ' "
        punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
        for punct in punctuation:
            # replace all punctuation with whitespace
            df['guess'] = df['guess'].str.replace(punct, '')
            # same for ground truth
            df['ground_truth'] = df['ground_truth'].str.replace(punct, '')
        # make guess and ground truth lower case
        df['guess'] = df['guess'].str.lower()
        df['ground_truth'] = df['ground_truth'].str.lower()
        # strip white space
        df['guess'] = df['guess'].str.strip()
        df['ground_truth'] = df['ground_truth'].str.strip()
        return df

    def calculate_accuracy(self, df):
        '''
        Calculates the accuracy of the model. Adds a column called 'accuracy' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        df['accuracy'] = df['ground_truth'] == df['guess']
        # random = np.random.randint(1, len(df), 10) 
        # df.iloc[random][['guess', 'ground_truth', 'accuracy']]  

        # if row['ground_truth'] starts with argmax(row['probs']) stripped and lowercase, then it's correct
        # def accuracy_lambda(row):
        #     if row['ground_truth'].lower().strip().startswith(guess):
        #         return 1
        #     else:
        #         return 0
        # df['accuracy'] = df.apply(accuracy_lambda, axis=1)

        return df


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