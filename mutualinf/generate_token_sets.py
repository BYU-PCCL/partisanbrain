import pandas as pd
import os

class Process:
    def __init__(self, pathnames):
        self.pathnames = pathnames
        self.filenames = self.parsefilenames()
        self.save_path = 'full_token_sets/'

    def parsefilenames(self):
        # split filename on / and take last element
        return [pathname.split('/')[-1] for pathname in self.pathnames]

    def process(self):
        for i in range(len(self.pathnames)):
            pathname = self.pathnames[i]
            try:
                # open pkl file and save to df
                df = pd.read_pickle(pathname)
            except Exception as e:
                print(f'ERROR reading pkl: {e}')
                exit(-1)

            # get the token sets
            resp_lst = list(df['resp'])
            
            # extract the keys from the token sets and add to key_set
            key_set = set()
            for x in resp_lst:
                x_keys = set(x.keys())
                key_set = key_set.union(x_keys)
            
            # check is dest directory exists
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            # save key_set to txt file
            with open(self.save_path + self.filenames[i].replace('.pkl','.txt'), 'w+') as f:
                f.write(str(key_set))


if __name__ == '__main__':
    filenames = []
    # traverse the directory and get all the pkl files
    for root, dirs, files in os.walk('data/'):
        for file in files:
            if file.endswith('.pkl') and file.startswith('exp_results') and '_processed' not in file:
                filenames.append(os.path.join(root, file))
    print(f'Found filenames: {filenames}')
    process = Process(filenames)
    process.process()
    print('done')