import pickle

def rpkl(pklpath):
    with open(pklpath, 'rb') as f:
        results = pickle.load(f)
    return results
