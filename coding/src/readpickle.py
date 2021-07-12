import pickle

def rpkl(experiment):
    with open(f'experiments/{experiment}/prompts.p', 'rb') as f:
        results = pickle.load(f)
        #for result in results:
        #    print(result['response'])
        #    print(result['text'])
        #    print(result['target'])
    return results
