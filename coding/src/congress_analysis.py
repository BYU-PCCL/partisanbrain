from readpickle import rpkl
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import seaborn as sns
from congress_categories import categories

categories = list(categories.values())

exp_name = 'slash'

def score_responses(exp_dir):
    #TODO add a strict match criterion

    responses = rpkl(exp_dir)
    data = []
    for response in responses:
        responsetext = response['response']
        bestresponsetext = responsetext['choices'][0]
        logprobs = bestresponsetext.logprobs
        toplogprobs = logprobs.top_logprobs
        fstlp = toplogprobs[0]
        keepn = 100
        firstntoplogprobs = sorted(fstlp.items(), key = lambda x: x[1], reverse = True)[:keepn]

        target = response['target']

        weights = {cat: 0 for cat in categories}
        for token, weight in firstntoplogprobs:
            token = token.strip().lower()
            for cat in categories:
                if token in cat.lower():
                    weights[cat] += np.exp(weight)

        weights['target'] = target
        weights['title'] = response['text'].split('\n')[-1].split(' /')[0]

        data.append(weights)

    data = pd.DataFrame(data)
    normalized = data.drop(columns=['target', 'title'])
    #Normalize by average category weight
    normalized = normalized / normalized.mean()
    #Normalize across all categories
    normalized = normalized.div(normalized.sum(axis=1), axis=0)
    # extract guesses
    guesses = normalized.idxmax(1)
    # add true labels
    normalized['target'] = data['target']
    normalized['title'] = data['title']
    normalized.to_pickle(os.path.join('experiments',exp_dir,'normalized.pkl'))
    acc = (guesses == data['target']).mean()
    return acc

def plot_avg_acc(experiments):
    accs = []
    for experiment in experiments:
        accs.append(score_responses(experiment))
    plt.plot(accs)
    plt.savefig('experiments/congress/07-09-21/fewshot/avgaccs.pdf')
    plt.show()



def plot_confusion_matrix(y_true, y_pred):
    labels = sorted_categories
    ticks = np.arange(len(sorted_categories))
    mapping = {cat: i for (cat, i) in zip(sorted_categories, ticks)}
    f = lambda x: np.array([mapping[val] for val in x])
    y_true, y_pred = f(y_true), f(y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    plt.imshow(matrix)
    plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels=labels)
    plt.show()



#sns.barplot(cats,scores)
#plt.ylim(0,1)
#plt.title('Accuracies for NYT codes')
##plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right')
#plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
#plt.subplots_adjust(bottom=0.5)
#plt.savefig(f'experiments/nyt/07-06-21/{exp_name}/accuracies4.png')
#plt.show()
#
#plot_confusion_matrix(targets, guesses)
if __name__ == '__main__':
    #score_responses('experiments/nyt/07-06-21/slash')
    plot_avg_acc([f'congress/07-09-21/fewshot/{n}' for n in np.arange(21)])