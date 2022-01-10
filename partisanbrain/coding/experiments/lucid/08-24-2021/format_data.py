import pickle
from pdb import set_trace as bp
import pandas as pd


#Load the pickle called final_results_positivity.pkl from this directory
with open('experiments/lucid/final_results.pkl', 'rb') as f:
	final_results = pickle.load(f)

for key in final_results.keys():
	for party in 'Republicans Democrats'.split():
		for coder in "human gpt3".split():
			dic = final_results[key][party][coder]
			df = pd.DataFrame(dic.items(), columns = 'id data'.split())
			df['prob'] = [el[0][0] for el in df.data]
			df['party'] = party
			df['horg'] = coder
			df.drop('data', axis = 1, inplace = True)
			df.to_csv(f'experiments/lucid/results_{key}_{party}_{coder}.csv', index = False)




bp()
