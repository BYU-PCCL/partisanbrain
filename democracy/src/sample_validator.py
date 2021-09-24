# Algorithm to randomly sample a dataframe and see how good the random samples are, keeping the best one
import pandas as pd

class SampleValidator:

	def __init__(self):
		# Setting up data
		all_data = pd.read_csv("anes_timeseries_2020_csv_20210719.csv")
		demographic_col_names =  ["V201507x", "V201600", "V201018", "V201510", "V201549x"]
		data = all_data[demographic_col_names]
		data = data.rename(columns={"V201507x": 'age',
					  			    "V201600" : "gender",
					  			    "V201018" : "party", 
					  			    "V201510" : 'education',
					  			    "V201549x": 'race'})
		data.dropna(axis=0)
		data = data[~data['age'].isin([-9])]
		data = data[~data['gender'].isin([-9])]
		data = data[~data['party'].isin([-9,-8,-1,4,5])]
		data = data[~data['education'].isin([-9,-8,4,5,95])]
		data = data[~data['race'].isin([-9,-8])]

		# moving ages into an age_range column to match the demographics of participants.
		data.insert(0, "age_range", [""]*len(data))
		for index, row in data.iterrows():
			if row['age'] in list(range(18, 25)):
				data.at[index, 'age_range'] = "18-24"
			elif row['age'] in list(range(25, 35)):
				data.at[index, 'age_range'] = "25-34"
			elif row['age'] in list(range(35, 45)):
				data.at[index, 'age_range'] = "35-44"
			elif row['age'] in list(range(45, 55)):
				data.at[index, 'age_range'] = "45-54"
			elif row['age'] in list(range(55, 65)):
				data.at[index, 'age_range'] = "55-64"
			elif row['age'] in list(range(65, 76)):
				data.at[index, 'age_range'] = "65-75"
			else:
				data.at[index, 'age_range'] = "75+"
		data = data.drop(columns='age')

		# updating education groupings. New groupings are:
			# 1 - No high school degree 
			# 2 - High school graduate
			# 3 - Some college
			# 4 - Bachelor's degree
			# 5 - Graduate degree 
		data = data.replace(to_replace=6, value=4)
		data = data.replace(to_replace=7, value=5)
		data = data.replace(to_replace=8, value=5)

		self.data = data
		
	# shuffling the dataframe so we get a random sample
	def shuffle_data(self):
		self.shuffled_data = self.data.sample(frac=1).reset_index(drop=True)

	# from the dataframe take the first n republicans and democrats
	def sample_democrats_and_republicans(self, n):
		democrats = self.shuffled_data[self.shuffled_data['party'] == 1]
		democrats = democrats.drop(columns='party')
		republicans = self.shuffled_data[self.shuffled_data['party'] == 2]
		republicans = republicans.drop(columns='party')
		# get a sample of 200 each
		self.democrats = democrats.head(n)
		self.republicans = republicans.head(n)

	def make_df_descriptive(self, df):
		descriptive_df = df.copy()
		value_encodings = {'gender':
					   			{1: "female",
						   		 2: "male"},
						   'education':
						   		{1: "no high school degree",
						   		 2: "high school graduate",
						   		 3: "some college",
						   		 4: "bachelor's degree",
						   		 5: "graduate degree"},
						   'race': 
						   		{1: "White",
						   		 2: "Black",
						   		 3: "Hispanic",
						   		 4: "Asian/Native Hawaiian/Pacific Islander",
						   		 5: "Native American/Alaskan Native"}
						   }
		col_names = ['gender', 'education', 'race']
		for col in col_names:
			descriptive_df[col] = descriptive_df[col].map(value_encodings[col])

		return descriptive_df


	# sees how close the random sample of democrats matches our desired sample
	def validate_democrat_sample(self):
		# expected percentages. Division on ones where the % didn't add up to 1
		true_prop_dictionary = {'age_range': 
							   		{'18-24': .13/.99,
							   		 '25-34': .17/.99,
							   		 '35-44': .17/.99,
							   		 '45-54': .15/.99,
							   		 '55-64': .17/.99,
							   		 '65-75': .15/.99,
							   		 '75+'  : .05/.99},
							   	'gender':
							   		{1: .57,
							   		 2: .43},
							   	'education':
							   		{1: .07,
							   		 2: .24,
							   		 3: .26,
							   		 4: .26,
							   		 5: .17},
							   	'race': 
							   		{1: .54/1.01,
							   		 2: .2/1.01,
							   		 3: .16/1.01,
							   		 4: .05/1.01,
							   		 5: .02/1.01,
							   		 6: .04/1.01}
							   	}
		col_names = ['age_range', 'gender', 'education', 'race']
		total_diff = 0
		for name in col_names:
			props = self.democrats[name].value_counts(normalize=True)
			true_props = true_prop_dictionary[name]
			col_value_counts = props.tolist()
			for i, col_value in enumerate(props.index.tolist()):
				total_diff += abs(true_props[col_value] - col_value_counts[i])

		return(total_diff / 20) # 20 is number of possible column values

	# sees how close the random sample of republicans matches our desired sample
	def validate_republican_sample(self):
		# expected percentages. Division on ones where the % didn't add up to 1
		true_prop_dictionary = {'age_range': 
							   		{'18-24': .07,
							   		 '25-34': .14,
							   		 '35-44': .16,
							   		 '45-54': .17,
							   		 '55-64': .21,
							   		 '65-75': .16,
							   		 '75+'  : .09},
							   	'gender':
							   		{1: .47,
							   		 2: .53},
							   	'education':
							   		{1: .07/1.01,
							   		 2: .28/1.01,
							   		 3: .32/1.01,
							   		 4: .23/1.01,
							   		 5: .11/1.01},
							   	'race': 
							   		{1: .82,
							   		 2: .03,
							   		 3: .08,
							   		 4: .03,
							   		 5: .02,
							   		 6: .02}
							   	}
		col_names = ['age_range', 'gender', 'education', 'race']
		total_diff = 0
		for name in col_names:
			props = self.republicans[name].value_counts(normalize=True)
			true_props = true_prop_dictionary[name]
			col_value_counts = props.tolist()
			for i, col_value in enumerate(props.index.tolist()):
				total_diff += abs(true_props[col_value] - col_value_counts[i])

		return(total_diff / 20) # 20 is number of possible column values

	def find_good_random_sample(self, n):
		max_iterations = 10000
		iterations = 0
		lowest_democrat_error = 1
		lowest_republican_error = 1
		while iterations <= max_iterations:
			self.shuffle_data()
			self.sample_democrats_and_republicans(n)
			percent_error_democrats = self.validate_democrat_sample()
			if percent_error_democrats < lowest_democrat_error:
				lowest_democrat_error = percent_error_democrats
				print("new lowest democrat error:")
				print(lowest_democrat_error)
				descriptive_df = self.make_df_descriptive(self.democrats)
				descriptive_df.to_csv("best_democrat_sample.csv", index=False)
			percent_error_republicans = self.validate_republican_sample()
			if percent_error_republicans < lowest_republican_error:
				lowest_republican_error = percent_error_republicans
				print("new lowest republican error:")
				print(lowest_republican_error)
				descriptive_df = self.make_df_descriptive(self.republicans)
				descriptive_df.to_csv("best_republican_sample.csv", index=False)
			iterations += 1


validator = SampleValidator()
validator.find_good_random_sample(100)
