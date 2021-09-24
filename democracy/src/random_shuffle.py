# Algorithm to randomly sample a dataframe and see how good the random samples are, keeping the best one
import pandas as pd

class Shuffler:

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
		print(self.data['gender'].unique())
		
	# shuffling the dataframe so we get a random sample
	def shuffle_data(self):
		self.shuffled_data = self.data.sample(frac=1).reset_index(drop=True)

	# from the dataframe take the first 200 republicans and democrats
	def sample_democrats_and_republicans(self):
		democrats = self.data[self.data['party'] == 1]
		republicans = self.data[self.data['party'] == 2]
		# get a sample of 200 each
		self.democrats = democrats.head(200)
		self.republicans = republicans.head(200)

	# sees how close the random sample of democrats matches our desired sample
	# EXPECTED PERCENTAGES:
		# Sex: 
			# 57% F (1)
			# 43% M (2)
		# Age: 
			# 13% 18-24
			# 17% 25-34
			# 17% 35-44
			# 15% 45-54
			# 17% 55-64
			# 15% 65-75
			# 5% 75+
		# Ethnicity:
			# White (non-Hispanic) (1) 54%
			# Black (non-Hispanic) (2) 20%
			# Hispanic (3) 16%
			# Asian / Native Hawaiian / Pacific Islander (4) 5%
			# Native American / Alaskan Native (5) 2%
			# Multiple Races (non-Hispanic) (6) 4%
		# Education:
			# No high school degree (1) 7%
			# High school graduate (2) 24%
			# Some college (3) 26%
			# Bachelor's degree (4) 26%
			# Graduate degree (5) 17%
	def validate_democrat_sample(self):
		pass

	# sees how close the random sample of republicans matches our desired sample
	#EXPECTED PERCENTAGES: 
		# Sex: 
			# 47% F (1)
			# 53% M (2)
		# Age: 
			# 7% 18-24
			# 14% 25-34
			# 16% 35-44
			# 17% 45-54
			# 21% 55-64
			# 16% 65-75
			# 9% 75+
		# Ethnicity:
			# White (non-Hispanic) (1) 82%
			# Black (non-Hispanic) (2) 3%
			# Hispanic (3) 8%
			# Asian / Native Hawaiian / Pacific Islander (4) 3%
			# Native American / Alaskan Native (5) 2%
			# Multiple Races (non-Hispanic) (6) 2%
		# Education:
			# No high school degree (1) 7%
			# High school graduate (2) 28%
			# Some college (3) 32%
			# Bachelor's degree (4) 23%
			# Graduate degree (5) 11%
	def validate_republican_sample(self):
		pass

shuffler = Shuffler()
shuffler.shuffle_data()
shuffler.sample_democrats_and_republicans()
print(shuffler.democrats)
print(shuffler.republicans)




