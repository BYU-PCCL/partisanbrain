import pandas as pd 
import numpy as np

class FormatANES:

	def __init__(self, filepath):
		all_data = pd.read_csv(filepath)
		demographic_col_names =  ["V201507x", "V201600", "V201018", "V201510", "V201200", 
								  "V201458x", "V201549x", "V203003", "V201508", "V201103"]
		data = all_data[demographic_col_names]
		data = data.rename(columns={"V201507x": 'age',
					  			    "V201600" : "gender",
					  			    "V201018" : "party", 
					  			    "V201510" : 'education',
					  			    "V201200" : "ideo",
					  			    "V201458x": "religion",
						            "V201549x": 'race',
						            "V203003" : "region" ,
						            "V201508" : "marital",
						            "V201103" : "2016_presidential_vote"})

		data = data[~data['2016_presidential_vote'].isin([-9, -8, -1, 3, 4, 5])]

		data['age'] = data['age'].astype(str)
		for index, row in data.iterrows():
			if row['age'] == '-9':
				data.at[index, 'age'] = np.nan

		value_encodings = {'gender':
					   			{1  : "female",
						   		 2  : "male",
						   		 99 : np.nan},
						   'education':
						   		{1: "less than high school",
						   		 2: "high school graduate",
						   		 3: "some college but no degree",
						   		 4: "trade school",
						   		 5: "associate degree",
						   		 6: "Bachelor's degree",
						   		 7: "Master's degree",
						   		 8: "professional school degree",
						   		 -9 : np.nan,
						   		 -8 : np.nan,
						   		 95 : np.nan},
						   'race': 
						   		{1: "White",
						   		 2: "Black",
						   		 3: "Hispanic",
						   		 4: "Asian/Native Hawaiian/Pacific Islander",
						   		 5: "Native American/Alaskan Native",
						   		 6: np.nan,
						   		 -9 : np.nan,
						   		 -8 : np.nan},
							'party':
								{1: "Democrat",
								 2: "Republican",
								 -9 : np.nan,
								 -8 : np.nan,
								 -1 : np.nan,
								 4  : "None/Independent",
								 5  : np.nan},
							'ideo':
								{-9 : np.nan,
								 -8 : np.nan,
								 1  : "Extremely Liberal",
								 2  : "Liberal",
								 3  : "Slightly Liberal",
								 4  : "Moderate",
								 5  : "Slightly conservative",
								 6  : "Conservative",
								 7  : "Extremely conservative",
								 99 : np.nan},
							'religion':
								{-1 : np.nan,
								 1 : "Mainline Protestant",
								 2 : "Evangelical Protestant",
								 3 : "Black Protestant",
								 4 : "Undifferentiated Protstant",
								 5 : "Roman Catholic",
								 6 : "Other Christian",
								 7 : "Jewish",
								 8 : np.nan,
								 9: "not religious"},
							'region':
								{1 : "Northeast",
								 2 : "Midwest",
								 3 : "South",
								 4 : "West"},
							'marital':
								{-9 : np.nan,
								 -8 : np.nan,
								 1  : "married",
								 2  : "married",
								 3  : "widowed",
								 4  : "divorced",
								 5  : "separated",
								 6  : "never married"},
							'2016_presidential_vote':
								{1 : "Hillary Clinton",
								 2 : "Donald Trump"}

						   }
		col_names = ['gender', 'education', 'race', 'party', 'ideo',
					 'religion', 'region', 'marital', '2016_presidential_vote']
		descriptive_df = data.copy()
		for col in col_names:
			descriptive_df[col] = descriptive_df[col].map(value_encodings[col])
		descriptive_df.to_csv("formatted_anes.csv", index=False)




if __name__ == "__main__":
	formatter = FormatANES("anes_timeseries_2020_csv_20210719.csv")
