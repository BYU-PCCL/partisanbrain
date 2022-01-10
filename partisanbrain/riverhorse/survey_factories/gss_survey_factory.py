from ..dataset_factory import DatasetFactory
from ..surveys.gss_survey import GssSurvey


class GssFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        df['gender'] = df['gender'].apply(lambda x: x.lower())

        df['party'] = df['party'].apply(lambda x: 'Independent, closer to Republican' if 'IND,NEAR REP' in x else x)
        df['party'] = df['party'].apply(lambda x: 'Independent, closer to Democrat' if 'IND,NEAR DEM' in x else x)
        df['party'] = df['party'].apply(lambda x: 'moderate Republican' if 'NOT STR REPUBLICAN' in x else x)
        df['party'] = df['party'].apply(lambda x: 'moderate Democrat' if 'NOT STR DEMOCRAT' in x else x)
        df['party'] = df['party'].apply(lambda x: 'strong Democrat' if 'STRONG DEMOCRAT' in x else x)
        df['party'] = df['party'].apply(lambda x: 'strong Republican' if 'STRONG REPUBLICAN' in x else x)
        df['party'] = df['party'].apply(lambda x: 'Independent' if 'INDEPENDENT' in x else x)
        df['party'] = df['party'].apply(lambda x: 'other party'if 'OTHER PARTY' in x else x)

        df['ideology'] = df['ideology'].apply(lambda x: x.lower())

        df['education'] = df['education'].apply(lambda x: str(x))
        df['education'] = df['education'].apply(lambda x: 'no formal schooling' if '0.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'some elementary school' if '1.0' in x or '2.0' in x or '3.0' in x or '4.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'elementary school' if '5.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'some middle school' if '6.0' in x or '7.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'midle school' if '8.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'some high school' if '9.0' in x or '10.0' in x or '11.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'high school' if '12.0' in x else x)
        df['education'] = df['education'].apply(lambda x: 'attended college' if '13.0' in x or '14.0' in x or '15.0' in x or '16.0' in x else x)

        df['income'] = df['income'].apply(lambda x: x.lower())
        df['income'] = df['income'].apply(lambda x: 'less than $1000' if 'lt' in x else x)

        df['religion'] = df['religion'].apply(lambda x: x.title())
        df['religion'] = df['religion'].apply(lambda x: 'a member of an Eastern religion' if "Other Eastern" in x else x)
        df['religion'] = df['religion'].apply(lambda x: 'a member of a Native American religion' if "Native American" in x else x)

        df['race_ethnicity'] = df[]'race_ethnicity'].apply(lambda x: x.title())

        df['region'] = df['region'].apply(lambda x: 'New England' if 'NEW' in x else x)
        df['region'] = df['region'].apply(lambda x: 'Middle Atlantic' if 'MIDDLE' in x else x)
        df['region'] = df['region'].apply(lambda x: 'Pacific' if 'PACIFIC' in x else x)
        df['region'] = df['region'].apply(lambda x: 'East North Central' if 'E. NOR.' in x else x)
        df['region'] = df['region'].apply(lambda x: 'South Atlantic' if 'SOUTH' in x else x)
        df['region'] = df['region'].apply(lambda x: 'West South Central' if 'W. SOU.' in x else x)
        df['region'] = df['region'].apply(lambda x: 'Mountain' if 'MOUNTAIN' in x else x)
        df['region'] = df['region'].apply(lambda x: 'West North Central' if 'W. NOR.' in x else x)
        df['region'] = df['region'].apply(lambda x: 'East South Central' if 'E. SOU.' in x else x)

        df['marital_status'] = df['marital_status'].apply(lambda x: x.lower())

    def get_templates(self):
        pass

    def make_backstory1(self):
        party_dict = {
            "Independent, closer to Republican" : "identify as an Independent, closer to Republican",
            "Independent, closer to Democrat" : "idendify as an Independent, closer to Democrat",
            "moderate Republican" : "identify as a moderate Republican",
            "moderate Democrat" : "identify as a moderate Democrat",
            "strong Democrat" : "identify as a strong Democrat",
            "strong Republican" : "identify as a strong Democrat",
            "Independent" : "identify as an Independent",
            "other party" : "don\'t identify as a member of any major party"
        }
        marital_dict = {
            ""
        }
        # create a generic backstory using values in the demographics
        return (lambda row: f"I am {row['age']} years old. I am {row['gender']}."
                            f"I {party_dict[row['party']])}."
                            f"My education is {row['education'].lower()}. I am {row['ideology'].lower()}."
                            f"My total income is {row['income']}."
                            f"My religion is best described as {row['religion']}."
                            f"I am {row['race_ethnicity']}. I am from the {row['region']} region."
                            f"{marital_dict[row['marital_status']]}."
                            )

    def make_backstory1(self):
        return "I am 21 years old. I am female. I identify as a member of the Democratic party. \
                My education is some college but no degree. My total income is $40,000. \
                My religion is described as not religious. I am white, not-hispanic. I am from the West. \
                I am never married."

    def make_backstory2(self):
        return (lambda row: f"Age: {row['age']}, Gender: {row['gender']}, Political Affiliation: {row['party']}"
                            f"Education: {row['education']}, Ideology: {row['ideology']}, Total Income: {row['income']}, Religion: {row['religion']},"
                            f"Race/Ethnicity: {row['race_ethnicity']}, Region: {row['region']}, Marital Status: {row['marital_status']}")

    def make_backstory2_shot(self):
        return "Age: 21, Gender: Female, Political Affiliation: Democratic Party, Education: Some college but no degree, \
                Ideology: Extremely liberal, Income: $40,000, Religion: Not religious, Race/Ethnicity: Black, non-Hispanic, \
                Region: Northeast, Marital Status: Never married"

    def make_backstory3(self):
        return (lambda row: f"Q: What is your age?\nA: {row['age']}\n\nQ: What is your gender?\nA: {row['gender']}\n\n"
                            f"Q: What is your political affiliation?\nA: {row['party']}\n\nQ: What is your education?\nA: {row['education']}"
                            f"Q: What is your ideology?\nA: {row['ideology']}\n\nQ: What is your income?\nA: {row['income']}\n\n"
                            f"Q: What is your religion?\nA: {row['religion']}\n\n"
                            f"Q: What is your race/ethnicity?\nA: {row['race_ethnicity']}\n\nQ: What region of the country are your from?\nA: {row['region']}"
                            f"Q: What is your marital status?\nA: {row['marital_status']}")

    def make_backstory4(self):
        return (lambda row: f"Question 1: What is your age?\nAnswer 1: {row['age']}\n\nQuestion 2: What is your gender?\nAnswer 2: {row['gender']}\n\n"
                            f"Question 3: What is your political affiliation?\nAnswer 3: {row['party']}\n\nQuestion 4: What is your education?\nAnswer 4: {row['education']}"
                            f"Question 5: What is your ideology?\nAnswer 5: {row['ideology']}\n\nQuestion 6: What is your income?\nAnswer 6: {row['income']}\n\n"
                            f"Question 7: What is your religion?\nAnswer 7: {row['religion']}\n\n"
                            f"Question 8: What is your race/ethnicity?\nAnswer 8: {row['race_ethnicity']}\n\nQuestion 9: What region of the country are your from?\nAnswer 9: {row['region']}"
                            f"Question 10: What is your marital status?\nAnswer 10: {row['marital_status']}")


if __name__ == "__main__":
    factory = GssFactory(GssSurvey())
