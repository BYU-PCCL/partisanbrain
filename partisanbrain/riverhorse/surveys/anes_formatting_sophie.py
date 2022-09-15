"""@author: Sophie Gee: PCCL Princess"""

def for_chris(row):
    # create a generic backstory using values in the demographics
    return (
        f"Age: {int(row['age'])}"
        f"Gender: {row['gender']}"
        f"Political party: {row['party']}"
        f"Ideology: {row['ideology']}"
        f"Education: {row['education']}"
        f"Income: {row['income']}"
        f"Religion: {row['religion']}"
        f"Race/ Ethnicity: {row['race_ethnicity']}"
        f"Region: {row['region']}"
        f"Marital status {row['marital_status']}: "
        f"Question: Four years ago, in 2016, Hillary Clinton ran on the Democratic ticket against Donald Trump for the Republicans. Do you remember for sure whether or not you voted in that election? Which one did you vote for?"
        f"1. Hillary Clinton"
        f"2. Donald Trump"
        f"5. Other (specify)"
        f"Answer:"
    )

if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = AnesSurveySophie()
    df = s.download_data()
    s.modify_data(df).to_csv("ANES_for_josh.csv")