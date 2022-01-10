import pathlib as Path

DEMOGRAPHIC_COLNAMES = ["age", "gender", "party",
                        "ideology", "education", "income",
                        "religion", "race_ethnicity",
                        "region", "marital_status"]
N_DVS = 20
SURVEY_DATA_PATH = Path(__file__).parent.absolute() / "survey_data"
