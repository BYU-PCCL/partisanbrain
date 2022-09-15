from pathlib import Path

DEMOGRAPHIC_COLNAMES = [
    "id",
    "age",
    "gender",
    "party",
    "ideology",
    "education",
    "income",
    "religion",
    "race_ethnicity",
    "region",
    "marital_status",
]
N_DVS = 20
SURVEY_DATA_PATH = Path(__file__).parent.absolute() / "survey_data"

DATA_PATH = Path(__file__).parent.absolute() / 'data'
FILLED_TEMPLATES_PATH = Path(__file__).parent.absolute() / 'filled_templates'
