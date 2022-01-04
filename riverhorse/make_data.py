# Each of these functions should download the data and put the
# data for each DV in data/dv_name/raw.csv. So, for example,
# if you call make_anes_data(), it should download the data
# from the ANES website and then file structure would look like
#
# data
#   |- intermediate
#       |- full_anes_download.csv
#   |- protect_env
#       |- raw.csv
#   |- government_temp
#       |- raw.csv
#
# Make sure to write out in natural language (in comments)
# anything that has to be done manually (like making an account)
# in detail.


import pandas as pd


def make_addhealth_data():
    pass


def make_anes_data():
    # 1. Navigate to https://electionstudies.org/data-center/2020-time-series-study/
    # 2. Log in with an account
    # TODO: Finish writing out steps for downloading
    # 4. Move the downloaded anes_timeseries_2020_csv_20210719.csv to
    # data/intermediate/full_anes_download.csv
    # 5. The below code will do the rest

    df = pd.read_csv('data/intermediate/full_anes_download.csv')

    # Do any handling that is common for all DVs here
    pass

    demographic_colnames = {"age": "V201507x",
                            "gender": "V201600",
                            # "gender": "V202637",
                            "party": "V201018",
                            "education": "V201510",
                            "ideology": "V201200",
                            # "income": "V201607",  # RESTRICTED
                            "religion": "V201458x",
                            "race": "V201549x",
                            "region": "V203003",
                            "marital": "V201508"}
    dv_codes = {"protect_env": "V201321"}  # Eventually 20 DVs here

    for dv_name, code in dv_codes.items():
        # Extract columns for demographics and this DV
        dv_df = df[list(demographic_colnames.values()) + [code]]
        # Save the DV data
        dv_df.to_pickle(f"data/{dv_name}/raw.pkl")


def make_baylor_data():
    pass


def make_cces_data():
    pass


def make_gss_data():
    pass


def make_pew67_data():
    pass


def make_pew78_data():
    pass


def make_prri_data():
    pass
