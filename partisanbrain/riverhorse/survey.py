from . import constants as k
import os
import pandas as pd
import re
import warnings


class UserInterventionNeededError(Exception):
    """
    An exception that indicates that manual work (like
    logging in or filling out a form) is needed
    """

    def __init__(self, message):
        super().__init__(message)


class Survey:

    def __init__(self, force_recreate=False):

        # Check if f"survey_data/{survey_dir}/full_data.pkl" exists
        survey_dir = self.camel_to_snake_case(self.get_survey_name())
        full_data_path = k.SURVEY_DATA_PATH / f"{survey_dir}/full_data.pkl"
        if force_recreate or (not os.path.exists(full_data_path)):

            # Make sure the directory we're saving to eventually
            # exists. We're calling this early so subclasses don't need
            # to worry about it.
            data_dir = k.SURVEY_DATA_PATH / f"{survey_dir}/"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            df = self.download_data()
            df = self.modify_data(df)

            # Ensure that self.get_dv_questions() is implemented
            self.get_dv_questions()

            # Check that df looks correct
            self.check_col_for_each_demographic(df)
            # self.check_correct_num_of_dv_cols(df)
            self.check_dv_col_question_keys_align(df)

            # Drop NAs (this might be done in individual subclasses
            # but it's here to make sure it happens)
            df = self.drop_nas(df)

            # Save df
            df.to_pickle(full_data_path)
        else:
            # Load df from pickle
            df = pd.read_pickle(full_data_path)

        self._df = df

    @property
    def df(self):
        return self._df

    def get_survey_name(self):
        return self.__class__.__name__

    def camel_to_snake_case(self, string):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()

    def drop_nas(self, df):
        """
        Drops rows with missing values for any of the demographic columns
        """
        # Check which demographic columns are present in df
        present_demographics = set(df.columns) & set(k.DEMOGRAPHIC_COLNAMES)
        return df.dropna(subset=present_demographics)

    def check_col_for_each_demographic(self, df):
        """
        Warns if df doesn't have a columnn for each
        demographic variable in k.DEMOGRAPHIC_COLNAMES.
        """
        for demographic_colname in k.DEMOGRAPHIC_COLNAMES:
            if demographic_colname not in df.columns:
                warnings.warn(
                    (f"{self.get_survey_name()} dataframe is missing "
                     f"column {demographic_colname}")
                )

    def check_correct_num_of_dv_cols(self, df):
        """
        Raises an exception if the number of DV columns in df
        is not equal to k.N_DVS.
        """
        dv_colnames = set(df.columns) - set(k.DEMOGRAPHIC_COLNAMES)
        n_dv_cols = len(dv_colnames)
        if n_dv_cols != k.N_DVS:
            msg = (f"{self.get_survey_name()} dataframe has {n_dv_cols} != "
                   f"{k.N_DVS} DV columns")
            raise Exception(msg)

    def check_dv_col_question_keys_align(self, df):
        """
        Make sure that the keys of self.get_dv_questions() and
        the values of the non-demographic column names are the same.
        This also ensures no extra columnns beyond demographics
        and DVs are allowed.
        """
        dv_colnames = set(df.columns) - set(k.DEMOGRAPHIC_COLNAMES)
        if set(self.get_dv_questions().keys()) != dv_colnames:
            msg = (f"{self.get_survey_name()} dataframe DV column names "
                   f"do not match get_dv_questions() keys")
            raise Exception(msg)

    def download_data(self):
        """
        Download the survey data from the source and put
        it into a pandas dataframe. Then return the dataframe.
        Saving intermediate files is entirely optional. If you
        do save intemediate data it should be saved to
        riverhorse/survey_data/name_of_your_dataset/filename.csv
        where name_of_your_dataset is one of "prri", "anes," "pew,"
        "addhealth," "baylor," "gss," or "cces." The filename
        would be something like "raw.csv." To
        indicate that manual work needs to be done, use the
        UserInterventionNeededError exception with a helpful
        message along with if statements to check for intermediate
        files that are the result of manual action.
        """
        raise NotImplementedError

    def modify_data(self, df):
        """
        Take df (a pandas dataframe) and return a new dataframe
        that has 30 columns. The ten demographic columns  should have
        exact names "age," "gender," "party," "ideology,"
        "education," "income," "religion," "race_ethnicity,"
        "region," and "marital_status." If multiple columns of the
        original data have relevant information, combine them together.
        Really try to have all ten demographics represented, but
        if you can't find them all in the data you don't need
        to include them all.
        The 20 DV columns should have *descriptive* names for each
        DV. Importantly, responses for demographic and DV columns
        should be interpretable strings (not numbers or codes)
        that are *as close as possible* to exactly what is in the codebook.
        You do not need to save anything as part of this method.
        This is also where you will clean up the data by dropping
        individuals who don't have all the demographic variables
        *present in the data*. Do not drop individuals who don't
        have all the DV variables present in the data.
        """
        raise NotImplementedError

    def get_dv_questions(self):
        """
        Return a dictionary that maps from the DV column names you
        chose in modify_data to the exact question for that DV from the
        codebook. Do not modify question text. Just use exactly what is
        in the codebook in full. Output of this function will look like
        {"voted_trump": "Did you vote for Donald Trump?",
         "climate_legislation": "Do you support climate legislation?",
         ...}
        """
        raise NotImplementedError
