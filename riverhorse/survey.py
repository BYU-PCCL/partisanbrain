class UserInterventionNeededError(Exception):
    """
    An exception that indicates that manual work (like
    logging in or filling out a form) is needed
    """

    def __init__(self, message):
        super().__init__(message)


class Survey:

    def __init__(self):
        # TODO: Updaate this. Currently this is just to throw
        # errors if parts of the code are not implemented.
        df = self.download_data()
        df = self.modify_data(df)
        print(self.get_dv_questions())

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
