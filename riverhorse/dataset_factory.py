from mi_codebase import Dataset

import constants as k


class SimpleDataset(Dataset):
    def __init__(self, templates, df, sample_seed=0, n=None, out_fname=None):
        self.simple_dataset_templates = templates
        super().__init__(sample_seed=sample_seed, n=n, in_fname=df, out_fname=out_fname)

    def _modify_raw_data(self, df):
        return df.copy()

    def _get_templates(self):
        return self.simple_dataset_templates


class DatasetFactory:
    def __init__(self, survey_obj, sample_seed=0, n=None):
        df = survey_obj.df

        # Alex added this to provide easier acces to the questions
        self.questions = survey_obj.get_dv_questions()

        df = self.modify_data(df)
        templates = self.get_templates()

        # Get the list of DV colnames
        dv_colnames = list(set(df.columns) - set(k.DEMOGRAPHIC_COLNAMES))

        # Get the list of demographic colnames present
        present_dems = list(set(df.columns) & set(k.DEMOGRAPHIC_COLNAMES))

        survey_name = survey_obj.get_survey_name()[: -len("Survey")].lower()

        # For each DV colname, make a dataset object
        for dv_colname in dv_colnames:
            sub_df = df.copy()[present_dems + [dv_colname]]
            sub_df = sub_df.rename(columns={dv_colname: "ground_truth"})
            SimpleDataset(
                templates=templates[dv_colname],
                df=sub_df,
                sample_seed=sample_seed,
                n=n,
                out_fname=f"data/{survey_name}/{dv_colname}/ds.pkl",
            )

    def modify_data(self, df):
        """
        Modify the df that is the output of Survey
        object initialization to prepare it for templatizing.
        Critically, make sure that DV responses match the responses
        in get_templates().
        """
        raise NotImplementedError

    def get_templates(self):
        """
        Return a dictionary of dictionaries. The keys of the
        outside dictionary should be the names of dv columns
        from Survey object dataframe. The value dictionaries
        should have keys that are informative template names
        and values that are template lambda functions.
        """
        raise NotImplementedError


if __name__ == "__main__":
    factory = DatasetFactory()
