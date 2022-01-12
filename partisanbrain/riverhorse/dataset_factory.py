from ..mutualinf.dataset import Dataset
from . import constants as k
import os


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
        self.dv_colnames = list(set(df.columns) - set(k.DEMOGRAPHIC_COLNAMES))

        # Get the list of demographic colnames present
        self.present_dems = list(set(df.columns) & set(k.DEMOGRAPHIC_COLNAMES))

        survey_name = survey_obj.get_survey_name()[: -len("Survey")].lower()

        # TODO for Chris, uncomment this line below and pic whichever dv you want to look at
        # self.sample_templates(df, dvs="whites_understand_blacks")

        # TODO for Chris, comment this for loop if you run the line above
        # For each DV colname, make a dataset object
        for dv_colname in self.dv_colnames:
            sub_df = df.copy()[self.present_dems + [dv_colname]]
            sub_df = sub_df.rename(columns={dv_colname: "ground_truth"})

            data_dir = f"data/{survey_name}/{dv_colname}"

            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            sub_df = sub_df.dropna(subset=["ground_truth"])
            SimpleDataset(
                templates=templates[dv_colname],
                df=sub_df,
                sample_seed=sample_seed,
                n=n,
                out_fname=os.path.join(data_dir, "ds.pkl"),
            )

    def sample_templates(self, df, dvs=None):
        if dvs is None:
            dvs = self.dv_colnames
        elif not isinstance(dvs, list):
            dvs = [dvs]

        templates = self.get_templates()

        for dv in dvs:
            sub_df = df[self.present_dems + [dv]]
            sub_df = sub_df.dropna()
            for type, (template, tokens) in templates[dv].items():
                row = sub_df.sample().iloc[0]
                print(type, template(row), sep="\n", end="\n\n")
                input()

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
