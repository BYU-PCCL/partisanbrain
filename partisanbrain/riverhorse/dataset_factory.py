from ..mutualinf.dataset import Dataset
from . import constants as k
from pdb import set_trace as breakpoint
import pandas as pd
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

        self.survey_obj = survey_obj

        # Alex added this to provide easier acces to the questions
        # TODO - this is a bit of a hack, might be a bug here.
        self.questions = survey_obj.get_dv_questions()

        df = self.modify_data(df)

        # df = self.modify_data(df)
        templates = self.get_templates()

        # Get the list of DV colnames
        self.dv_colnames = list(set(df.columns) - set(k.DEMOGRAPHIC_COLNAMES))

        # Get the list of demographic colnames present
        # Including processed demographic colnames
        self.present_dems = list(set(df.columns) - set(self.questions.keys()))


        survey_name = survey_obj.get_survey_name()[: -len("Survey")].lower()
        self.survey_name = survey_name

        # TODO for Chris, uncomment this line below and pic whichever dv you want to look at
        # self.sample_templates(df, dvs="whites_understand_blacks")

        # TODO for Chris, comment this for loop if you run the line above
        # For each DV colname, make a dataset object
        # for dv_colname in self.dv_colnames:
        for dv_colname in ['voting_frequency']:
            try:
                sub_df = df.copy()[self.present_dems + [dv_colname]]
                sub_df = sub_df.rename(columns={dv_colname: "ground_truth"})
                data_dir = f"{k.DATA_PATH}/{survey_name}/{dv_colname}"
                shotsfname = os.path.join(data_dir, "shots.pkl")
                #data_dir = f"data/{survey_name}/{dv_colname}"

                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                # Sample 5 instances for few-shot exemplars and drop them so as
                # to not corrupt the test set

                sub_df = sub_df.dropna(subset=["ground_truth"])
                shot_df = sub_df.sample(n=5, random_state=0)
                sub_df.drop(shot_df.index, inplace=True)
                SimpleDataset(
                    templates={key:val for key,val in templates[dv_colname].items() if not key.endswith("shot")},
                    df=shot_df,
                    n=None,
                    out_fname=shotsfname,
                )
                SimpleDataset(
                    templates=templates[dv_colname],
                    df=sub_df,
                    sample_seed=sample_seed,
                    n=n,
                    out_fname=os.path.join(data_dir, "ds.pkl"),
                )
                print(f"Created dataset for {dv_colname}")
            except Exception as e:
                print(f"Failed to create dataset for {dv_colname}")
                print(e)

    def get_shots(self, dv_colname, template_name, n, sep):
        #Load the pickle in data_dir called ds.pkl"
        survey_name = self.survey_name
        data_dir = f"{k.DATA_PATH}/{survey_name}/{dv_colname}"
        shotsfname = os.path.join(data_dir, "shots.pkl")
        shotsdf = pd.read_pickle(shotsfname)
        shotsdf = shotsdf[shotsdf.template_name == template_name]
        # Add the prompt and ground truth columns
        shotsdf['shots'] = shotsdf.prompt + " " + shotsdf.ground_truth
        return sep.join(shotsdf.shots.sample(n=n, random_state=0).tolist())

    def sample_templates(self, df, dvs=None):
        if dvs is None:
            dvs = self.dv_colnames
        elif not isinstance(dvs, list):
            dvs = [dvs]

        templates = self.get_templates()


        for dv in dvs:
            filled_templates_dir = f"{k.FILLED_TEMPLATES_PATH}/{self.survey_name}/{dv}"

            if not os.path.exists(filled_templates_dir):
                os.makedirs(filled_templates_dir)
            with open(os.path.join(filled_templates_dir, 'filled_templates.txt'), "w") as f:
                sub_df = df[self.present_dems + [dv]]
                sub_df = sub_df.dropna()
                for type, (template, tokens) in templates[dv].items():
                    row = sub_df.sample().iloc[0]
                    f.write(type)
                    f.write("\n\n")
                    f.write(template(row))
                    f.write("\n\n==============================\n\n")

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
    factory = DatasetFactory(n=200)