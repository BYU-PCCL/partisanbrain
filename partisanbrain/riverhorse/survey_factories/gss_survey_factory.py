from ..dataset_factory import DatasetFactory
from ..surveys.gss_survey import GssSurvey


class GssFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=n)

    def modify_data(self, df):
        pass

    def get_templates(self):
        pass


if __name__ == "__main__":
    factory = GssFactory(GssSurvey())
