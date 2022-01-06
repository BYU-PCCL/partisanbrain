from parent_dir import DatasetFactory


class GssFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=None)

    def modify_data(self, df):
        pass

    def get_templates(self):
        pass


if __name__ == "__main__":
    factory = GssFactory()
