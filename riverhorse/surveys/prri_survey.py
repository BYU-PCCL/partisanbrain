from parent_dir import Survey, UserInterventionNeededError


class PrriSurvey(Survey):

    def __init__(self):
        super().__init__()

    def download_data(self):
        pass

    def modify_data(self, df):
        pass

    def get_dv_questions(self):
        pass


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = PrriSurvey()
