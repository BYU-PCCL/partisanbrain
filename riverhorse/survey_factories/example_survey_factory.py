from parent_dir import DatasetFactory


class ExampleFactory(DatasetFactory):

    def __init__(self, survey_obj, sample_seed=0, n=None):
        super().__init__(survey_obj=survey_obj,
                         sample_seed=sample_seed,
                         n=None)

    # def get_backstory_1(self):
    #     # You're probably going to want to make other convenience
    #     # functions for forming backstories, doing logic for
    #     # demographics, etc.
    #     pass

    def modify_data(self, df):
        # Here is where you would modify the dataframe
        # to prepare it for get_templates. For example,
        # you might want to reword things for convenience.
        # Importantly, we'd change DV responses to match the
        # ones we're looking for in get_templates. So for the
        # papusa and whiteboard questions we'd switch responses
        # to be either yes or no.
        pass

    def get_templates(self):
        # Continuing from survey/example_survey.py example
        # Don't use these prompts as good examples for prompt types -
        # use the curriculum for that
        return {
            "papusas_best_food": {
                "qa": (lambda row: ("Q: What is your gender?\nA: I "
                                    f"am {row['gender']}.\nQ: Are papusas "
                                    "the best food?\nA:"), ["yes", "no"]),
                # More templates here
            },
            "dark_light_whiteboard": {
                "friend_asked": (lambda row: (f"I am {row['gender']}. When "
                                              "asked if dark whiteboards "
                                              "are better than light "
                                              "ones I said"), ["yes", "no"]),
                # More templates here
            },
            # More DVs here
        }


if __name__ == "__main__":
    factory = ExampleFactory()
