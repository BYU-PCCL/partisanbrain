import openai
import os
import pickle

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


class Experiment:
    """For running experiments with GPT-3 API and saving them"""

    def __init__(self, dataset, gpt_3_engine="davinci"):
        self._ds = dataset

        # See engines at https://beta.openai.com/pricing
        self._gpt_3_engine = gpt_3_engine

        # Results has form
        # {idx: [(prompt_1, response_1)...(prompt_n, response_n)]}
        self._results = {}

    def _process_prompt(self, prompt):
        """Process prompt with self._gpt_3_version version of GPT-3"""
        try:
            response = None
            # TODO: Are these arguments correct?
            # TODO: Do we ever want max_tokens > 1?
            response = openai.Completion.create(engine=self._gpt_3_engine,
                                                prompt=prompt,
                                                max_tokens=1,
                                                logprobs=100)
        # TODO: Catch more specific exception here
        except Exception as exc:
            print(exc)
        return response

    def run(self):
        """Get results from GPT-3 API"""
        for (row_idx, row_prompts) in self._ds.prompts.items():
            self._results[row_idx] = [self._process_prompt(p) for p in
                                      row_prompts]

    def save_results(self, fname):
        """Save results obtained from run method"""
        with open(fname, "wb") as f:
            pickle.dump(self._results, f)


if __name__ == '__main__':
    from survey_datasets import ExampleSurveyDataset
    ds = ExampleSurveyDataset(n_exemplars=5)
    e = Experiment(ds, gpt_3_engine="ada")
    e.run()
    e.save_results("star_wars_results.pkl")
