from ..mutualinf.dataset import Dataset
from . import constants as k
import os
from pdb import set_trace as breakpoint
import requests
import numpy as np


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
        # TODO - this is a bit of a hack, might be a bug here.
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
            try:
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
                print(f"Created dataset for {dv_colname}")
            except Exception as e:
                print(f"Failed to create dataset for {dv_colname}")
                print(e)


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

    def _substitute(self, token):
        """This just converts Jurassic's output to normal text."""
        return token.replace("▁", " ").replace("<|newline|>", "\n")

    def _expected_response_token_equality(self, expected_response, token):
        """Returns whether an expected response corresponds to the output token."""
        token = token.lower()
        expected_response = expected_response.lower()
        return token.startswith(expected_response) or expected_response.startswith(
            token
        )

    def _get_prob_mass(self, expected_response, tokens):
        """Returns the probability mass associated with an expected_response."""
        total_prob = 0
        for token, prob in tokens:
            token = token.strip()
            if self._expected_response_token_equality(expected_response, token):
                total_prob += prob
        return total_prob

    def _is_enough_prob_mass(self, expected_responses, tokens, thresh):
        """Returns whether a survey response has at least thresh probability mass associated with it."""
        total_prob = 0
        for value in expected_responses:
            total_prob += self._get_prob_mass(value, tokens)
        return total_prob >= thresh

    def _is_survey_response_top_k(self, expected_responses, tokens, k=7):
        """Returns whether or not a survey response is found in the output tokens."""

        are_expected_responses_top_k = [
            any(
                map(
                    lambda token: self._expected_response_token_equality(
                        expected_response, token
                    ),
                    tokens[:k],
                )
            )
            for expected_response in expected_responses
        ]

        is_survey_response_top_k = any(are_expected_responses_top_k)

        return is_survey_response_top_k

    def _has_enough_tokens(self, all_expected_responses, tokens, min_tokens=5, k=7):
        """Returns whether or not min_tokens of the top k tokens correspond to survey responses."""
        are_tokens_top_k = [
            any(
                map(
                    lambda expected_response: self._expected_response_token_equality(
                        expected_response, token
                    ),
                    all_expected_responses,
                )
            )
            for token in tokens[:k]
        ]
        return sum(are_tokens_top_k) >= min_tokens

    def playground(
        self,
        api_token,
        prompts,
        thresh=0.001,
        min_survey_responses=2,
        min_tokens=5,
        top_k=7,
    ):
        """
        This playgrounds the input prompts and returns which did not pass the playgrounding steps.
        """

        bad_prompts = []
        for prompt, token_dict in prompts:
            response = requests.post(
                "https://api.ai21.com/studio/v1/j1-jumbo/complete",
                headers={f"Authorization": "Bearer {api_token}"},
                json={
                    "prompt": prompt,
                    "numResults": 1,
                    "maxTokens": 1,
                    "topKReturn": 64,
                },
            )
            tokens = response.json()["completions"][0]["data"]["tokens"][0]["topTokens"]
            tokens = [
                (self._substitute(d["token"]), np.exp(d["logprob"])) for d in tokens
            ]

            results = []

            for survey_response, expected_responses in token_dict.items():
                if not isinstance(expected_responses, list):
                    expected_responses = [expected_responses]

                is_enough_prob_mass = self._is_enough_prob_mass(
                    expected_responses, tokens, thresh
                )
                is_survey_response_top_k = self._is_survey_response_top_k(
                    expected_responses, tokens, k=top_k
                )

                results.append(
                    (survey_response, is_enough_prob_mass, is_survey_response_top_k)
                )

            # Get any possible response
            all_expected_responses = sum(
                [expected_responses for expected_responses in token_dict.values()], []
            )

            # All survey responses need a bit of probability mass
            has_enough_prob_mass = all(map(lambda result: result[1], results))
            # At least 2 of possible survey responses are in top 7
            has_enough_top_k = (
                sum(map(lambda result: result[2], results)) >= min_survey_responses
            )
            # 5 of top 7 tokens are in response set
            has_enough_tokens = self._has_enough_tokens(
                all_expected_responses, tokens, min_tokens=min_tokens, k=top_k
            )

            # This is where we decide if a prompt is good or bad
            if not all([has_enough_prob_mass, has_enough_top_k, has_enough_tokens]):
                bad_prompts += prompt

        return bad_prompts

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
