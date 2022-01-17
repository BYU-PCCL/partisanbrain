from ..mutualinf.dataset import Dataset
from . import constants as k
from pdb import set_trace as bp
import pandas as pd
import os
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
                prompt_tok_list = []
                for type, (template, tokens) in templates[dv].items():
                    f.write("\n\n==============================\n\n")
                    row = sub_df.sample().iloc[0]
                    f.write(type)
                    f.write("\n\n")
                    prompt = template(row)
                    prompt_tok_list.append((prompt,tokens))
                    f.write(prompt)
                f.write("\n\n==============================\n\n")
                bad_prompts = self.playground(prompt_tok_list)
                f.write(f"\n\nBAD TEMPLATES (N={len(bad_prompts)}): These failed playgrounding\n\n==============================\n\n")
                formatted_bad_prompts = "\n\n==============================\n\n".join(bad_prompts)
                f.write(f"The following prompts were rejected: {formatted_bad_prompts}")


    def _substitute(self, token):
        """This just converts Jurassic's output to normal text."""
        return token.replace("▁", " ").replace("<|newline|>", "\n")

    def _expected_response_token_equality(self, expected_response, token):
        """Returns whether an expected response corresponds to the output token."""
        token = token.strip().lower()
        expected_response = expected_response.lower()
        return token.startswith(expected_response) or expected_response.startswith(
            token
        )

    def _get_prob_mass(self, expected_response, token_logprob_pairs):
        """Returns the probability mass associated with an expected_response."""
        total_logprob = 0
        for token, logprob in token_logprob_pairs:
            if self._expected_response_token_equality(expected_response, token):
                total_logprob += logprob
        return total_logprob

    def _is_enough_prob_mass(self, expected_responses, token_logprob_pairs, thresh):
        """Returns whether a survey response has at least thresh probability mass associated with it."""
        total_prob = 0
        for expected_response in expected_responses:
            total_prob += self._get_prob_mass(expected_response, token_logprob_pairs)
        return total_prob >= thresh

    def _any_expected_response_top_k(self, expected_responses, token_logprob_pairs, k=7):
        """Returns whether or not any of the expected responses are "equal" to 
        any of the top k logprob tokens."""

        top_k_tokens = [token for token, _ in token_logprob_pairs[:k]]
        are_expected_responses_top_k = [
            any(
                map(
                    lambda token: self._expected_response_token_equality(
                        expected_response, token
                    ),
                    top_k_tokens,
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
        prompts,
        api_token=None,
        thresh=0.001,
        min_survey_responses=2,
        min_tokens=5,
        top_k=7,
    ):
        """
        This playgrounds the input prompts and returns which did not pass the playgrounding steps.
        TODO: Add a way to cache finished prompts and only re-run the ones you need.

        Args: 
            prompts[list]: List of tuples (prompt[str], token_dict[dict])
            api_token[str]: The AI21 Lab API token 
        """

        if api_token is None:
            api_token = os.getenv('AI21_API_KEY')
        bad_prompts = []
        for prompt, token_dict in prompts:
            response = requests.post(
                "https://api.ai21.com/studio/v1/j1-jumbo/complete",
                headers={"Authorization": f"Bearer {api_token}"},
                json={
                    "prompt": prompt,
                    "numResults": 1,
                    "maxTokens": 1,
                    "topKReturn": 64,
                },
            )
            token_logprob_pairs = response.json()["completions"][0]["data"]["tokens"][0]["topTokens"]
            token_logprob_pairs = [
                (self._substitute(d["token"]), np.exp(d["logprob"])) for d in token_logprob_pairs
            ]

            results = []

            # For every single survey response
            for survey_response, expected_responses in token_dict.items():
                if not isinstance(expected_responses, list):
                    expected_responses = [expected_responses]

                # Record whether

                # (1) all the expected responses have enough collective probability mass associated with them
                is_enough_prob_mass = self._is_enough_prob_mass(
                    expected_responses, token_logprob_pairs, thresh
                )
                # and

                # (2) any of the expected responses are in the top k tokens
                is_survey_response_top_k = self._any_expected_response_top_k(
                    expected_responses, token_logprob_pairs, k=top_k
                )

                # and save the result for every survey response
                results.append(
                    (survey_response, is_enough_prob_mass, is_survey_response_top_k)
                )

            # Get any possible response
            all_expected_responses = []
            for expected_responses in token_dict.values():
                if not isinstance(expected_responses, list):
                    expected_responses = [expected_responses]
                all_expected_responses += expected_responses

            # All survey responses need a bit of probability mass
            has_enough_prob_mass = all(map(lambda result: result[1], results))
            # At least l=min_survey_responses of possible survey responses are in top k tokens
            enough_survey_responses_in_top_k = (
                sum(map(lambda result: result[2], results)) >= min_survey_responses
            )
            # At least of top 7 tokens are in response set
            enough_top_k_tokens_in_expected_responses = self._has_enough_tokens(
                all_expected_responses, [k for k,_ in token_logprob_pairs], min_tokens=min_tokens, k=top_k
            )

            # This is where we decide if a prompt is good or bad
            if not all([has_enough_prob_mass, 
                        enough_survey_responses_in_top_k, 
                        enough_top_k_tokens_in_expected_responses]):
                bad_prompts.append(prompt)
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
