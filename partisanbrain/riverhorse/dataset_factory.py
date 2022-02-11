from ..mutualinf.dataset import Dataset
from . import constants as k
from pdb import set_trace as breakpoint
import pickle
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

        # Get the list of DV colnames
        self.dv_colnames = list(set(df.columns) - set(k.DEMOGRAPHIC_COLNAMES))

        df = self.modify_data(df)

        # df = self.modify_data(df)
        templates = self.get_templates()

        # Get the list of demographic colnames present
        # Including processed demographic colnames
        self.present_dems = list(set(df.columns) - set(self.questions.keys()))

        survey_name = survey_obj.get_survey_name()[: -len("Survey")].lower()
        self.survey_name = survey_name

        # TODO for Chris, uncomment this line below and pic whichever dv you want to look at
        # self.sample_templates(df, dvs="whites_understand_blacks")

        # TODO for Chris, comment this for loop if you run the line above
        # For each DV colname, make a dataset object
        # for dv_colname in ["vote_2016"]:
        for dv_colname in self.dv_colnames:
            try:
                sub_df = df.copy()[self.present_dems + [dv_colname]]
                sub_df["ground_truth"] = sub_df[dv_colname]
                # sub_df = sub_df.rename(columns={dv_colname: "ground_truth"})
                data_dir = f"{k.DATA_PATH}/{survey_name}/{dv_colname}"
                shotsfname = os.path.join(data_dir, "shots.pkl")
                # data_dir = f"data/{survey_name}/{dv_colname}"

                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                # Sample 5 instances for few-shot exemplars and drop them so as
                # to not corrupt the test set

                sub_df = sub_df.dropna(subset=["ground_truth"])
                shot_df = sub_df.sample(n=10, random_state=0)
                sub_df.drop(shot_df.index, inplace=True)
                SimpleDataset(
                    templates={
                        key: val
                        for key, val in templates[dv_colname].items()
                        if not key.endswith("shot")
                    },
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

    def get_shots(
        self,
        dv_colname,
        dv_dic,
        template_name,
        n,
        sep,
    ):
        # Load the pickle in data_dir called ds.pkl"
        survey_name = self.survey_name
        data_dir = f"{k.DATA_PATH}/{survey_name}/{dv_colname}"
        shotsfname = os.path.join(data_dir, "shots.pkl")
        shotsdf = pd.read_pickle(shotsfname)
        shotsdf = shotsdf[shotsdf.template_name == template_name]
        # Add the prompt and ground truth columns
        shotsdf["shots"] = shotsdf.prompt + " " + shotsdf.ground_truth.map(dv_dic)
        return sep.join(shotsdf.shots.sample(n=n, random_state=0).tolist() + [""])

    def sample_templates(self, df, dvs=None, force_overwrite=False, playground=False):
        if dvs is None:
            dvs = self.dv_colnames
        elif not isinstance(dvs, list):
            dvs = [dvs]

        templates = self.get_templates()

        for dv in dvs:
            filled_templates_dir = f"{k.FILLED_TEMPLATES_PATH}/{self.survey_name}/{dv}"

            if not os.path.exists(filled_templates_dir):
                os.makedirs(filled_templates_dir)
            with open(
                os.path.join(filled_templates_dir, "filled_templates.txt"),
                "w",
                encoding="utf8",
            ) as f:
                sub_df = df[self.present_dems + [dv]]
                sub_df = sub_df.dropna()
                type_prompt_tokens_list = []
                for template_name, (template, tokens) in templates[dv].items():
                    f.write("\n\n==============================\n\n")
                    row = sub_df.sample().iloc[0]
                    f.write(template_name)
                    f.write("\n\n")
                    prompt = template(row)
                    type_prompt_tokens_list.append((template_name, prompt, tokens))
                    f.write(prompt)
                if playground:
                    f.write("\n\n==============================\n\n")
                    template_condition_dic = self.playground(
                        type_prompt_tokens_list, dv, force_overwrite=force_overwrite
                    )
                    bad_templates = [
                        (template, result)
                        for (template, result) in template_condition_dic.items()
                        if result["passed_playgrounding"] == False
                    ]
                    f.write(
                        f"\n\nBAD TEMPLATES (N={len(bad_templates)}): These failed playgrounding\n\n==============================\n\n"
                    )
                    for template, results in bad_templates:
                        f.write(f"Template={template}:\n\n")
                        # Print 'prompt', 'top_k_token_logprob_pairs', 'has_enough_prob_mass', 'enough_survey_responses_in_top_k', 'enough_top_k_tokens_in_expected_responses', 'passed_playgrounding', 'top_token_survey_response'
                        f.write(f"Prompt = {results['prompt']}\n")
                        split_top_k_token_logprob_pairs = [
                            f"{token}, {prob}"
                            for token, prob in results["top_k_token_logprob_pairs"]
                        ]
                        split_top_k_token_logprob_pairs_str = "\n".join(
                            split_top_k_token_logprob_pairs
                        )
                        f.write(
                            f"Top K Token Logprob Pairs = {split_top_k_token_logprob_pairs_str}\n"
                        )
                        f.write(
                            f"Has Enough Prob Mass = {results['has_enough_prob_mass']}\n"
                        )
                        f.write(
                            f"Enough Survey Responses in Top K = {results['enough_survey_responses_in_top_k']}\n"
                        )
                        f.write(
                            f"Enough Top K Tokens in Expected Responses = {results['enough_top_k_tokens_in_expected_responses']}\n"
                        )
                        f.write(
                            f"Top Token Survey Response = {results['top_token_survey_response']}\n"
                        )
                        f.write("\n\n==============================\n\n")

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

    def _any_expected_response_top_k(
        self, expected_responses, token_logprob_pairs, top_k=7
    ):
        """Returns whether or not any of the expected responses are "equal" to
        any of the top k logprob tokens."""

        top_k_tokens = [token for token, _ in token_logprob_pairs[:top_k]]
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

    def _has_enough_tokens(self, all_expected_responses, tokens, min_tokens=5, top_k=7):
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
            for token in tokens[:top_k]
        ]
        return sum(are_tokens_top_k) >= min_tokens

    def playground(
        self,
        type_prompt_tokens_list,
        dv,
        force_overwrite=False,
        api_token=None,
        thresh=0.01,
        min_survey_responses=2,
        min_tokens=5,
        top_k=7,
        debug=False,
    ):
        """
        This playgrounds the input tuple of type-prompt-tokens and returns which did not pass the playgrounding steps.
        TODO: Add a way to cache finished prompts and only re-run the ones you need.


        Args:
            type_prompt_tokens_list[list]: List of tuples (prompt[str], token_dict[dict])
            api_token[str]: The AI21 Lab API token
            force_overwrite[bool]: Whether or not to overwrite the cached prompts
        """

        if api_token is None:
            api_token = os.getenv("AI21_API_KEY")

        condition_dic = {}
        filled_templates_dir = f"{k.FILLED_TEMPLATES_PATH}/{self.survey_name}/{dv}"
        playgrounded_templates_path = (
            f"{filled_templates_dir}/playgrounded_templates.pkl"
        )
        try:
            if os.path.exists(playgrounded_templates_path) and not force_overwrite:
                with open(playgrounded_templates_path, "rb") as f:
                    condition_dic = pickle.load(f)
        except Exception as e:
            pass

        with open(f"{filled_templates_dir}/playgrounded_templates.pkl", "wb") as f:
            for template_name, prompt, token_dict in type_prompt_tokens_list:
                if template_name in condition_dic and force_overwrite is False:
                    print(
                        "Skipping {template_name} because it is already playgrounded."
                    )
                    continue

                if debug:
                    print("Debugging, so reusing one response from cache.")
                    with open(
                        f"{filled_templates_dir}/place_holder_response.pkl", "rb"
                    ) as g:
                        response = pickle.load(g)
                else:
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
                    if response.status_code != 200:
                        print(f"Error: {response.status_code}: {response.reason}")
                        return

                token_logprob_pairs = response.json()["completions"][0]["data"][
                    "tokens"
                ][0]["topTokens"]
                token_logprob_pairs = [
                    (self._substitute(d["token"]), np.exp(d["logprob"]))
                    for d in token_logprob_pairs
                ]

                results = []

                survey_response_dict = {}
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
                    is_expected_response_top_k = self._any_expected_response_top_k(
                        expected_responses, token_logprob_pairs, top_k=top_k
                    )

                    # (2) any of the expected responses are in the top k tokens
                    is_expected_response_top_1 = self._any_expected_response_top_k(
                        expected_responses, token_logprob_pairs, top_k=1
                    )

                    # and save the result for every survey response
                    results.append(
                        (
                            survey_response,
                            is_enough_prob_mass,
                            is_expected_response_top_k,
                            is_expected_response_top_1,
                        )
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

                # Any survey response is top token
                top_token_survey_response = any(map(lambda result: result[3], results))

                # At least min_tokens of top 7 tokens are in response set
                enough_top_k_tokens_in_expected_responses = self._has_enough_tokens(
                    all_expected_responses,
                    [token for token, _ in token_logprob_pairs],
                    min_tokens=min_tokens,
                    top_k=top_k,
                )

                # This is where we decide if a prompt is good or bad
                passed_playgrounding = all(
                    [
                        has_enough_prob_mass,
                        enough_survey_responses_in_top_k,
                        enough_top_k_tokens_in_expected_responses,
                    ]
                )
                condition_dic[template_name] = {
                    "prompt": prompt,
                    "top_k_token_logprob_pairs": token_logprob_pairs[:top_k],
                    "has_enough_prob_mass": has_enough_prob_mass,
                    # "prob_mass": ,
                    "enough_survey_responses_in_top_k": enough_survey_responses_in_top_k,
                    "top_token_survey_response": top_token_survey_response,
                    # "n_survey_responses_in_top_k":  ,
                    "enough_top_k_tokens_in_expected_responses": enough_top_k_tokens_in_expected_responses,
                    # "n_top_k_tokens_in_expected_responses":  ,
                    "passed_playgrounding": passed_playgrounding,
                }
            pickle.dump(condition_dic, f)
        return condition_dic

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
