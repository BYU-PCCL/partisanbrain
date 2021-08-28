from opener import Opener

import abc
import warnings


class PromptSpecs:

    def __init__(self, question, answer_prefix, answer_map):
        self._question = question
        self._answer_prefix = answer_prefix
        self._answer_map = answer_map

    @property
    def question(self):
        return self._question

    @property
    def answer_prefix(self):
        return self._answer_prefix

    @property
    def answer_map(self):
        return self._answer_map


class Dataset(abc.ABC):
    """Base class for datasets."""

    def __init__(self,
                 fname,
                 min_samples=800,
                 opening_func=None,
                 samples=1000,
                 sampling_random_state=0):

        self._min_samples = min_samples
        self._samples = samples
        self._seed = sampling_random_state

        # Load data into pandas DataFrame
        data = Opener().open(fname, opening_func=opening_func)

        # Filter down to rows where respondents are from USA
        data = self._filter_to_usa(data)

        # Make demographics table
        demographic_col_names = self._get_demographic_col_names()
        self._demographics = data[list(demographic_col_names.keys())]
        self._demographics = self._demographics.rename(demographic_col_names,
                                                       axis=1)

        # Filter out rows from demographics that have uninformative
        # or missing values
        self._demographics = self._filter_demographics(self._demographics)
        self._demographics = self._demographics.dropna(axis=0)

        # Get row backstories (so they only need to be calculated once)
        self._row_backstories = {idx: self._make_backstory(row) for (idx, row)
                                 in self._demographics.iterrows()}

        # Get DV dataframe
        dv_col_names = self._get_dv_col_names()
        dvs = data[list(dv_col_names.keys())]
        dvs = dvs.rename(dv_col_names, axis=1)
        dvs = dvs.iloc[self.kept_indices]
        self._dvs = {cn: dvs[cn] for cn in dv_col_names.values()}

        # Make a mapping from row index to row's prompts data
        self._prompts_dict = {}
        for dv in dv_col_names.values():

            # Filter the dv series
            ok_keys = self._get_col_prompt_specs()[dv].answer_map.keys()

            # Warn if self._dvs[dv] has values not represented in the
            # keys of the dv's associated answer_map
            unique_vals = self._dvs[dv].unique()
            missing_vals = set(unique_vals) - set(ok_keys)

            self._dvs[dv] = self._dvs[dv][self._dvs[dv].isin(list(ok_keys))]
            self._dvs[dv] = self._dvs[dv].dropna()
            for val in missing_vals:
                warnings.warn((f"The dv {dv} has value \"{val}\" not "
                               "represented in its associated "
                               "answer_map"))

            # Sample
            if len(self._dvs[dv]) >= self._samples:
                self._dvs[dv] = self._dvs[dv].sample(n=self._samples,
                                                     random_state=self._seed)
            elif len(self._dvs[dv]) >= self._min_samples:
                warnings.warn((f"DV {dv} only has {len(self._dvs[dv])} "
                               "samples. Sampling anyways because {dv} has "
                               f"at least {self._min_samples} (min_samples) "
                               "values."))
            else:
                raise ValueError((f"DV {dv} only has {len(self._dvs[dv])} "
                                  "values, which is not enough "
                                  "to allow for sampling responses from "
                                  f"{self._samples} respondents"))

            self._prompts_dict[dv] = {}
            for idx in self._dvs[dv].index:
                self._prompts_dict[dv][idx] = self._make_prompt(idx, dv)

    @property
    def prompts(self):
        """{row_index: {dv_colname: prompt}}"""
        return self._prompts_dict

    @property
    def dvs(self):
        """{dv_name: dv_series}"""
        return self._dvs

    @property
    def kept_indices(self):
        return self._demographics.index.tolist()

    @property
    def row_backstories(self):
        return self._row_backstories

    @abc.abstractclassmethod
    def _filter_demographics(self, df):
        """Filter out rows with unhelpful values"""
        pass

    @abc.abstractclassmethod
    def _filter_to_usa(self, df):
        """Return a new dictionary where all respondents are from USA"""
        pass

    @abc.abstractclassmethod
    def _get_demographic_col_names(self):
        """
        Return a dictionary with column names as keys and convenience
        names as values
        """
        pass

    @abc.abstractclassmethod
    def _get_dv_col_names(self):
        """
        Return a dictionary with column names as keys and convenience
        names as values
        """
        pass

    @abc.abstractmethod
    def _make_backstory(self, row):
        """
        Here subclass should use taken row to make the demographics
        based backstory. This should not include DV information. This
        method is only abstract because this method will help make
        the _make_prompts method cleaner.
        """
        pass

    @abc.abstractmethod
    def _get_col_prompt_specs(self):
        """
        Here subclass should return a dictionary where each key
        is a column name present in the formatted self._data and
        each value is a tuple consisting of the prompt without
        value and a function for handling value
        (i.e., {column_name: (prompt_str, handler_func)}). For
        example, if we are working with a DV called partisanship
        the prompt string might be "The party I identify most
        with is" (note no space on the end). The function might be
        lambda x: x.upper() if we want values from out dataframe
        to appear capitalized when in a prompt.
        """
        pass

    def _make_prompt(self, row_idx, col_name):
        row_backstory = self._row_backstories[row_idx]
        specs = self._get_col_prompt_specs()[col_name]

        prompt = "If asked to choose either "
        answer_opts = []
        for opt in specs.answer_map.values():
            if opt not in answer_opts:
                answer_opts.append(opt)

        # There should be a space after prefix if it exists
        prefix = f"\"{specs.answer_prefix}"
        if len(specs.answer_prefix):
            prefix = "\"" + specs.answer_prefix + " "

        answers = [prefix + v + "\"" for v in answer_opts]
        prompt += " OR ".join(answers)
        prompt += " in response to the question, "
        prompt += f"\"{specs.question}\""
        prompt += f" I'd choose \"{specs.answer_prefix}"

        return f"{row_backstory} {prompt}"
