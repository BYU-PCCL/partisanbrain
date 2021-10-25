from collections import defaultdict

import abc
import glob
import os.path
import pandas as pd
import re
import tqdm


class Opener:
    """Utility class for loading a file into a pandas DataFrame"""

    def __init__(self):
        self._opening_funcs = {
            "csv": self._load_csv,
            "pkl": self._load_pickled_df,
            "pickle": self._load_pickled_df,
            "sav": pd.read_spss,
            "dta": pd.read_stata
        }

    def _load_csv(self, fname):
        return pd.read_csv(fname, encoding="unicode_escape")

    def _load_pickled_df(self, fname):
        return pd.read_pickle(fname)

    def _get_file_type(self, fname):
        return fname.split(".")[-1]

    def open(self, fname, opening_func=None):
        """
        opening_func is for a custom function to use to turn the file
        specified by fname into a pandas DataFrame. If not included,
        opening_func will be chosen from a list of reasonable
        defaults if there is one matching fname's file type.
        File types with defaults are the keys in self._opening_funcs
        above.
        """

        if opening_func is None:
            ftype = self._get_file_type(fname)

            # Get opening function if available
            try:
                opening_func = self._opening_funcs[ftype]
            except KeyError:
                msg = f"Opener class has no function for opening .{ftype} file"
                raise NotImplementedError(msg)

        # Open file if it exists
        if os.path.isfile(fname):
            return opening_func(fname)
        else:
            msg = f"Opener could not open {fname} because it doesn't exist"
            raise FileNotFoundError(msg)


class Dataset(abc.ABC):

    def __init__(self,
                 in_fname=None,
                 opening_func=None,
                 out_fname=None,
                 sample_seed=0,
                 n=None):

        self._snake_case_cls_name = self._get_snake_case_cls_name()
        recommended_ds_dir = f"data/{self._snake_case_cls_name}"

        if in_fname is None:
            in_fname = self._get_raw_data_fname(recommended_ds_dir)

        # Turn the dataset specified by in_fname
        # into a pandas dataframe
        self._df = Opener().open(fname=in_fname,
                                 opening_func=opening_func)

        # Modify the dataframe based on child class
        # specifications
        self._df = self._modify_raw_data(self._df)

        # Ensure ground_truth is now a column name
        if "ground_truth" not in list(self._df):
            msg = ("The dataframe returned from "
                   f"{self.__class__.__name__}._modify_raw_data is "
                   "missing column name \"ground_truth.\"")
            raise RuntimeError(msg)

        # Sample down to n rows if requested
        if n is not None:
            self._df = self._df.sample(n, random_state=sample_seed)

        # Make results dataframe based on self._raw_df
        # and templates
        self._templates = self._get_templates()
        result_df = self._make_result_df()

        # Save results dataframe
        save_fname = f"{recommended_ds_dir}/ds.pkl"
        result_df.to_pickle(out_fname or save_fname)

    @abc.abstractmethod
    def _modify_raw_data(self, df):
        pass

    @abc.abstractmethod
    def _get_templates(self):
        pass

    def _get_snake_case_cls_name(self):
        cls_name = self.__class__.__name__
        snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()
        snake_name = snake_name[:-len("_dataset")]
        return snake_name

    def _get_raw_data_fname(self, ds_dir):
        dir_fnames = [f.split("/")[-1] for f in
                      glob.glob(f"{ds_dir}/*")]
        for fname in dir_fnames:
            if fname.startswith("raw"):
                return f"{ds_dir}/{fname}"
        msg = f"No file starting with {ds_dir}/raw could be found"
        raise FileNotFoundError(msg)

    def _make_result_df(self):
        result_dict = defaultdict(list)
        for i, row in tqdm.tqdm(self._df.iterrows(), total=self._df.shape[0]):
            for template_name, template_info in self._templates.items():
                template_fn, token_sets = template_info

                # Allow for lambda token set
                if callable(token_sets):
                    token_sets = token_sets(row)

                result_dict["raw_idx"].append(i)
                result_dict["template_name"].append(template_name)
                result_dict["prompt"].append(template_fn(row))
                result_dict["ground_truth"].append(row["ground_truth"])
                result_dict["token_sets"].append(token_sets)
                result_dict["dataset"].append(self._snake_case_cls_name)

                # Add all other columns in self._df
                # for record keeping purposes
                other_cols = list(self._df)
                other_cols.remove("ground_truth")
                for colname in other_cols:
                    result_dict[colname].append(row[colname])

        return pd.DataFrame(result_dict)
