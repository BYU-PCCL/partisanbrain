from datetime import date
from lmsampler import LMSampler

import pandas as pd
import tqdm
import warnings


class Experiment:

    def __init__(self,
                 model_name,
                 ds_name=None,
                 in_fname=None,
                 out_fname=None,
                 n_probs=100):

        # Get in_fname and out_fname
        if ds_name is not None:
            # Use ds_name if it is available,
            # regardless of in_fname/out_fname
            # availability
            in_fname = f"data/{ds_name}/ds.pkl"
            date_str = date.today().strftime("%d-%m-%Y")
            # replace '/' with '-'
            out_fname = (f"data/{ds_name}/exp_results_"
                         f"{model_name.replace('/', '-')}_{date_str}.pkl")
        else:
            if (in_fname is None) or (out_fname is None):
                msg = ("Please either specify ds_name "
                       "OR (in_fname AND out_fname)")
                raise RuntimeError(msg)
            else:
                # Use the in_fname and out_fname
                # provided
                msg = ("Specifying in_fname and out_fname "
                       "instead of ds_name is not generally "
                       "recommended.")
                warnings.warn(msg)
                pass

        self._model_name = model_name
        self._out_fname = out_fname
        self._n_probs = n_probs

        # Open the data for this experiment
        self._ds_df = self._open_ds_df(in_fname)

        # Create model
        self._model = LMSampler(model_name)

        # Run the experiment and save the resulting data
        self._run()

    def _open_ds_df(self, ds_fname):
        return pd.read_pickle(ds_fname)

    def _run(self):
        resps = []
        for _, row in tqdm.tqdm(self._ds_df.iterrows(),
                                total=self._ds_df.shape[0]):
            try:
                resp = self._model.send_prompt(row["prompt"],
                                               n_probs=self._n_probs)
                resps.append(resp)
            except Exception as e:
                print(e)
                resps.append(None)

        # Make new df and save it
        self._ds_df["resp"] = resps
        self._ds_df["model"] = [self._model_name] * len(resps)
        self._ds_df.to_pickle(self._out_fname)
