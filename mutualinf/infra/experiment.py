from lmsampler import LMSampler

import pandas as pd


class Experiment:

    def __init__(self,
                 in_fname,
                 model_name,
                 out_fname,
                 n_probs=100):

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
        for _, row in self._ds_df.iterrow():
            try:
                resp = self._model.send_prompt(row["prompt"],
                                               n_probs=self._n_probs)
                resps.append(resp)
            except Exception as e:
                print(e)
                resps.append(None)  # TODO: Does this go in as NaN

        # Make new df and save it
        self._df["resp"] = resps
        self._df["model"] = [self._model_name] * len(resps)
        self._df.to_pickle(self._out_fname)


if __name__ == "__main__":
    e = Experiment()
