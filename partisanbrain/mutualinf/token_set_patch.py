from wic import WicDataset
from copa import CopaDataset
from imdb import ImdbDataset
from anes import AnesDataset
from squad import SquadDataset
from rocstories import RocstoriesDataset
from boolq import BoolqDataset
from common_sense_qa import CommonSenseQaDataset

import glob
import numpy as np
import os
import pandas as pd


CLS_DIR = {
    WicDataset: "data/wic",
    CopaDataset: "data/copa",
    ImdbDataset: "data/imdb",
    AnesDataset: "data/anes",
    SquadDataset: "data/squad",
    RocstoriesDataset: "data/rocstories",
    BoolqDataset: "data/boolq",
    CommonSenseQaDataset: "data/common_sense_qa"
}


def get_new_token_set(row, obj):

    try:
        ts = obj._get_templates()[row.template_name][1]
    except KeyError:
        # This template_name no longer exists
        print("Template name", row.template_name, "does not exist")
        return "DROP_ROW"

    if callable(ts):
        return ts(row)
    else:
        return ts


for cls, dir in CLS_DIR.items():

    # Save dataset pickle again with updated templates
    obj = cls(n=500)

    # Remove processed files
    for f in glob.glob(f"{dir}/*_processed.pkl"):
        os.remove(f)

    # Fix experiment file token sets
    for fname in glob.glob(f"{dir}/exp_results_*"):

        print("Updating token sets for", fname)

        # Load pickle file associated with fname
        df = pd.read_pickle(fname)

        # Update token sets
        df["token_sets"] = df.apply(lambda x: get_new_token_set(x, obj),
                                    axis=1)

        # Drop all rows from df where token_sets has a value of "DROP_ROW"
        df = df[df.token_sets != "DROP_ROW"]

        # Assert we have 10,000 rows in df (500 samples * 20 templates)
        assert len(df) == 10000, f"Got {len(df)} rows instead of 10000"

        # Pickle df
        df.to_pickle(fname)
