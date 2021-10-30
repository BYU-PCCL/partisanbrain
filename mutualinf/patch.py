from wic import WicDataset
from copa import CopaDataset
from imdb import ImdbDataset
from anes import AnesDataset
from squad import SquadDataset
from rocstories import RocstoriesDataset
from boolq import BoolqDataset
from common_sense_qa import CommonSenseQaDataset

import glob
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


for cls, dir in CLS_DIR.items():

    # Save dataset pickle again with updated templates
    obj = cls(n=500)

    # Remove processed files
    for f in glob.glob(f"{dir}/*_processed.pkl"):
        os.remove(f)

    # Fix experiment file key sets
    for fname in glob.glob(f"{dir}/exp_results_*"):

        # Load pickle file associated with fname
        df = pd.read_pickle(fname)

        # Iterate over rows and replace token sets
        for _, row in df.iterrows():
            template_name = row["template_name"]
            row["token_sets"] = obj._get_templates()[template_name][1]

        # Pickle df
        df.to_pickle(fname)
