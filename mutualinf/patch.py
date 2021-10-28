from wic import WicDataset
from copa import CopaDataset
from imdb import ImdbDataset
from anes import AnesDataset
from squad import SquadDataset
from rocstories import RocStoriesDataset
from boolq import BoolQDataset
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
    RocStoriesDataset: "data/rocstories",
    BoolQDataset: "data/boolq",
    CommonSenseQaDataset: "data/common_sense_qa"
}


for cls, dir in CLS_DIR.items():

    obj = cls()

    # Remove processed files
    for f in glob.glob(f"{dir}/*_processed.pkl"):
        os.remove(f)

    # Fix experiment file key sets
    for fname in glob.glob(f"{dir}/exp_results_*"):

        # Load pickle file associated with fname
        df = pd.read_pickle(fname)

        # Iterate over rows and replace token sets
        for idx, row in df.iterrows():
            template_name = row["template_name"]
            row["token_set"] = obj._get_templates()[template_name][1]

        # Save pickle file
        df.save_pickle(fname)
