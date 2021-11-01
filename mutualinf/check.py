from wic import WicDataset
from copa import CopaDataset
from imdb import ImdbDataset
from anes import AnesDataset
from squad import SquadDataset
from rocstories import RocstoriesDataset
from boolq import BoolqDataset
from common_sense_qa import CommonSenseQaDataset

import glob
import pandas as pd


CLS_DIR = {
    # WicDataset: "data/wic",
    # CopaDataset: "data/copa",
    # ImdbDataset: "data/imdb",
    # AnesDataset: "data/anes",
    # SquadDataset: "data/squad",
    # RocstoriesDataset: "data/rocstories",
    # BoolqDataset: "data/boolq",
    CommonSenseQaDataset: "data/common_sense_qa"
}


for cls, dir in CLS_DIR.items():

    # Fix experiment file key sets
    for fname in glob.glob(f"{dir}/exp_results_*"):

        print("="*50)
        print(fname)
        print("="*50)

        # Load pickle file associated with fname
        df = pd.read_pickle(fname)

        print(set([str(x) for x in df["token_sets"].tolist()]))
