import os
import sys


cur_dir = os.getcwd()
if cur_dir.split("/")[-1] != "mutualinf":
    msg = ("Ensure you're running this file "
           "from the mutualinf directory, not a "
           "subdirectory.")
    raise RuntimeError(msg)

sys.path.append("infra")

from dataset import Dataset
