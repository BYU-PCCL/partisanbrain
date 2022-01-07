import sys
import os

# Get name of directory where this file is
current = os.path.dirname(os.path.realpath(__file__))

# Get parent directory name from the current directory
parent = os.path.dirname(current)

# Add the mutualinf to sys.path
sys.path.append(parent + "/mutualinf")

# Import mutual information modules
from dataset import Dataset
from experiment import Experiment
from postprocessor import Postprocessor
