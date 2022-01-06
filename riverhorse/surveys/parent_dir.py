import sys
import os

# Get name of directory where this file is
current = os.path.dirname(os.path.realpath(__file__))

# Get parent directory name from the current directory
parent = os.path.dirname(current)

# Add the parent directory to sys.path
sys.path.append(parent)

# Import parent directory modules
from survey import Survey, UserInterventionNeededError
from constants import DEMOGRAPHIC_COLNAMES, N_DVS
