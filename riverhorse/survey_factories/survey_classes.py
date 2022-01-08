import sys
import os

# Get name of directory where this file is
current = os.path.dirname(os.path.realpath(__file__))

# Get parent directory name from the current directory
parent = os.path.dirname(current)

# Add the surveys to sys.path
sys.path.append(parent + "/surveys")

# Import parent directory modules
from gss_survey import GssSurvey
from anes_survey import AnesSurvey
from addhealth_survey import AddhealthSurvey
from cces_survey import CcesSurvey
from pew_survey import PewSurvey
from prri_survey import PrriSurvey
from baylor_survey import BaylorSurvey
