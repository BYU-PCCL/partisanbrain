from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

import numpy as np


def cramers_v_from_vecs(a, b):
    """cramers_v function but including making confusion matrix"""
    return cramers_v(confusion_matrix(a, b))


def cramers_v(cm):
    """
    Cramers V calculator (from confusion matrix cm).
    Checked against
    https://mathcracker.com/cramers-v-calculator?#results
    http://vassarstats.net/newcs.html
    """

    # Calculate chi-squared statistic
    chi_sq = chi2_contingency(cm, correction=False)[0]

    # Equation
    n = cm.sum()
    numerator = chi_sq / n
    r, k = cm.shape
    denominator = min(k - 1, r - 1)
    return np.sqrt(numerator / denominator)
