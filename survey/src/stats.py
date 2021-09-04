from scipy.stats import chi2_contingency

import numpy as np


def list_to_val_map(x):
    val_map = dict()
    n = 0
    for val in x:
        if val not in val_map.keys():
            val_map[val] = n
            n += 1
    return val_map


def get_pairwise_counts(a, b):
    a_map = list_to_val_map(a)
    b_map = list_to_val_map(b)
    tbl = np.zeros((len(a_map), len(b_map)))
    pairs = zip(a, b)
    for pair in pairs:
        tbl[a_map[pair[0]], b_map[pair[1]]] += 1
    return tbl


def cramers_v_from_vecs(a, b):
    """cramers_v function but including making confusion matrix"""
    return cramers_v(get_pairwise_counts(a, b))


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
