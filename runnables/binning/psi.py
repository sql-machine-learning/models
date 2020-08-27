import numpy as np 
import pandas as pd


def calc_psi_per_bin(
    expected_prob,
    actual_prob):
    FALLBACK_VALUE = 0.001
    expected_prob = FALLBACK_VALUE if expected_prob == 0.0 else expected_prob
    actual_prob = FALLBACK_VALUE if actual_prob == 0.0 else actual_prob

    return (expected_prob - actual_prob) * np.log(expected_prob * 1.0 / actual_prob)


def calc_psi(
    expected_bin_probs,
    actual_bin_probs):
    assert(len(expected_bin_probs) == len(actual_bin_probs))

    result = 0.0
    for i in range(len(expected_bin_probs)):
        result += calc_psi_per_bin(expected_bin_probs[i], actual_bin_probs[i])

    return result


def get_cols_bin_probs(
    stats_df,
    bin_prob_column_name):
    col_bin_probs = {}
    for _, row in stats_df.iterrows():
        col_name = row['name']
        bin_probs = [float(item) for item in row[bin_prob_column_name].split(',')]
        col_bin_probs[col_name] = bin_probs

    return col_bin_probs
