import mars.dataframe as md
import mars.tensor as mt
import numpy as np
import pandas as pd


class BinningMethod(object):
    BUCKET = "bucket"
    QUANTILE = "quantile"
    LOG_BUCKET = "log_bucket"


def binning(
    in_md,
    col_name,
    bin_method,
    bins,
    boundaries):
    if boundaries:
        bin_o, bins = md.cut(in_md[col_name], bins=boundaries, labels=False, retbins=True)
        bins_np = bins.to_numpy()
    else:
        if bin_method.lower() == BinningMethod.BUCKET.lower():
            bin_o, bins = md.cut(in_md[col_name], bins=bins, labels=False, retbins=True)
            bins_np = bins.to_numpy()
        elif bin_method.lower() == BinningMethod.LOG_BUCKET.lower():
            bin_o, bins = md.cut(mt.log(in_md[col_name]), bins=bins, labels=False, retbins=True)
            bins_np = np.exp(bins.to_numpy())
        else:
            raise ValueError("Unsupport binning method: {}".format(bin_method))

    return bin_o, bins_np


def cumsum(arr, reverse):
    if type(arr) == np.ndarray:
        sum_arr = arr
    elif type(arr) == pd.DataFrame:
        sum_arr = arr.to_numpy()
    else:
        raise ValueError("Invalid input type: {}".format(type(arr)))

    for i in range(np.ndim(arr)):
        sum_arr = np.flip(np.cumsum(np.flip(sum_arr, i), i), i) if reverse else np.cumsum(sum_arr, i)

    if type(arr) == np.ndarray:
        return sum_arr
    elif type(arr) == pd.DataFrame:
        return pd.DataFrame(sum_arr)
    else:
        raise ValueError("Invalid input type: {}".format(type(arr)))


def calc_binning_stats(
    in_md,
    sel_cols,
    bin_methods,
    bin_nums,
    cols_bin_boundaries,
    reverse_cumsum):
    cols_bin_stats = []
    for i in range(len(sel_cols)):
        sel_col = sel_cols[i]
        bin_o, bins = binning(in_md, sel_col, bin_methods[i], bin_nums[i], cols_bin_boundaries.get(sel_col, None))
        bin_num = len(bins) - 1
        bin_prob_df = bin_o.value_counts(normalize=True).to_pandas().to_frame()
        bin_prob_df = bin_prob_df.reindex(range(bin_num), fill_value=0)
        bin_cumsum_prob_df = cumsum(bin_prob_df, reverse_cumsum)

        cols_bin_stats.append(
            {
                "name": sel_col,
                "bin_boundaries": ','.join(bins.astype(str)),
                "bin_prob": ','.join(bin_prob_df[bin_prob_df.columns[0]].to_numpy().astype(str)),
                "bin_cumsum_prob": ','.join(bin_cumsum_prob_df[bin_cumsum_prob_df.columns[0]].to_numpy().astype(str))
            }
        )

    return pd.DataFrame(cols_bin_stats)


def calc_basic_stats(
    in_md,
    sel_cols):
    stats_data = [
        {
            "name": sel_col,
            "min": mt.min(in_md[sel_col]).to_numpy(),
            "max": mt.max(in_md[sel_col]).to_numpy(),
            "mean": mt.mean(in_md[sel_col]).to_numpy(),
            "median": mt.median(in_md[sel_col]).to_numpy(),
            "std": mt.std(in_md[sel_col]).to_numpy(),
        } for sel_col in sel_cols
    ]

    return pd.DataFrame(stats_data)


def calc_stats(
    in_md,
    sel_cols,
    bin_methods,
    bin_nums,
    cols_bin_boundaries,
    reverse_cumsum):
    basic_stats_df = calc_basic_stats(in_md, sel_cols)
    cols_bin_stats_df = calc_binning_stats(in_md, sel_cols, bin_methods, bin_nums, cols_bin_boundaries, reverse_cumsum)
    
    stats_df = pd.merge(basic_stats_df, cols_bin_stats_df, how='inner', on='name')

    return stats_df


def calc_two_dim_binning_stats(
    in_md,
    sel_col_1,
    sel_col_2,
    bin_method_1,
    bin_method_2,
    bin_num_1,
    bin_num_2,
    bin_boundaries_1,
    bin_boundaries_2,
    reverse_cumsum):
    bin_o1, bins_1 = binning(in_md, sel_col_1, bin_method_1, bin_num_1, bin_boundaries_1)
    bin_o2, bins_2 = binning(in_md, sel_col_2, bin_method_2, bin_num_2, bin_boundaries_2)

    bin_num_1 = len(bins_1) - 1
    bin_num_2 = len(bins_2) - 1

    bin_o = bin_o1 * bin_num_2 + bin_o2
    bin_prob_df = bin_o.value_counts(normalize=True).to_pandas().to_frame()
    bin_prob_df = bin_prob_df.reindex(range(bin_num_1 * bin_num_2), fill_value=0)
    two_dim_bin_prob_np = bin_prob_df.to_numpy().reshape((bin_num_1, bin_num_2))
    two_dim_bin_cumsum_prob_np = cumsum(two_dim_bin_prob_np, reverse_cumsum)

    return pd.DataFrame(two_dim_bin_prob_np), pd.DataFrame(two_dim_bin_cumsum_prob_np)


def get_cols_bin_boundaries(stats_df):
    col_boundaries = {}
    for _, row in stats_df.iterrows():
        col_name = row['name']
        boundaries = [float(item) for item in row['bin_boundaries'].split(',')]
        col_boundaries[col_name] = boundaries

    return col_boundaries
