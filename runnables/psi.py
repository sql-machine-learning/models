import argparse
import os
import pandas as pd
from binning.psi import calc_psi, get_cols_bin_probs
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dbname", type=str, required=True)
    parser.add_argument("--refer_stats_table", type=str, required=True)
    parser.add_argument("--bin_prob_column", type=str, default="bin_prob")

    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    url = convertDSNToRfc1738(datasource, args.dbname)
    engine = create_engine(url)

    input_df = pd.read_sql(
        sql=select_input,
        con=engine)
    refer_stats_df = pd.read_sql_table(
        table_name=args.refer_stats_table,
        con=engine)

    actual_cols_bin_probs = get_cols_bin_probs(input_df, args.bin_prob_column)
    expected_cols_bin_probs = get_cols_bin_probs(input_df, args.bin_prob_column)

    common_column_names = set.intersection(
        set(actual_cols_bin_probs.keys()),
        set(expected_cols_bin_probs.keys()))

    print("Calculate the PSI value for {} fields.".format(len(common_column_names)))
    cols_psi_data = []
    for column_name in common_column_names:
        psi_value = calc_psi(actual_cols_bin_probs[column_name], expected_cols_bin_probs[column_name])
        cols_psi_data.append(
            {
                "name": column_name,
                "psi": psi_value
            }
        )
    cols_psi_df = pd.DataFrame(cols_psi_data)

    print("Persist the PSI result into the table {}".format(output))
    cols_psi_df.to_sql(
        name=output,
        con=engine,
        index=False
    )
