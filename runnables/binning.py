import argparse
import mars.dataframe as md
import os
import pandas as pd
from bin.binning_calculator import calc_stats, calc_two_dim_binning_stats, get_cols_bin_boundaries
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dbname", type=str, required=True)
    parser.add_argument("--columns", type=str, required=True)
    parser.add_argument("--bin_method", type=str, required=False)
    parser.add_argument("--bin_num", type=str, required=False)
    parser.add_argument("--bin_input_table", type=str, required=False)
    parser.add_argument("--reverse_cumsum", type=bool, default=False)
    parser.add_argument("--two_dim_bin_cols", type=str, required=False)

    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    columns = args.columns.split(',')
    bin_method_array = args.bin_method.split(',') if args.bin_method else None
    bin_num_array = [int(item) for item in args.bin_num.split(',')] if args.bin_num else None

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    assert len(output_tables) == 1, "The output tables shouldn't be null and can contain only one."

    url = convertDSNToRfc1738(datasource, args.dbname)
    engine = create_engine(url)
    input_md = md.read_sql(
        sql=select_input,
        con=engine)
    input_md.execute()

    cols_bin_boundaries = {}
    if args.bin_input_table:
        print("Get provided bin boundaries from table {}".format(args.bin_input_table))
        bin_input_df = pd.read_sql_table(
            table_name=args.bin_input_table,
            con=engine)
        cols_bin_boundaries = get_cols_bin_boundaries(bin_input_df)

        if set(columns) > cols_bin_boundaries.keys():
            raise ValueError("The provided bin boundaries contains keys: {}. But they cannot cover all the \
                input columns: {}".format(cols_bin_boundaries.keys(), columns))

        print("Ignore the bin_num and bin_method arguments")
        bin_num_array = [None] * len(columns)
        bin_method_array = [None] * len(columns)
    else:
        if len(bin_num_array) == 1:
            bin_num_array = bin_num_array * len(columns)
        else:
            assert(len(bin_num_array) == len(columns))

        if len(bin_method_array) == 1:
            bin_method_array = bin_method_array * len(columns)
        else:
            assert(len(bin_method_array) == len(columns))
    
    print("Calculate the statistics result for columns: {}".format(columns))
    stats_df = calc_stats(
        input_md,
        columns,
        bin_method_array,
        bin_num_array,
        cols_bin_boundaries,
        args.reverse_cumsum)

    print("Persist the statistics result into the table {}".format(output_tables[0]))
    stats_df.to_sql(
        name=output_tables[0],
        con=engine,
        index=False
    )
