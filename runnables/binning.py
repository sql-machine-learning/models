import argparse
import mars.dataframe as md
import os
import pandas as pd
from binning.binning import calc_stats, calc_two_dim_binning_stats, get_cols_bin_boundaries
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dbname", type=str, required=True)
    parser.add_argument("--columns", type=str, required=True)
    parser.add_argument("--bin_methods", type=str, required=False)
    parser.add_argument("--bin_nums", type=str, required=False)
    parser.add_argument("--bin_input_table", type=str, required=False)
    parser.add_argument("--reverse_cumsum", type=bool, default=False)
    parser.add_argument("--two_dim_bin_cols", type=str, required=False)

    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    columns = args.columns.split(',')
    bin_methods = args.bin_methods.split(',') if args.bin_methods else None
    bin_nums = [int(item) for item in args.bin_nums.split(',')] if args.bin_nums else None
    two_dim_bin_cols = args.two_dim_bin_cols.split(',') if args.two_dim_bin_cols else None

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    # Check arguments
    if two_dim_bin_cols:
        assert(len(two_dim_bin_cols) == 2)
        assert(len(output_tables) == 3)

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

        print("Ignore the bin_nums and bin_methods arguments")
        bin_nums = [None for i in range(len(columns))]
        bin_methods = [None for i in range(len(columns))]
    
    print("Calculate the statistics result for columns: {}".format(columns))
    stats_df = calc_stats(
        input_md,
        columns,
        bin_methods,
        bin_nums,
        cols_bin_boundaries,
        args.reverse_cumsum)

    print("Persist the statistics result into the table {}".format(output_tables[0]))
    stats_df.to_sql(
        name=output_tables[0],
        con=engine,
        index=False
    )

    if args.two_dim_bin_cols:
        print("Calculate two dimension binning result for columns: {}".format(columns))
        bin_prob_df, bin_cumsum_prob_df = calc_two_dim_binning_stats(
            input_md,
            columns[0],
            columns[1],
            bin_methods[0],
            bin_methods[1],
            bin_nums[0],
            bin_nums[1],
            cols_bin_boundaries.get(columns[0], None),
            cols_bin_boundaries.get(columns[1], None),
            args.reverse_cumsum)

        print("Persist the binning probabilities into table {}".format(output_tables[1]))
        bin_prob_df.to_sql(
            name=output_tables[1],
            con=engine,
            index=False
        )
        print("Persist the binning accumulated probabilities into table {}".format(output_tables[2]))
        bin_cumsum_prob_df.to_sql(
            name=output_tables[2],
            con=engine,
            index=False
        )
