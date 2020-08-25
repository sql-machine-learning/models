import argparse
import mars.dataframe as md
import os
from binning.binning import calc_stats
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dbname", type=str, required=True)
    parser.add_argument("--columns", type=str, required=True)
    parser.add_argument("--bin_methods", type=str, required=False)
    parser.add_argument("--bin_nums", type=str, required=False)
    parser.add_argument("--reverse_cumsum", type=bool, default=False)

    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    columns = args.columns.split(',')
    bin_methods = args.bin_methods.split(',') if args.bin_methods else None
    bin_nums = [int(item) for item in args.bin_nums.split(',')] if args.bin_nums else None

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    url = convertDSNToRfc1738(datasource, args.dbname)
    engine = create_engine(url)
    input_md = md.read_sql(
        sql=select_input,
        con=engine)
    input_md.execute()

    stats_df = calc_stats(
        input_md,
        columns,
        bin_methods,
        bin_nums,
        {},
        args.reverse_cumsum)

    print("Persist the statistics result into the table {}".format(output))
    stats_df.to_sql(
        name=output,
        con=engine,
        index=False
    )