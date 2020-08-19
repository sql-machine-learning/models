import argparse
import os
import pandas as pd 
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine
from time_series_processing.ts_feature_extractor import add_features_extracted_from_ts_data, add_lag_columns


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dbname", type=str, required=True)
    parser.add_argument("--column_id", type=str, required=True)
    parser.add_argument("--column_time", type=str, required=True)
    parser.add_argument("--columns_value", type=str, required=True)
    parser.add_argument("--lag_num", type=int, default=1)
    parser.add_argument("--windows", type=str, required=True)
    parser.add_argument("--min_window", type=str, default=0)
    parser.add_argument("--extract_setting", type=str, default="minimal", choices=["minimal", "efficient", "comprehensive"])

    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    columns_value = args.columns_value.split(',')
    windows = [int(item) for item in args.windows.split(',')]

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    url = convertDSNToRfc1738(datasource, args.dbname)
    engine = create_engine(url)
    input = pd.read_sql(
        sql=select_input,
        con=engine)

    df_with_lag_columns, lag_column_names = add_lag_columns(input, columns_value, args.lag_num)

    print("Start extracting the features from the time series data.")
    df_with_extracted_features = add_features_extracted_from_ts_data(
        df_with_lag_columns,
        column_id=args.column_id,
        column_time=args.column_time,
        columns_value=lag_column_names,
        windows=windows,
        min_window=args.min_window,
        extract_setting=args.extract_setting)
    print("Complete the feature extraction.")

    df_with_extracted_features = df_with_extracted_features.drop(columns=lag_column_names)

    df_with_extracted_features.to_sql(
        name=output,
        con=engine,
        index=False)
    print("Complete save the result data into table {}.".format(output))
