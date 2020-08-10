import argparse
import os
import pandas as pd 
from sqlalchemy import create_engine
from ts_feature_extractor import add_features_extracted_from_ts_data


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dbname", type=str, required=True)
    parser.add_argument("--column_id", type=str, required=True)
    parser.add_argument("--column_time", type=str, required=True)
    parser.add_argument("--columns_value", type=str, required=True)
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

    engine = create_engine("{}/{}".format(datasource, args.dbname))
    input = pd.read_sql(
        sql=select_input,
        con=engine)

    print("Start extracting the features from the time series data.")
    df_with_extracted_features = add_features_extracted_from_ts_data(
        input,
        column_id=args.column_id,
        column_time=args.column_time,
        columns_value=columns_value,
        windows=windows,
        min_window=args.min_window,
        extract_setting=args.extract_setting)
    print("Complete the feature extraction.")

    df_with_extracted_features.to_sql(
        name=output,
        con=engine,
        index=False)
    print("Complete save the result data into table {}.".format(output))
