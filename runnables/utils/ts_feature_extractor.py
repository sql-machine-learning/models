import pandas as pd
from functools import reduce
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, ComprehensiveFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series


EXTRACT_SETTING_NAME_TO_CLASS_DICT = {
    "minimal": MinimalFCParameters,
    "efficient": EfficientFCParameters,
    "comprehensive": ComprehensiveFCParameters
}

ROLLED_TS_ID_COLUMN_NAME = "id"
ORIGIN_JOIN_ID_COLUMN_NAME = "join_id"
ROLLED_TS_ID_FORMAT = "id={},timeshift={}"


def _roll_ts_and_extract_features(
    input,
    column_id,
    column_time,
    columns_value,
    max_window,
    min_window,
    extract_setting):
    rolled_ts = roll_time_series(
        input,
        column_id=column_id,
        column_kind=None,
        column_sort=column_time,
        rolling_direction=1,
        max_timeshift=max_window,
        min_timeshift=min_window,
        n_jobs=0)

    rename_columns = {
        value_column: "{}_w_{}".format(value_column, max_window)
        for value_column in columns_value
        }
    rolled_ts = rolled_ts.rename(columns=rename_columns)
    rolled_ts = rolled_ts.drop(columns=[column_id])

    extract_setting_clz = EXTRACT_SETTING_NAME_TO_CLASS_DICT.get(extract_setting, MinimalFCParameters)
    extracted_features = extract_features(
        rolled_ts,
        column_id=ROLLED_TS_ID_COLUMN_NAME,
        column_sort=column_time,
        n_jobs=0,
        default_fc_parameters=extract_setting_clz())

    return extracted_features


def add_features_extracted_from_ts_data(
    input,
    column_id,
    column_time,
    columns_value,
    windows,
    min_window=0,
    extract_setting="minimal"):
    """Extract features from the time series data and append them to the
    original data.

    Build the rolled time series data with various window sizes, extract
    the features using TSFresh and then append the derived features to
    the original data.

    Args:
        input: A pandas DataFrame for the input data.
        column_id: The name of the id column to group by the time series data.
            The input data can contain the time series for various entities.
            For example, the UV for different websites.
        column_time: The name of the time column.
        columns_value: Array. The names of the columns for the time series data.
        windows: Array of window sizes. The time series data will be rolled with
            each window size.
        min_window: The extract forecast windows smaller or equal than this will
            be throwed away.
        extract_setting: minimal | efficient | comprehensive. Control which features
            will be extracted. The order of feature numbers is:
            minimal < efficient < comprehensive

    Returns:
        A pandas DataFrame containing the original input data and extracted features.
    """

    input_with_join_id = pd.DataFrame()
    input_with_join_id[ORIGIN_JOIN_ID_COLUMN_NAME] = input.apply(
        lambda row: ROLLED_TS_ID_FORMAT.format(row[column_id], row[column_time]),
        axis=1)

    input_with_join_id = pd.concat(
        [input, input_with_join_id],
        axis=1)

    input = input[[column_id, column_time] + columns_value]
    input.sort_values(by=[column_id, column_time])

    extracted_features_multi_windows = [
        _roll_ts_and_extract_features(
            input=input,
            column_id=column_id,
            column_time=column_time,
            columns_value=columns_value,
            max_window=window,
            min_window=min_window,
            extract_setting=extract_setting
        ) for window in windows
    ]

    extracted_features_multi_windows = reduce(lambda left, right: pd.merge(
        left=left,
        right=right,
        how="left",
        on=ROLLED_TS_ID_COLUMN_NAME
    ), extracted_features_multi_windows)

    original_data_with_extracted_features = pd.merge(
        input_with_join_id,
        extracted_features_multi_windows,
        how='inner',
        left_on=ORIGIN_JOIN_ID_COLUMN_NAME,
        right_on=ROLLED_TS_ID_COLUMN_NAME
    )

    original_data_with_extracted_features.sort_values(by=[column_id, column_time])
    original_data_with_extracted_features = original_data_with_extracted_features.drop(columns=[ORIGIN_JOIN_ID_COLUMN_NAME])

    return original_data_with_extracted_features
