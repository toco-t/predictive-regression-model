"""
Imputing missing values in the dataset.

This module contains functions for imputing missing values in the dataset.

This module can be imported as follows:
    from Utilities.imputing import impute_missing_values

@author: Toco Tachibana
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def impute_missing_values(
        *, data_frame: pd.DataFrame, sequential: list[str]
) -> pd.DataFrame:
    """
    Impute missing values in the dataset.

    Use KNNImputer for sequential values, and mode for ordinal data.

    :param data_frame: pandas DataFrame containing the dataset
    :param sequential: list of sequential variables
    :return: DataFrame with imputed values
    """
    # common to use -999 represent missing values in the dataset
    data_frame = data_frame.replace(to_replace=-999, value=np.nan)

    ordinal_data = data_frame.drop(columns=sequential)
    ordinal_data = ordinal_data.fillna(ordinal_data.mode().iloc[0])

    numeric_data = data_frame[sequential]
    imputer = KNNImputer(n_neighbors=5)
    numeric_data = pd.DataFrame(
        imputer.fit_transform(numeric_data),
        columns=numeric_data.columns
    )

    imputed_data_frame = pd.concat([ordinal_data, numeric_data], axis=1)
    data_frame.update(imputed_data_frame)
    return data_frame
