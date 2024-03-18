"""
Provides a function to display a dataset.

This module can be imported as follows:
    from Utilities.dataset_description import summarise_dataset

@author: Toco Tachibana
"""
import pandas as pd


def summarise_dataset(*, data_frame: pd.DataFrame):
    """
    Display the dataset.

    :param data_frame: pandas DataFrame containing the dataset
    """

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(data_frame.count())
    print(data_frame.describe(include="all").round(2).to_markdown())
