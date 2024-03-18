"""
Feature selection methods.

This module contains functions that perform feature selection on a dataset.
The following methods are implemented:
    - Recursive Feature Elimination
    - Forward Feature Selection

This module can be imported as follows:
    from Utilities.feature_selection import recursive_feature_elimination
    from Utilities.feature_selection import forward_feature_selection

@author: Toco Tachibana
"""
import pandas as pd
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LinearRegression


def recursive_feature_elimination(*, data_frame: pd.DataFrame, target: str):
    # Select significant predictors using recursive feature elimination
    X = data_frame.drop(target, axis=1)
    y = data_frame[target]

    # Perform recursive feature elimination
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=10).fit(X, y)

    # Return the list of selected predictors
    return [predictor for predictor, selected
            in zip(X.columns, rfe.support_) if selected]


def forward_feature_selection(*, data_frame: pd.DataFrame, target: str):
    # Select significant predictors using forward feature selection
    X = data_frame.drop(target, axis=1)
    y = data_frame[target]

    # Perform forward feature selection
    ffs = f_regression(X, y)

    # Return the list of selected predictors
    return [predictor for predictor in X.columns
            if ffs[1][list(X.columns).index(predictor)] < 0.00001]
