"""
Split dataset into subgroups using cross-fold validation to train and test.

This module contains functions for splitting the dataset into subgroups using
cross-fold validation to train and test the model.

This module can be imported as follows:
    from Utilities.cross_fold_validation import train_and_test

@author: Toco Tachibana
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from math import sqrt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def train_and_test(
        *, data_frame: pd.DataFrame, target: str, predictors: list[str]
):
    cross_fold = KFold(n_splits=10, shuffle=True)
    rmse_list = []
    r_squared_list = []
    r_squared_adj_list = []
    aic_list = []
    bic_list = []

    for index, (train_indexes, test_indexes) \
            in enumerate(cross_fold.split(data_frame)):
        X_train = data_frame.iloc[train_indexes, :][predictors]
        X_test = data_frame.iloc[test_indexes, :][predictors]
        y_train = data_frame.iloc[train_indexes, :][target]
        y_test = data_frame.iloc[test_indexes, :][target]

        # Scale the training and test data
        # scaler = MinMaxScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        # y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
        # y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

        # if len(predictors) > 1:
        #     # For each predictor, calculate VIF and save in dataframe
        #     print(f"\nVariance Inflation Factor (VIF):")
        #     for feature in X_train_scaled:
        #         vif = variance_inflation_factor(
        #             X_train_scaled.values,
        #             X_train_scaled.columns.get_loc(feature)
        #         )
        #         print(f"{feature} -> {vif}")

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        linear_model = sm.OLS(y_train, X_train).fit()
        prediction = linear_model.predict(X_test)

        # prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

        print(f"\nK-FOLD VALIDATION: {index + 1}")
        print(linear_model.summary())

        rmse = sqrt(metrics.mean_squared_error(y_test, prediction))
        rmse_list.append(rmse)
        print(f"RMSE: {rmse}")

        r_squared = metrics.r2_score(y_test, prediction)
        r_squared_list.append(r_squared)
        print(f"R-squared: {r_squared}")

        r_squared_adj = 1 - (1 - r_squared) * (len(y_test) - 1) / \
            (len(y_test) - len(predictors) - 1)
        r_squared_adj_list.append(r_squared_adj)
        print(f"Adjusted R-squared: {r_squared_adj}")

        aic = linear_model.aic
        aic_list.append(aic)
        print(f"AIC: {aic}")

        bic = linear_model.bic
        bic_list.append(bic)
        print(f"BIC: {bic}")

    average_rmse = sum(rmse_list) / len(rmse_list)
    print(f"\nAverage RMSE: {average_rmse}")

    average_r_squared = sum(r_squared_list) / len(r_squared_list)
    print(f"Average R-squared: {average_r_squared}")

    average_r_squared_adj = sum(r_squared_adj_list) / len(r_squared_adj_list)
    print(f"Average Adjusted R-squared: {average_r_squared_adj}")

    average_aic = sum(aic_list) / len(aic_list)
    print(f"Average AIC: {average_aic}")

    average_bic = sum(bic_list) / len(bic_list)
    print(f"Average BIC: {average_bic}")

    return average_rmse, average_r_squared, average_r_squared_adj, average_aic, average_bic
