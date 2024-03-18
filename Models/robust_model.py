"""
Linear regression model with outlier treatment.

@author: Toco Tachibana
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from driver import DATA_PATH
from Utilities.visual_exploration import visual_exploration
from Utilities.imputing import impute_missing_values
from Utilities.cross_fold_validation import train_and_test
from Utilities.dataset_description import summarise_dataset


def scale_data(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()

    scaled_data_frame = scaler.fit_transform(data_frame)
    scaled_data_frame = pd.DataFrame(
        scaled_data_frame,
        columns=data_frame.columns
    )

    return scaled_data_frame


def clip_outliers(
        *, data_frame: pd.DataFrame, outliers: list[str]
) -> pd.DataFrame:
    # Clip outliers in the dataset
    clipped_data_frame = data_frame.copy()[outliers]
    for column in outliers:
        clipped_data_frame[column] = data_frame[column].clip(
            lower=clipped_data_frame[column].quantile(0.01),
            upper=clipped_data_frame[column].quantile(0.99)
        )

    data_frame.update(clipped_data_frame)
    return data_frame


def main():
    dataset_path = DATA_PATH / "loan_v2.csv"
    data_frame = pd.read_csv(dataset_path)

    dependent_variable = "Loan Sanction Amount (USD)"
    sequential_variables = [
            "Age",
            "Income (USD)",
            "Loan Amount Request (USD)",
            "Current Loan Expenses (USD)",
            "Dependents",
            "Credit Score",
            "Loan Sanction Amount (USD)",
            "Property Price"
    ]

    # Impute missing values in the dataset
    cleansed_data_frame = impute_missing_values(
        data_frame=data_frame,
        sequential=sequential_variables
    )

    # Clip outliers in the dataset
    potential_outliers = [
        "Income (USD)",
        "Loan Amount Request (USD)",
        "Current Loan Expenses (USD)",
        "Property Price",
        "Loan Sanction Amount (USD)"
    ]
    data_frame_with_clipped_outliers = clip_outliers(
        data_frame=cleansed_data_frame,
        outliers=potential_outliers
    )

    # Obtain a list of predictors showing a positive correlation with the target
    # predictors = visual_exploration(
    #     data_frame=data_frame_with_clipped_outliers,
    #     target=dependent_variable
    # )

    # Perform linear regression on the dataset
    train_and_test(
        data_frame=data_frame_with_clipped_outliers,
        target=dependent_variable,
        predictors=[
            "Loan Amount Request (USD)",
            "Credit Score",
            "Co-Applicant"
        ]
    )


if __name__ == "__main__":
    main()
