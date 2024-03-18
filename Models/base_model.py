"""
Linear regression model without outlier treatment or binned/dummy variables.

@author: Toco Tachibana
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from driver import DATA_PATH
from Utilities.visual_exploration import visual_exploration
from Utilities.imputing import impute_missing_values
from Utilities.cross_fold_validation import train_and_test
from Utilities.dataset_description import summarise_dataset


def scale_data(
        *, data_frame: pd.DataFrame, scalable: list[str]
) -> pd.DataFrame:
    numeric_data = data_frame[scalable]

    scaler = RobustScaler()

    scaled_numeric_data = scaler.fit_transform(numeric_data)
    scaled_numeric_data = pd.DataFrame(
        scaled_numeric_data,
        columns=numeric_data.columns
    )

    data_frame.update(scaled_numeric_data)
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
            "Loan Sanction Amount (USD)"
    ]

    # Impute missing values in the dataset
    cleansed_data_frame = impute_missing_values(
        data_frame=data_frame,
        sequential=sequential_variables
    )

    # Obtain a list of predictors showing a positive correlation with the target
    predictors = visual_exploration(
        data_frame=cleansed_data_frame,
        target=dependent_variable
    )

    # Perform linear regression on the dataset
    train_and_test(
        data_frame=cleansed_data_frame,
        target=dependent_variable,
        predictors=[
            "Loan Amount Request (USD)",
            "Credit Score",
            "Co-Applicant"
        ]
    )


if __name__ == "__main__":
    main()
