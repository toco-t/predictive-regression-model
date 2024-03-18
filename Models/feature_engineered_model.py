"""
Linear regression model with binned and dummy variables.

@author: Toco Tachibana
"""
import pandas as pd

from driver import DATA_PATH
from Utilities.visual_exploration import visual_exploration
from Utilities.imputing import impute_missing_values
from Utilities.cross_fold_validation import train_and_test
from Utilities.dataset_description import summarise_dataset


def bin_data(
        *, data_frame: pd.DataFrame,
        categorical: list[str],
        binned: list[list[int]]
) -> pd.DataFrame:
    # Bin data in the dataset
    binned_data_frame = data_frame.copy()[categorical]
    for column, criteria in zip(categorical, binned):
        binned_data_frame[column] = pd.cut(
            x=data_frame[column],
            bins=criteria,
            labels=False
        )

    data_frame.update(binned_data_frame)
    return data_frame


def replace_with_dummies(
        *, data_frame: pd.DataFrame, categorical: list[str]
) -> pd.DataFrame:
    # Replace categorical variables with dummies
    data_frame = pd.get_dummies(data_frame, columns=categorical, dtype=int)

    return data_frame


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
            # "Loan Sanction Amount (USD)"
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
        # "Loan Sanction Amount (USD)",
        "Property Age",
        "Property Price"
    ]
    data_frame_with_outliers_clipped = clip_outliers(
        data_frame=cleansed_data_frame,
        outliers=potential_outliers
    )

    # Bin data in the dataset
    categorical_variables = [
        "Credit Score"
    ]
    binning_criteria = [
        [550, 600, 650, 750, 900]
    ]
    binned_data_frame = bin_data(
        data_frame=data_frame_with_outliers_clipped,
        categorical=categorical_variables,
        binned=binning_criteria
    )

    categorical_variables.append("Income Stability")
    categorical_variables.append("Location")
    # Replace categorical variables with dummies
    sequential_data_frame = replace_with_dummies(
        data_frame=binned_data_frame,
        categorical=categorical_variables
    )

    # Rename binary columns to be more descriptive
    sequential_data_frame.rename(
        columns={
            "Credit Score_0.0": "Credit Score [550-600)",
            "Credit Score_1.0": "Credit Score [600-650)",
            "Credit Score_2.0": "Credit Score [650-750)",
            "Credit Score_3.0": "Credit Score [750-900)",
        },
        inplace=True
    )

    summarise_dataset(data_frame=sequential_data_frame)

    # Obtain a list of predictors showing a positive correlation with the target
    predictors = visual_exploration(
        data_frame=sequential_data_frame,
        target=dependent_variable
    )

    # Perform linear regression on the dataset
    train_and_test(
        data_frame=sequential_data_frame,
        target=dependent_variable,
        predictors=[
            "Loan Amount Request (USD)",
            "Co-Applicant",
            "Credit Score [550-600)",
            "Credit Score [600-650)",
            "Credit Score [650-750)",
            "Credit Score [750-900)",
            "Income Stability_Low"
        ]
    )


if __name__ == "__main__":
    main()

