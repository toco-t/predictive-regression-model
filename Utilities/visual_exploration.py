"""
Visual exploration of data.

This module contains functions for visual exploration of data.

This module can be imported as follows:
    from Utilities.visual_exploration import visual_exploration
    from Utilities.visual_exploration import scatter_plot

@author: Toco Tachibana
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def visual_exploration(*, data_frame: pd.DataFrame, target: str) -> list[str]:
    # Produce box plots for each number column
    # for column in data_frame.select_dtypes(include="number").columns:
    #     data_frame.boxplot(column=column)
    #     plt.show()

    correlations = data_frame.corr(numeric_only=True).round(1)

    sns.heatmap(data=correlations.round(1))
    plt.tight_layout()
    plt.savefig("correlation_heatmap_outliers.png")
    plt.show()

    return [predictor for predictor in correlations[target].index.values
            if abs(correlations.loc[target, predictor]) > 0
            and predictor != target]
