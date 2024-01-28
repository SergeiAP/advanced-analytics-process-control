from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
from matplotlib.collections import PathCollection

import pandas as pd
import numpy as np


def create_explore_ols(df: pd.DataFrame,
                       target_col: str,
                       features_formula: str, 
                       normality_test_kind: str = "Shapiro-Wilk", #"Kolmogorov-Smirnov"
                       ) -> tuple[OLS, PathCollection, mpl_figure.Figure]:
    """Create OLS from statsmodels and explore it's predictions

    Args:
        df (pd.DataFrame): dataset with target and features
        target_col (str): target name in df
        features_formula (str): df features combined in formula as str
        normality_test_kind (str, optional): "Shapiro-Wilk" or "Kolmogorov-Smirnov"
        normality test, if other - no tets. Defaults to "Shapiro-Wilk"

    Returns:
        tuple[OLS, PathCollection, mpl_figure.Figure]: model itself, 
        residual plot, plot with residuals ditribution
    """

    model = ols(f'{target_col} ~' + features_formula, data=df).fit()
    y_pred_df_raw = model.predict(df)

    # view model summary
    print(model.summary())
    # residuals
    residuals = df[target_col] - y_pred_df_raw
    # residual plot
    residual_plot = plt.scatter(residuals , y_pred_df_raw)
    distribution_fig, (ax1, ax2) = plt.subplots(1, 2)
    # check normality
    stats.probplot(residuals , dist="norm", plot=ax1)
    count, bins, ignored = ax2.hist(residuals , 25, density=True)
    match normality_test_kind:
        case "Shapiro-Wilk":
            print("Shapiro-Wilk normality test:\n", stats.shapiro(residuals))
        case "Kolmogorov-Smirnov":
            print("Kolmogorov-Smirnov normality test:\n",
                  stats.kstest(residuals, "norm", args=(0, residuals.std())))
        case _:
            print("No normality test is chosen")
    return model, residual_plot, distribution_fig
