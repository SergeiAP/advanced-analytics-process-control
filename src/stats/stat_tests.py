# stats
from statsmodels.stats.diagnostic import het_white, het_breuschpagan

import pandas as pd
import numpy as np


def get_hetero_white_test(reiduals: pd.Series,
                          x_variables: np.ndarray) -> dict[str, float]:
    """Get results of heteroscedasticity test (White test)

    Args:
        reiduals (pd.Series): residuals (real - target) from the model
        x_variables (np.ndarray): variables

    Returns:
        dict[str, float]: test results
    """
    labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    # H0 - no heteroscedasticity 
    white_test = het_white(reiduals, x_variables)
    dict_white = {key: val for key, val in zip(labels, white_test)}
    return dict_white

def get_hetero_breusch_pagan(reiduals: pd.Series,
                             x_variables: np.ndarray) -> dict[str, float]:
    """Get results of heteroscedasticity test (Breusch-Pagan test)

    Args:
        reiduals (pd.Series): residuals (real - target) from the model
        x_variables (np.ndarray): variables

    Returns:
        dict[str, float]: test results
    """
    labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    # H0 - no heteroscedasticity 
    breusch_pagan_test = het_breuschpagan(reiduals, x_variables)
    dict_breusch_pagan = {key: val for key, val in zip(labels, breusch_pagan_test)}
    return dict_breusch_pagan
