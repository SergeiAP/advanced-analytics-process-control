import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils._bunch import Bunch


def get_permutation_importance(df: pd.DataFrame,
                               target_col: str,
                               train_test_cfg: dict,
                               random_forest_cfg: dict,
                               n_repeats: int = 10,
                               random_state: int = 42,
                               reverse_transform: str = "no") -> Bunch:
    """
    Inspect feature importance by permuting each feature 
    More info here -> 
    https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html

    Args:
        df (pd.DataFrame): dataset with selected features
        target_col (str): target column
        train_test_cfg (dict): config for train_test_split
        random_forest_cfg (dict): config for training model for permutation
        n_repeats (int, optional): permutation repeats. Defaults to 10.
        random_state (int, optional): random state to fix seed. Defaults to 42.
        reverse_transform (str, optional): to reverse transform for metrics

    Returns:
        Bunch: results of permutation_importance in dict-alike format
    """
    feature_cols = df.columns.drop(target_col)
    x_train, x_test, y_train, y_test = train_test_split(df[feature_cols],
                                                        df[target_col],
                                                        random_state=random_state,
                                                        **train_test_cfg)
    rf = ExtraTreesRegressor(random_state=random_state,
                             **random_forest_cfg)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    
    match reverse_transform:
        case "log10":
            pred = 10 ** pred
            y_test_metric = 10 ** y_test
        case "log":
            pred = np.exp ** pred
            y_test_metric = np.exp ** y_test
        case _:
            pred = pred
            y_test_metric = y_test
    print(f"Permeation: random forest MAE is "
          f"{mean_absolute_error(pred, y_test_metric):.3f}")
    print(f"Permeation: random forest RMSE is "
          f"{mean_squared_error(pred, y_test_metric, squared=False):.3f}")
    
    result = permutation_importance(
        rf, x_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return result


def plot_vert_boxplot(permutation_res: Bunch,
                      columns: np.ndarray,
                      whiskers_len: float = 10,
                      figsize: tuple = (20, 10),
                      ) -> mpl_axes.Axes:
    """
    Plot vertical boxplot in descending format first of all for permutation_importance

    Args:
        permutation_res (Bunch): results of permutation_importance in 
        dict-alike format
        columns (np.ndarray): column names to set them for the plot
        whiskers_len (float, optional): len to detect outliers in the formula 
        Q1 - / Q3 + whis*(Q3-Q1). Defaults to 10 - no visible outliers.
        figsize (tuple, optional): size of the plot. Defaults to (20, 10).

    Returns:
        mpl_figure.Figure: matplotlib figure to save
    """
    plt.figure(figsize=figsize)
    sorted_importances_idx = permutation_res.importances_mean.argsort()
    importances = pd.DataFrame(
        permutation_res.importances[sorted_importances_idx].T,
        columns=columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=whiskers_len)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    return ax


def plot_power_prediciton(df: pd.DataFrame,
                          figsize: tuple = (20, 10),
                          fontsize: int = 14) -> mpl_axes.Axes:
    """
    Calculate and plot power prediction based on ppscore lib from 
    https://github.com/8080labs/ppscore

    Args:
        df (pd.DataFrame): dataset with selected features
        figsize (tuple, optional): size of the plot. Defaults to (20, 10).
        fontsize (int, optional): fontsize for corr value. Defaults to 14.

    Returns:
        mpl_axes.Axes: matplotlib axis
    """
    plt.figure(figsize=figsize)
    pps_matrix = pps.matrix(df).loc[:,['x', 'y', 'ppscore']]  # type: ignore
    matrix_df = pps_matrix.pivot(columns='x', index='y', values='ppscore')
    heatmap = sns.heatmap(matrix_df,
                          vmin=0, vmax=1,
                          annot=True,
                          fmt=".2f",
                          annot_kws={"fontsize": fontsize})
    heatmap.set_title('Power prediciton Heatmap', fontdict={'fontsize':16}, pad=12)
    return heatmap
