import pandas as pd
import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import seaborn as sns


# Multiple hists
def plot_grouped_hists(df: pd.DataFrame,
                       pal: list,
                       nbins: int = 20,
                       ncols: int = 3,
                       alpha: float = 0.5,
                       density: bool = True) -> mpl_figure.Figure:
    """Plot several distributions with mean, median, std"""
    # TODO:add params describtion
    df = df.copy()
    df_cols = df.columns
    df_cols_num = len(df_cols)
    col_num = 0

    rows = int(np.ceil(df_cols_num / ncols))
    fig, axises = plt.subplots(nrows=rows,
                               ncols=ncols,
                               figsize=(24, 8*rows))

    for nrow in range(axises.shape[0]):
        for ncol in range(axises.shape[1]):
            if col_num > df_cols_num - 1:
                break
            mean = df[df_cols[col_num]].mean()
            median = df[df_cols[col_num]].median()
            sigma = df[df_cols[col_num]].std()
            gauss_x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)

            axises[nrow, ncol].hist(
                df[df_cols[col_num]], 
                nbins, 
                alpha=alpha, 
                density=density,
                histtype='bar',
                label='real data')
            axises[nrow, ncol].axvline(x=mean, label=f'Mean', color=pal[0])
            axises[nrow, ncol].axvline(x=median, label=f'Median', color=pal[1])
            axises[nrow, ncol].axvline(x=mean+sigma, label=f'Sigma', color=pal[2])
            axises[nrow, ncol].axvline(x=mean-sigma, label=f'Sigma', color=pal[2])
            axises[nrow, ncol].axvline(x=mean+3*sigma, label=f'3 Sigma', color=pal[3])
            axises[nrow, ncol].axvline(x=mean-3*sigma, label=f'3 Sigma', color=pal[3])
            axises[nrow, ncol].plot(
                gauss_x, stats.norm.pdf(gauss_x, mean, sigma), label='Ideal')
            axises[nrow, ncol].set_title(f'{df_cols[col_num]}')
            col_num += 1
    fig.tight_layout()
    plt.legend(loc="upper right")
    return fig

def plot_corr_matrix(df: pd.DataFrame,
                     title: str = '',
                     ax: np.ndarray | None = None,
                     figsize: tuple = (30, 15),
                     fontsize: int = 14
                     ) -> mpl_axes.Axes: # pylint: disable=no-member
    """Plot correlation matrix

    Args:
        df (pd.DataFrame): dataset with columns to be displayed
        title (str, optional): title for f'Correlation Heatmap {title}' form.
        Defaults to ''.
        ax (np.ndarray | None, optional): matplotlib axis. Defaults to None.
        figsize (tuple, optional): size of the plot. Defaults to (20, 10).

    Returns:
        mpl_axes.Axes: matplotlib axis
    """
    plt.figure(figsize=figsize)
    # To show one half only
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    heatmap: mpl_axes.Axes = sns.heatmap(round(df.corr(), 2), # type: ignore
                                         mask=mask,
                                         annot_kws={"fontsize": fontsize},
                                         annot=True, ax=ax)
    # Give a title to the heatmap.
    # Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title(f'Correlation Heatmap {title}', fontdict={'fontsize': 16}, pad=12)
    return heatmap


def plot_boxplot(df: pd.DataFrame,
                 is_standardized: bool = True,
                 figsize=(20, 10)) -> mpl_figure.Figure:
    """Plot boxplots for presented in dataframe features

    Args:
        df (pd.DataFrame): dataframe to be represented in boxplot
        is_standardized (bool, optional): standardize vals by `(val - mean)/std`.
        Defaults to True.
        figsize (tuple, optional): Size of the picture. Defaults to (20, 10).

    Returns:
        mpl_figure.Figure: matplotlib figure to save
    """
    if is_standardized:
        df = df.select_dtypes(include=np.number) # type: ignore
        df = (df - df.mean(axis="index")) / df.std(axis="index")
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.2)
    ax.tick_params(axis='x', rotation=45)
    ax = sns.boxplot(data=df, notch=True, ax=ax)
    return fig


def plot_pairplot(df: pd.DataFrame,
                  hue_col: str,
                  kind: str = 'reg',
                  diag_kind: str = 'kde',
                  alpha: float = 0.3) -> sns.axisgrid.PairGrid:
    """Plot pairwise relationships in a dataset in triangular format 

    Args:
        df (pd.DataFrame): dataset with columns to be ploted
        hue_col (str): columns denoted classes and show them by diffrent colors
        kind (str, optional): kind of plots under the diagonal. Defaults to 'reg'.
        diag_kind (str, optional): kind of plots on the diagonal. Defaults to 'kde'.
        alpha (float, optional): transparency of elements. Defaults to 0.3.

    Returns:
        sns.axisgrid.PairGrid: seaborn pairplot to save
    """
    pairplot = sns.pairplot(df,
                            hue=hue_col, 
                            kind=kind,
                            diag_kind=diag_kind, 
                            plot_kws={'scatter_kws': {'alpha': alpha}}, 
                            corner=True)
    sns.move_legend(pairplot, "upper right", bbox_to_anchor=(0.95, 0.95))
    return pairplot

def plot_qq_hist_plot(series: pd.Series) -> mpl_figure.Figure:
    """Plot qq-plot and hist of one feature/target variable 

    Args:
        series (pd.Series): data to be plot

    Returns:
        mpl_figure.Figure: figure itself with 2 plots at one canvas
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    stats.probplot(series, dist="norm", plot=ax1)
    count, bins, ignored = ax2.hist(series, 25, density=True)
    return fig

def plot_pred_with_intervals(df: pd.DataFrame,
                             x_col: str,
                             pred_col: str,
                             prefix: str = "",
                             true_col: str | None = None,
                             is_obs: bool = True) -> None:
    """Plot prediction vs feature (x_col) with prediction and confidence interval

    Args:
        df (pd.DataFrame): data with "x_col", "pred_col", "true_col" and 
        mean/obs_ci_lower/upper cols, as in ols model "get_prediction().summary_frame()"
        x_col (str): x column name
        pred_col (str): prediciton column name
        prefix (str, optional): prefix for mean/obs_ci_lower/upper. Defaults to "".
        true_col (str | None, optional): true values column for scatter plot.
        Defaults to None.
        is_obs (bool, optional): whether to plot preediciton interval. Defaults to True.
    """
    if true_col is not None:
        plt.plot(df[x_col], df[true_col], 'o', label="Data points")
    plt.plot(df[x_col], df[pred_col], label=f"Prediction")
    plt.plot(df[x_col], df[prefix + "mean_ci_lower"], 'r--', lw=2, label="Confidence interval")
    plt.plot(df[x_col], df[prefix + "mean_ci_upper"], 'r--', lw=2)
    if is_obs:
        plt.plot(df[x_col], df[prefix + "obs_ci_lower"], 'g--', lw=2, label="Prediction interval")
        plt.plot(df[x_col], df[prefix + "obs_ci_upper"], 'g--', lw=2)
    plt.ylabel(pred_col)
    plt.title(f"{x_col} vs {pred_col} prediction")
    plt.legend()
    plt.show()

