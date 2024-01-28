# pylint: disable=missing-module-docstring
from pathlib import Path

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.graph_objs._figure import Figure as plotly_Figure
from pylab import rcParams  # type: ignore


def set_plot_params(figsize: tuple[int, int] = (20, 10),
                    font_small: int = 16,
                    font_medium : int = 18,
                    font_big: int = 20) -> None:
    """Set default params for better plot representation

    Args:
        figsize (tuple[int, int]): default size of plots
        font_small (int): small font size
        font_medium (int): medium font size
        font_big (int): big font size
    """

    font_small = 16
    font_medium = 18
    font_big = 20

    rcParams['font.size'] = font_small          # controls default text sizes
    rcParams['axes.titlesize'] = font_small     # fontsize of the axes title
    rcParams['axes.labelsize'] = font_medium    # fontsize of the x and y labels
    rcParams['xtick.labelsize'] = font_small    # fontsize of the tick labels
    rcParams['ytick.labelsize'] = font_small    # fontsize of the tick labels
    rcParams['legend.fontsize'] = font_small    # legend fontsize
    rcParams['figure.titlesize'] = font_big  # fontsize of the figure title
    
    rcParams['figure.figsize'] = figsize
    plt.set_loglevel('WARNING') # type: ignore
    sns.set_theme()


def save_plot(
    figure: mpl_figure.Figure | sns.axisgrid.PairGrid | mpl_axes.Axes | plotly_Figure,
    figname: str | Path) -> None:
    """Save figure

    Args:
        figure (mpl_figure.Figure | sns.axisgrid.PairGrid | mpl_axes.Axes): what to 
        save
        figname (str | Path): path to save
    """
    match figure:
        case sns.axisgrid.PairGrid():
            figure.savefig(figname)
        case mpl_figure.Figure() | mpl_axes.Axes():
            figure.get_figure().savefig(figname)
        case plotly_Figure():
            figure.write_html(figname)
    print(f"Save plot as {figname}")
