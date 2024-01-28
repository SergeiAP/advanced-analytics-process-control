import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def plot_and_get_pca(df: pd.DataFrame,
                     seed: int,
                     is_plot: bool = False,
                     explained_tresh: float = 0.95) -> np.ndarray:
    """Plot explained variance by PCA, draw tresh line and return PCA components 
    regarding to tresh 

    Args:
        df (pd.DataFrame): dataframe with features
        seed (int): random seed to fix state
        is_plot (bool, optional): whether plot or not PCA. Defaults to False.
        explained_tresh (float, optional): Treshold for PCA components. 
        Defaults to 0.95.

    Returns:
        np.ndarray: _description_
    """
    pca = PCA(random_state=seed)
    pca.fit(df)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    components_threshold = np.argwhere(
        explained_variance > explained_tresh).reshape(-1)[0]

    pca = PCA(n_components=components_threshold)
    pca_transformed = pca.fit_transform(df)
    print("Explained variance of 2 components", 
          np.sum(pca.explained_variance_ratio_[:2]))
    if is_plot:
        plt.vlines(components_threshold,
                   explained_variance.min(),
                   explained_variance[components_threshold],
                   linestyle='dashed')
        plt.plot(explained_variance)
        plt.show()

    return pca_transformed
