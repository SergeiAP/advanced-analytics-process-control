import pandas as pd

# stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


def display_df_info(df: pd.DataFrame,
                    percentiles: list[float] =[.002, .01, .05, .5, .95, .99],
                     ) -> None:
    """Give basic info about dataset

    Args:
        df (pd.DataFrame): pandas dataset
        percentiles (list[float], optional): Percentiles for dataset. 
            Defaults to [.002, .01, .05, .5, .95, .99].
    """
    display(df.info(show_counts=True))  # type: ignore pylint: disable=undefined-variable
    display(df.describe(percentiles=percentiles))  # type: ignore pylint: disable=undefined-variable
    print(f"NaNs portion: {sum(df.isna().any(axis=1)) / len(df):.3f}")

def get_vif(df: pd.DataFrame,
            target_col: str,
            features_formula: str) -> pd.DataFrame:
    """Get VIF - Variance Inflation Factor and R2 for each feature

    Args:
        df (pd.DataFrame): dataset with target and features
        target_col (str): target name in df
        features_formula (str): df features combined in formula as str

    Returns:
        pd.DataFrame: vif results
    """
    y, X = dmatrices(f'{target_col} ~' + '0 +' + features_formula, df, return_type='dataframe')
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["R2"] = 1 - 1 / vif["VIF Factor"]
    vif["features"] = X.columns
    return vif.T
