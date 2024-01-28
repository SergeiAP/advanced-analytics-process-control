import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(pred: pd.Series,
                      target: pd.Series,
                      is_pow: bool = False) -> None:
    """Calculate MSE and RMSE

    Args:
        pred (pd.Series): predictions
        target (pd.Series): target
        is_pow (bool, optional): whether rise to pow=10. Defaults to False.
    """
    if is_pow:
        pred = 10 ** pred
        target = 10 ** target
    print("MAE is " f"{mean_absolute_error(pred, target):.3f}",
        "RMSE is "f"{mean_squared_error(pred, target, squared=False):.3f}")
