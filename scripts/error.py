"""
Error orders for pandas dataframes.
"""

import pandas as pd
import numpy as np


def get_numerical_error_order(df: pd.DataFrame, columns: pd.Index) -> np.ndarray:
    """
    Get the numerical error order of a dataframe.
    """
    assert len(columns) == 2, "The columns must be two"
    numerical_order = [0]
    for i in range(len(df) - 1):
        numerical_order.append(
            -np.log(np.abs(df[columns[0]][i] / df[columns[1]][i + 1]))
            / np.log(np.abs(df[columns[1]][i] / df[columns[1]][i + 1]))
        )
    return np.array(numerical_order)
