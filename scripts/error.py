"""
Error orders for pandas dataframes.
"""

import pandas as pd
import numpy as np


def get_numerical_error_order(df: pd.DataFrame, columns: pd.Index) -> None:
    """
    Get the numerical error order of a dataframe.
    Append to the original dataframe a new column with the numerical error order.
    """
    numerical_order = [0]
    for i in range(len(df) - 1):
        numerical_order.append(
            -np.log(df[columns[1]][i + 1] / df[columns[1]][i])
            / np.log(df[columns[0]][i + 1] / df[columns[0]][i])
        )
    df[f"{columns[1]}_order"] = numerical_order


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        }
    )
    get_numerical_error_order(df, df.columns)
    print(df)
