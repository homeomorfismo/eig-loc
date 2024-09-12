"""
Wrapper for the graph generation functions.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data as pandas dataframes
etas = pd.read_csv("etas_dual_problem.csv")
evals = pd.read_csv("eval_dual_problem.csv")
errs = pd.read_csv("errs_dual_problem.csv")

# Plotting the graphs
etas.drop("Unnamed: 0", axis=1).plot(
    x="ndofs",
    title="Eta values",
    loglog=True,
    grid=True,
)
plt.show()

evals.drop("Unnamed: 0", axis=1).plot(
    x="ndofs",
    title="Eigenvalues",
    loglog=True,
    grid=True,
)
plt.show()

errs.drop("Unnamed: 0", axis=1).plot(
    x="ndofs",
    title="Errors",
    loglog=True,
    grid=True,
)
plt.show()
