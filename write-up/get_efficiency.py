"""
Get efficiency = error/etas
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Read the data
    error_estimators = pd.read_csv("error_estimator.csv")
    error_eigenvalues = pd.read_csv("error_eigenvalues.csv")

    # Get keys
    print(f"error_estimators keys: {error_estimators.keys()}")
    print(f"error_eigenvalues keys: {error_eigenvalues.keys()}")

    # Compute efficiency
    efficiency = error_eigenvalues.copy()
    for i in range(7):
        efficiency[f"Efficiency Eigenvalue {i}"] = error_eigenvalues[f"Error Eigenvalue {i}"] / error_estimators["Eta Eigenvalue " + str(i)]
    efficiency.to_csv("efficiency.csv", index=False)

    # Plot efficiency
    fig, ax = plt.subplots()
    for i in range(7):
        ax.semilogy(error_estimators["Ndofs"][1:], efficiency[f"Efficiency Eigenvalue {i}"], label=f"Efficiency Eigenvalue {i}")
        # ax.plot(efficiency["Time"], efficiency[f"Efficiency Eigenvalue {i}"], label=f"Efficiency Eigenvalue {i}")
    ax.legend()
    plt.show()
