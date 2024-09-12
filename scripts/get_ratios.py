"""
Small python script to calculate the convergence ration of the error estimators
and the error for the eigenvalues.
"""
import numpy as np
import pandas as pd

ERR_EIGENVALUES = "error_eigenvalues_eigenvalues.csv"
ERR_ESTIMATORS = "error_estimator_eigenvalues.csv"

OUT_EIGENVALUES = "ratio_ev_eigenvalues.csv"
OUT_ESTIMATORS = "ratio_ee_eigenvalues.csv"


def convergence_ratio(value1, value2, ndofs1, ndofs2):
    """
    Function to calculate the order of convergence of two values.
    """
    return - np.log(value1 / value2) / np.log(ndofs1 / ndofs2)


def get_convergence_ratios(error_estimators, error_eigenvalues, outputs=None):
    """
    Function to calculate the convergence ratios of the error estimators and the error
    for the eigenvalues.
    """
    assert len(outputs) == 2, "The outputs should be a list of two strings."
    # Read the data
    error_estimators = pd.read_csv(error_estimators)
    error_eigenvalues = pd.read_csv(error_eigenvalues)

    ndofs = np.array(error_eigenvalues["Ndofs"])
    error0 = np.array(error_eigenvalues["Error Eigenvalue 0"])
    error1 = np.array(error_eigenvalues["Error Eigenvalue 1"])
    error2 = np.array(error_eigenvalues["Error Eigenvalue 2"])
    error3 = np.array(error_eigenvalues["Error Eigenvalue 3"])
    error4 = np.array(error_eigenvalues["Error Eigenvalue 4"])
    error5 = np.array(error_eigenvalues["Error Eigenvalue 5"])
    error6 = np.array(error_eigenvalues["Error Eigenvalue 6"])

    ratio0 = convergence_ratio(error0[1:], error0[:-1], ndofs[1:], ndofs[:-1])
    ratio1 = convergence_ratio(error1[1:], error1[:-1], ndofs[1:], ndofs[:-1])
    ratio2 = convergence_ratio(error2[1:], error2[:-1], ndofs[1:], ndofs[:-1])
    ratio3 = convergence_ratio(error3[1:], error3[:-1], ndofs[1:], ndofs[:-1])
    ratio4 = convergence_ratio(error4[1:], error4[:-1], ndofs[1:], ndofs[:-1])
    ratio5 = convergence_ratio(error5[1:], error5[:-1], ndofs[1:], ndofs[:-1])
    ratio6 = convergence_ratio(error6[1:], error6[:-1], ndofs[1:], ndofs[:-1])

    ratio_df = pd.DataFrame({
        "Ndofs": ndofs[1:],
        "Ratio 0": ratio0,
        "Ratio 1": ratio1,
        "Ratio 2": ratio2,
        "Ratio 3": ratio3,
        "Ratio 4": ratio4,
        "Ratio 5": ratio5,
        "Ratio 6": ratio6
    })
    
    ratio_df.to_csv(outputs[0], index=False)

    ndofs = np.array(error_estimators["Ndofs"])
    eta = np.array(error_estimators["Eta"])
    eta0 = np.array(error_estimators["Eta Eigenvalue 0"])
    eta1 = np.array(error_estimators["Eta Eigenvalue 1"])
    eta2 = np.array(error_estimators["Eta Eigenvalue 2"])
    eta3 = np.array(error_estimators["Eta Eigenvalue 3"])
    eta4 = np.array(error_estimators["Eta Eigenvalue 4"])
    eta5 = np.array(error_estimators["Eta Eigenvalue 5"])
    eta6 = np.array(error_estimators["Eta Eigenvalue 6"])

    ratio = convergence_ratio(eta[1:], eta[:-1], ndofs[1:], ndofs[:-1])
    ratio0 = convergence_ratio(eta0[1:], eta0[:-1], ndofs[1:], ndofs[:-1])
    ratio1 = convergence_ratio(eta1[1:], eta1[:-1], ndofs[1:], ndofs[:-1])
    ratio2 = convergence_ratio(eta2[1:], eta2[:-1], ndofs[1:], ndofs[:-1])
    ratio3 = convergence_ratio(eta3[1:], eta3[:-1], ndofs[1:], ndofs[:-1])
    ratio4 = convergence_ratio(eta4[1:], eta4[:-1], ndofs[1:], ndofs[:-1])
    ratio5 = convergence_ratio(eta5[1:], eta5[:-1], ndofs[1:], ndofs[:-1])
    ratio6 = convergence_ratio(eta6[1:], eta6[:-1], ndofs[1:], ndofs[:-1])

    ratio_df = pd.DataFrame({
        "Ndofs": ndofs[1:],
        "Ratio": ratio,
        "Ratio 0": ratio0,
        "Ratio 1": ratio1,
        "Ratio 2": ratio2,
        "Ratio 3": ratio3,
        "Ratio 4": ratio4,
        "Ratio 5": ratio5,
        "Ratio 6": ratio6
    })

    ratio_df.to_csv(outputs[1], index=False)


if __name__ == "__main__":
    get_convergence_ratios(ERR_ESTIMATORS, ERR_EIGENVALUES, [OUT_ESTIMATORS, OUT_EIGENVALUES])
