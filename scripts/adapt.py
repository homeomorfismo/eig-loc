"""
Wrappers for adaptivity routines from the notebook
"""

import numpy as np
from ngsolve import (
    GridFunction,
    Mesh,
)
from main_utils import (
    append_to_dict,
    error_estimator_landscape,
    get_forms,
    make_unit_square,
    mark,
    solve,
    solve_eigenvalue,
    solve_eigenvalue_arnoldi,
    to_file,
)

ORDER = 1
MAXITER = 50
MAXNDOFS = 2_000_000
THETA = 0.9

CENTER = 5 * np.pi**2 + 325
RADIUS = 10.0
NPTS = 8
NSPAN = 2

MAXH = 0.1


def test_ard(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    source_coeff=None,
    **kwargs,
):
    """
    Test the adaptive routine for the original problem
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    assert source_coeff is not None, "Source coefficient is required"

    mesh = Mesh(make_unit_square().GenerateMesh(maxh=parameters["maxh"]))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = parameters["center"]
    iteration = 0
    ndofs = 0
    while iteration < parameters["maxiter"] and ndofs < parameters["maxndofs"]:
        a, m, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
            order=parameters["order"],
            is_complex=True,
        )
        ndofs = fes.ndof
        iteration += 1
        u = solve(a, f, fes)
        _, _, evals = solve_eigenvalue(a, m, fes, center=parameters["center"], **kwargs)
        eta, _, _ = error_estimator_landscape(
            u,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
        )
        append_to_dict(
            eta_dict,
            ndofs=fes.ndof,
            eta_avg=np.mean(eta),
            eta_l2=np.linalg.norm(eta),
            eta_max=np.max(eta),
        )
        append_to_dict(
            eig_dict,
            ndofs=fes.ndof,
            avg_evals=np.mean(evals),
            avg_evals_real=np.real(np.mean(evals)),
            avg_evals_imag=np.imag(np.mean(evals)),
        )
        append_to_dict(
            err_dict,
            ndofs=fes.ndof,
            error_avg=np.absolute(np.mean(evals) - parameters["center"]),
            error_cauchy=np.absolute(np.mean(evals) - prev_avg_eig),
            error_relative=np.absolute(np.mean(evals) - parameters["center"])
            / np.absolute(parameters["center"]),
            error_cauchy_relative=np.absolute(np.mean(evals) - prev_avg_eig)
            / np.absolute(prev_avg_eig),
        )
        prev_avg_eig = np.mean(evals)
        # Mark
        mark(mesh, eta, theta=THETA)
        # Refine
        mesh.Refine()
    to_file(eta_dict, "etas_original_problem.csv")
    to_file(eig_dict, "eval_original_problem.csv")
    to_file(err_dict, "errs_original_problem.csv")
    return eta_dict, eig_dict, err_dict


def test_ard_eig_feast(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    **kwargs,
):
    """
    Test the adaptive routine for the eigenvalue problem
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"

    mesh = Mesh(make_unit_square().GenerateMesh(maxh=MAXH))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = parameters["center"]
    iteration = 0
    ndofs = 0
    while iteration < parameters["maxiter"] and ndofs < parameters["maxndofs"]:
        a, m, _, fes = get_forms(
            mesh,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
            order=parameters["order"],
            is_complex=True,
        )
        iteration += 1
        ndofs = fes.ndof

        right, _, evals = solve_eigenvalue(a, m, fes, center=CENTER, **kwargs)
        etas = []
        for k in range(right.m):
            eta, _, _ = error_estimator_landscape(
                right[k],
                matrix_coeff=matrix_coeff,
                vector_coeff=vector_coeff,
                scalar_coeff=scalar_coeff,
                source_coeff=evals[k] * right[k],
            )
            etas.append(eta)
        stacked_etas = np.stack(etas)
        average_etas = np.mean(stacked_etas, axis=0)
        maximum_etas = np.maximum.reduce(stacked_etas)
        append_to_dict(
            eta_dict,
            ndofs=fes.ndof,
            eta_avg_avg=np.mean(average_etas),
            eta_avg_max=np.max(average_etas),
            eta_avg_l2=np.linalg.norm(eta),
            eta_max_avg=np.mean(maximum_etas),
            eta_max_max=np.max(maximum_etas),
            eta_max_l2=np.linalg.norm(eta),
        )
        append_to_dict(
            eig_dict,
            ndofs=fes.ndof,
            avg_evals=np.mean(evals),
            avg_evals_real=np.real(np.mean(evals)),
            avg_evals_imag=np.imag(np.mean(evals)),
        )
        append_to_dict(
            err_dict,
            ndofs=fes.ndof,
            error_avg=np.absolute(np.mean(evals) - CENTER),
            error_cauchy=np.absolute(np.mean(evals) - prev_avg_eig),
            error_relative=np.absolute(np.mean(evals) - CENTER) / np.absolute(CENTER),
            error_cauchy_relative=np.absolute(np.mean(evals) - prev_avg_eig)
            / np.absolute(prev_avg_eig),
        )
        prev_avg_eig = np.mean(evals)
        # Mark
        mark(mesh, eta, theta=THETA)
        # Refine
        mesh.Refine()
    to_file(eta_dict, "etas_eig_problem.csv")
    to_file(eig_dict, "eval_eig_problem.csv")
    to_file(err_dict, "errs_eig_problem.csv")
    return eta_dict, eig_dict, err_dict


def test_ard_eig_arnoldi(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    # **kwargs,
):
    """
    Test the adaptive routine for the eigenvalue problem
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"

    mesh = Mesh(make_unit_square().GenerateMesh(maxh=MAXH))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = parameters["center"]
    iteration = 0
    ndofs = 0
    while iteration < parameters["maxiter"] and ndofs < parameters["maxndofs"]:
        a, m, _, fes = get_forms(
            mesh,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
            order=parameters["order"],
            is_complex=True,
        )
        iteration += 1
        ndofs = fes.ndof

        right, evals = solve_eigenvalue_arnoldi(a, m, fes, center=parameters["center"])
        etas = []
        for k in range(len(right.vecs)):
            temp = GridFunction(fes, name="Eigenfunctions")
            temp.vec.data = right.vecs[k]
            eta, _, _ = error_estimator_landscape(
                temp,
                matrix_coeff=matrix_coeff,
                vector_coeff=vector_coeff,
                scalar_coeff=scalar_coeff,
                source_coeff=evals[k] * temp,
            )
            etas.append(eta)
        stacked_etas = np.stack(etas)
        average_etas = np.mean(stacked_etas, axis=0)
        maximum_etas = np.maximum.reduce(stacked_etas)
        append_to_dict(
            eta_dict,
            ndofs=fes.ndof,
            eta_avg_avg=np.mean(average_etas),
            eta_avg_max=np.max(average_etas),
            eta_avg_l2=np.linalg.norm(eta),
            eta_max_avg=np.mean(maximum_etas),
            eta_max_max=np.max(maximum_etas),
            eta_max_l2=np.linalg.norm(eta),
        )
        append_to_dict(
            eig_dict,
            ndofs=fes.ndof,
            avg_evals=np.mean(evals),
            avg_evals_real=np.real(np.mean(evals)),
            avg_evals_imag=np.imag(np.mean(evals)),
        )
        append_to_dict(
            err_dict,
            ndofs=fes.ndof,
            error_avg=np.absolute(np.mean(evals) - parameters["center"]),
            error_cauchy=np.absolute(np.mean(evals) - prev_avg_eig),
            error_relative=np.absolute(np.mean(evals) - parameters["center"])
            / np.absolute(parameters["center"]),
            error_cauchy_relative=np.absolute(np.mean(evals) - prev_avg_eig)
            / np.absolute(prev_avg_eig),
        )
        prev_avg_eig = np.mean(evals)
        # Mark
        mark(mesh, eta, theta=THETA)
        # Refine
        mesh.Refine()
    to_file(eta_dict, "etas_eig_problem.csv")
    to_file(eig_dict, "eval_eig_problem.csv")
    to_file(err_dict, "errs_eig_problem.csv")
    return eta_dict, eig_dict, err_dict
