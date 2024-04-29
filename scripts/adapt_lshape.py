"""
Wrapper on L-shaped domains.
"""

import numpy as np
from ngsolve import (
    CoefficientFunction,
    GridFunction,
    Mesh,
    Draw,
)
from main_utils import (
    append_to_dict,
    error_estimator_landscape,
    get_forms,
    make_unit_square,
    make_l_shape,
    mark,
    solve,
    solve_transpose,
    solve_eigenvalue,
    solve_eigenvalue_arnoldi,
    to_file,
)

ORDER = 2
MAXITER = 25
MAXNDOFS = 300_000
THETA = 0.9

CENTER = 101.60529
RADIUS = 10.0
NPTS = 8
NSPAN = 2

MAXH = 0.15


def adaptivity(
    mesh,
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    source_coeff=None,
    **kwargs,
):
    """
    Adaptivity loop.
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    print(f"\tParameters: {parameters}")

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
            source_coeff=source_coeff,
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
            source_coeff=source_coeff,
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
        mark(mesh, eta, theta=parameters["theta"])
        # Refine
        mesh.Refine()

    to_file(eta_dict, "etas_original_problem.csv")
    to_file(eig_dict, "eval_original_problem.csv")
    to_file(err_dict, "errs_original_problem.csv")

    solution = {
        "u": u,
        "mesh": mesh,
    }
    return eta_dict, eig_dict, err_dict, solution


if __name__ == "__main__":
    # Create L-shaped domain
    mesh = Mesh(make_l_shape().GenerateMesh(maxh=MAXH))
    # Draw(mesh)
    # Set coefficients
    matrix_coeff = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector_coeff = CoefficientFunction((0, 0))
    scalar_coeff = CoefficientFunction(0)
    source_coeff = CoefficientFunction(1.0)

    parameters = {
        "order": ORDER,
        "maxiter": MAXITER,
        "maxndofs": MAXNDOFS,
        "theta": THETA,
        "center": CENTER,
    }

    prev_avg_eig = parameters["center"]
    iteration = 0
    ndofs = 0
    while iteration < parameters["maxiter"] and ndofs < parameters["maxndofs"]:
        a, m, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
            source_coeff=source_coeff,
            order=parameters["order"],
            is_complex=True,
        )
        ndofs = fes.ndof
        iteration += 1
        u = solve(a, f, fes)
        right, _, evals = solve_eigenvalue(a, m, fes, center=parameters["center"])
        for i in range(right.m):
            Draw(right[i], mesh, f"eig_{i}")
        eta, _, _ = error_estimator_landscape(
            u,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
            source_coeff=source_coeff,
        )
        prev_avg_eig = np.mean(evals)
        # Mark
        mark(mesh, eta, theta=parameters["theta"])
        # Refine
        mesh.Refine()

    solution = {
        "u": u,
        "mesh": mesh,
    }
    Draw(u, mesh, "u")
