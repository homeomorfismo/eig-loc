"""
Wrappers for adaptivity routines from the notebook
"""

import numpy as np
from ngsolve import (
    CoefficientFunction,
    GridFunction,
    Mesh,
)
from main_utils import (
    append_to_dict,
    error_estimator_grad_landscape,
    error_estimator_landscape,
    get_forms,
    make_unit_square,
    mark,
    solve,
    solve_eigenvalue,
    solve_eigenvalue_arnoldi,
    to_file,
)

ORDER = 3
MAXITER = 30
THETA = 0.9

CENTER = 5 * np.pi**2 + 325
RADIUS = 1.0
NPTS = 4
NSPAN = 2

MAXH = 0.1


def test_ard():
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=MAXH))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = CENTER
    iteration = 0
    while iteration < MAXITER:
        iteration += 1
        a, m, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=ORDER,
            is_complex=True,
        )
        u = solve(a, f, fes)
        _, _, evals = solve_eigenvalue(a, m, fes, center=CENTER)
        eta, _, _ = error_estimator_landscape(
            u, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
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
    to_file(eta_dict, "etas_original_problem.csv")
    to_file(eig_dict, "eval_original_problem.csv")
    to_file(err_dict, "errs_original_problem.csv")


def test_ard_mod():
    matrix = CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
    vector = CoefficientFunction((0.0, 0.0))
    scalar = CoefficientFunction(325.0)
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=MAXH))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = CENTER
    iteration = 0
    while iteration < MAXITER:
        iteration += 1
        a, m, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=ORDER,
            is_complex=True,
        )
        u = solve(a, f, fes)
        _, _, evals = solve_eigenvalue(a, m, fes, center=CENTER)
        eta, _, _ = error_estimator_landscape(
            u, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
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
    to_file(eta_dict, "etas_mod_problem.csv")
    to_file(eig_dict, "eval_mod_problem.csv")
    to_file(err_dict, "errs_mod_problem.csv")


def test_ard_eig():
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=MAXH))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = CENTER
    iteration = 0
    while iteration < MAXITER:
        iteration += 1
        a, m, _, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=ORDER,
            is_complex=True,
        )
        right, evals = solve_eigenvalue_arnoldi(a, m, fes, center=CENTER)
        etas = []
        for k in range(len(right.vecs)):
            temp = GridFunction(fes, name="Eigenfunctions")
            temp.vec.data = right.vecs[k]
            eta, _, _ = error_estimator_landscape(
                temp,
                matrix_coeff=matrix,
                vector_coeff=vector,
                scalar_coeff=scalar,
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


def test_ard_grad():
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=MAXH))

    eta_dict = {}
    eig_dict = {}
    err_dict = {}
    prev_avg_eig = CENTER
    iteration = 0
    while iteration < MAXITER:
        iteration += 1
        a, m, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=ORDER,
            is_complex=True,
        )
        u = solve(a, f, fes)
        _, _, evals = solve_eigenvalue(a, m, fes, center=CENTER)
        eta, _, _ = error_estimator_grad_landscape(
            u, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
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
    to_file(eta_dict, "etas_grad_problem.csv")
    to_file(eig_dict, "eval_grad_problem.csv")
    to_file(err_dict, "errs_grad_problem.csv")


if __name__ == "__main__":
    test_ard()
    test_ard_mod()
    test_ard_eig()
    test_ard_grad()
