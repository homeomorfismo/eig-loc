"""
Wrappers for adaptivity routines from the notebook
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
    Solves the original problem using the right-hand side estimator.
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    print(f"\tParameters: {parameters}")

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


def test_ard_dual(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    source_coeff=None,
    **kwargs,
):
    """
    Solve the primal and dual problems and use both estimators.
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    print(f"\tParameters: {parameters}")

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
            source_coeff=source_coeff,
            order=parameters["order"],
            is_complex=True,
        )
        ndofs = fes.ndof
        iteration += 1
        u = solve(a, f, fes)
        _, _, evals = solve_eigenvalue(a, m, fes, center=parameters["center"], **kwargs)
        eta_r, _, _ = error_estimator_landscape(
            u,
            matrix_coeff=matrix_coeff,
            vector_coeff=vector_coeff,
            scalar_coeff=scalar_coeff,
            source_coeff=source_coeff,
        )
        v = solve_transpose(a, f, fes)
        eta_l, _, _ = error_estimator_landscape(
            v,
            matrix_coeff=matrix_coeff,
            vector_coeff=-1.0 * vector_coeff,
            scalar_coeff=scalar_coeff,
            source_coeff=source_coeff,
        )
        temp_stack = np.stack([eta_r, eta_l])
        # eta = np.mean(temp_stack, axis=0)
        eta = np.maximum.reduce(temp_stack, axis=0)
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

    to_file(eta_dict, "etas_dual_problem.csv")
    to_file(eig_dict, "eval_dual_problem.csv")
    to_file(err_dict, "errs_dual_problem.csv")

    solution = {
        "u": u,
        "v": v,
        "mesh": mesh,
    }
    return eta_dict, eig_dict, err_dict, solution


def test_ard_eig_feast(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    **kwargs,
):
    """
    Solve the primal problem and use the right eiegnvalue estimator.
    The implementation uses the FEAST eigensolver.
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    print(f"\tParameters: {parameters}")

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
        mark(mesh, eta, theta=parameters["theta"])
        # Refine
        mesh.Refine()

    a, m, f, fes = get_forms(
        mesh,
        matrix_coeff=matrix_coeff,
        vector_coeff=vector_coeff,
        scalar_coeff=scalar_coeff,
        order=parameters["order"],
        is_complex=True,
    )
    u = solve(a, f, fes)

    to_file(eta_dict, "etas_eig_problem.csv")
    to_file(eig_dict, "eval_eig_problem.csv")
    to_file(err_dict, "errs_eig_problem.csv")

    solution = {
        "u": u,
        "mesh": mesh,
    }
    return eta_dict, eig_dict, err_dict, solution


def test_ard_eig_feast_dual(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    **kwargs,
):
    """
    Solve the primal and dual problems and use both estimators.
    The implementation uses the FEAST eigensolver.
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    print(f"\tParameters: {parameters}")

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

        right, left, evals = solve_eigenvalue(a, m, fes, center=CENTER, **kwargs)
        etas = []
        for k in range(right.m):
            eta_r, _, _ = error_estimator_landscape(
                right[k],
                matrix_coeff=matrix_coeff,
                vector_coeff=vector_coeff,
                scalar_coeff=scalar_coeff,
                source_coeff=evals[k] * right[k],
            )
            eta_l, _, _ = error_estimator_landscape(
                left[k],
                matrix_coeff=matrix_coeff,
                vector_coeff=-1.0 * vector_coeff,
                scalar_coeff=scalar_coeff,
                source_coeff=np.conj(evals[k]) * left[k],
            )
            temp_stack = np.stack([eta_r, eta_l])
            eta = np.mean(temp_stack, axis=0)
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
        mark(mesh, eta, theta=parameters["theta"])
        # Refine
        mesh.Refine()

    a, m, f, fes = get_forms(
        mesh,
        matrix_coeff=matrix_coeff,
        vector_coeff=vector_coeff,
        scalar_coeff=scalar_coeff,
        order=parameters["order"],
        is_complex=True,
    )
    u = solve(a, f, fes)

    to_file(eta_dict, "etas_eig_problem.csv")
    to_file(eig_dict, "eval_eig_problem.csv")
    to_file(err_dict, "errs_eig_problem.csv")

    solution = {
        "u": u,
        "mesh": mesh,
    }
    return eta_dict, eig_dict, err_dict, solution


def test_ard_eig_arnoldi(
    parameters=None,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    # **kwargs,
):
    """
    Solve the primal problem and use the right eiegnvalue estimator.
    The implementation uses the Arnoldi eigensolver.
    """
    assert parameters is not None, "Parameters are required"
    assert matrix_coeff is not None, "Matrix coefficient is required"
    assert vector_coeff is not None, "Vector coefficient is required"
    assert scalar_coeff is not None, "Scalar coefficient is required"
    print(f"\tParameters: {parameters}")

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
        mark(mesh, eta, theta=parameters["theta"])
        # Refine
        mesh.Refine()

    to_file(eta_dict, "etas_eig_problem.csv")
    to_file(eig_dict, "eval_eig_problem.csv")
    to_file(err_dict, "errs_eig_problem.csv")

    a, m, f, fes = get_forms(
        mesh,
        matrix_coeff=matrix_coeff,
        vector_coeff=vector_coeff,
        scalar_coeff=scalar_coeff,
        order=parameters["order"],
        is_complex=True,
    )
    u = solve(a, f, fes)

    solution = {
        "u": u,
        "mesh": mesh,
    }

    return eta_dict, eig_dict, err_dict, solution


if __name__ == "__main__":
    # eta_dict, eig_dict, err_dict, sol = test_ard(
    #     parameters={
    #         "order": ORDER,
    #         "maxiter": MAXITER,
    #         "maxndofs": MAXNDOFS,
    #         "center": CENTER,
    #         "radius": RADIUS,
    #         "npts": NPTS,
    #         "nspan": NSPAN,
    #         "maxh": MAXH,
    #         "theta": THETA,
    #     },
    #     matrix_coeff=CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2)),
    #     vector_coeff=CoefficientFunction((0.0, 0.0)),
    #     scalar_coeff=CoefficientFunction(0.0),
    #     source_coeff=None,
    # )

    eta_dict, eig_dict, err_dict, sol = test_ard_dual(
        parameters={
            "order": ORDER,
            "maxiter": MAXITER,
            "maxndofs": MAXNDOFS,
            "center": CENTER,
            "radius": RADIUS,
            "npts": NPTS,
            "nspan": NSPAN,
            "maxh": MAXH,
            "theta": THETA,
        },
        matrix_coeff=CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2)),
        vector_coeff=CoefficientFunction((20.0, 30.0)),
        scalar_coeff=CoefficientFunction(0.0),
        source_coeff=None,
    )

    Draw(sol["u"], mesh=sol["mesh"], name="Solution")
    Draw(sol["v"], mesh=sol["mesh"], name="Dual Solution")
    input("Press any key to continue...")

    # eta_dict, eig_dict, err_dict, sol = test_ard_eig_feast(
    #     parameters={
    #         "order": ORDER,
    #         "maxiter": MAXITER,
    #         "maxndofs": MAXNDOFS,
    #         "center": CENTER,
    #         "radius": RADIUS,
    #         "npts": NPTS,
    #         "nspan": NSPAN,
    #         "maxh": MAXH,
    #         "theta": THETA,
    #     },
    #     matrix_coeff=CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2)),
    #     vector_coeff=CoefficientFunction((0.0, 0.0)),
    #     scalar_coeff=CoefficientFunction(0.0),
    # )

    # eta_dict, eig_dict, err_dict, sol = test_ard_eig_arnoldi(
    #     parameters={
    #         "order": ORDER,
    #         "maxiter": MAXITER,
    #         "maxndofs": MAXNDOFS,
    #         "center": CENTER,
    #         "radius": RADIUS,
    #         "npts": NPTS,
    #         "nspan": NSPAN,
    #         "maxh": MAXH,
    #         "theta": THETA,
    #     },
    #     matrix_coeff=CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2)),
    #     vector_coeff=CoefficientFunction((0.0, 0.0)),
    #     scalar_coeff=CoefficientFunction(0.0),
    # )
    print("Done!")
