"""
ARD equation in 2D on a unit square domain with Dirichlet boundary conditions.
"""

import numpy as np
import pandas as pd
from ngsolve import (
    Draw,
    H1,
    grad,
    dx,
    Mesh,
    BilinearForm,
    LinearForm,
    specialcf,
    TaskManager,
    SetHeapSize,
    GridFunction,
    InnerProduct,
    Integrate,
    x,
    y,
    z,
    CoefficientFunction,
)
from netgen.geom2d import SplineGeometry
from pyeigfeast import NGvecs, SpectralProjNG


# Geometry
def make_unit_square() -> SplineGeometry:
    """
    Create a unit square mesh.
    """
    geo = SplineGeometry()
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    points_ids = [geo.AppendPoint(*point) for point in points]
    lines = [
        [["line", points_ids[i % 4], points_ids[(i + 1) % 4]], "boundary"]
        for i in range(4)
    ]
    for line, bc in lines:
        geo.Append(line, bc=bc)
    return geo


def make_l_shape() -> SplineGeometry:
    """
    Create an L-shaped domain mesh.
    """
    geo = SplineGeometry()
    points = [(0, 0), (0, 1), (-1, 1), (-1, -1), (1, -1), (1, 0)]
    points_ids = [geo.AppendPoint(*point) for point in points]
    lines = [
        [["line", points_ids[i % 6], points_ids[(i + 1) % 6]], "boundary"]
        for i in range(6)
    ]
    for line, bc in lines:
        geo.Append(line, bc=bc)
    return geo


# Finite element spaces and forms
def assemble(*args) -> None:
    """
    Assemble the forms
    """
    for form in args:
        with TaskManager():
            try:
                form.Assemble()
            except Exception as e:
                print(f"Unable to assemble {form}, increasing heap size\nError: {e}")
                SetHeapSize(int(1e9))
                form.Assemble()
            finally:
                pass


def get_forms(
    mesh,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    order: int = 1,
    is_complex: bool = True,
):
    """
    Get the forms and finite element space for the Laplace equation
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert vector_coeff is not None, "Vector coefficient must be provided"
    assert scalar_coeff is not None, "Scalar coefficient must be provided"
    fes = H1(
        mesh, order=order, complex=is_complex, dirichlet="boundary", autoupdate=True
    )
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += matrix_coeff * grad(u) * grad(v) * dx
    a += vector_coeff * grad(u) * v * dx
    a += scalar_coeff * u * v * dx
    m = BilinearForm(fes)
    m += u * v * dx
    f = LinearForm(fes)
    f += 1.0 * v * dx
    assemble(a, m, f)
    return a, m, f, fes


# Solvers
def solve(matrix, rhs, space):
    """
    Solve the Laplace equation
    """
    sol = GridFunction(space, name="Laplace solution", autoupdate=True)
    sol.vec.data = matrix.mat.Inverse(space.FreeDofs()) * rhs.vec
    return sol


def solve_eigenvalue(a, m, fes, **kwargs) -> GridFunction:
    """
    Solve for the eigenvalue problem
    """
    defaults = get_default_values_feast(**kwargs)

    right = NGvecs(fes, defaults["nspan"])
    right.setrandom(seed=defaults["seed"])
    left = NGvecs(fes, defaults["nspan"])
    left.setrandom(seed=defaults["seed"])

    projector = SpectralProjNG(
        fes,
        a.mat,
        m.mat,
        checks=defaults["checks"],
        radius=defaults["radius"],
        center=defaults["center"],
        npts=defaults["npts"],
    )

    evals, right, history, left = projector.feast(
        right,
        Yl=left,
        hermitian=defaults["hermitian"],
        stop_tol=defaults["stop_tol"],
        cut_tol=defaults["cut_tol"],
        nrestarts=defaults["nrestarts"],
        niterations=defaults["niterations"],
    )

    assert history[-1], "FEAST did not converge"

    if defaults["verbose"]:
        print("FEAST converged!")
        print(f"Eigenvalues: {evals}")
    return right, left, evals


def get_default_values_feast(**kwargs):
    """
    Get default values for FEAST
    """
    defaults = {
        "center": 1.0,
        "radius": 0.1,
        "nspan": 2,
        "npts": 4,
        "checks": False,
        "seed": 1,
        "hermitian": False,
        "stop_tol": 1e-11,
        "cut_tol": 1e-11,
        "nrestarts": 5,
        "niterations": 100,
        "verbose": True,
    }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    return kwargs


# Error estimator
def error_estimator_landscape(
    gf, matrix_coeff=None, vector_coeff=None, scalar_coeff=None
):
    """
    Compute the landscape error estimator.
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert vector_coeff is not None, "Vector coefficient must be provided"
    assert scalar_coeff is not None, "Scalar coefficient must be provided"
    h = specialcf.mesh_size
    n = specialcf.normal(gf.space.mesh.dim)
    xs = [x, y, z][: gf.space.mesh.dim]

    grad_gf = grad(gf)
    mat_grad_gf = matrix_coeff * grad_gf
    vec_grad_gf = vector_coeff * grad_gf
    div_grad_gf = sum((mat_grad_gf[i].Diff(xs[i]) for i in range(gf.space.mesh.dim)))

    integrand_1 = -div_grad_gf + vec_grad_gf + scalar_coeff * gf - 1.0
    integrand_2 = 0.5**0.5 * h**0.5 * (mat_grad_gf - mat_grad_gf.Other()) * n

    eta_1 = Integrate(
        InnerProduct(integrand_1, integrand_1) * dx, gf.space.mesh, element_wise=True
    )
    eta_2 = Integrate(
        InnerProduct(integrand_2, integrand_2) * dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )

    eta = np.sqrt(eta_1.NumPy().real + eta_2.NumPy().real)
    etas = {"eta_1": eta_1, "eta_2": eta_2}
    max_etas = {
        "max_eta_1": np.max(eta_1.NumPy()),
        "max_eta_2": np.max(eta_2.NumPy()),
    }
    return eta, etas, max_etas


def error_estimator_eigenfunction(
    gf, ev, matrix_coeff=None, vector_coeff=None, scalar_coeff=None
):
    """
    Compute the eigenfunction error estimator.
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert vector_coeff is not None, "Vector coefficient must be provided"
    assert scalar_coeff is not None, "Scalar coefficient must be provided"
    h = specialcf.mesh_size
    n = specialcf.normal(gf.space.mesh.dim)
    xs = [x, y, z][: gf.space.mesh.dim]

    grad_gf = grad(gf)
    mat_grad_gf = matrix_coeff * grad_gf
    vec_grad_gf = vector_coeff * grad_gf
    div_grad_gf = sum((mat_grad_gf[i].Diff(xs[i]) for i in range(gf.space.mesh.dim)))

    integrand_1 = -div_grad_gf + vec_grad_gf + scalar_coeff * gf - ev * gf
    integrand_2 = 0.5**0.5 * h**0.5 * (mat_grad_gf - mat_grad_gf.Other()) * n

    eta_1 = Integrate(
        InnerProduct(integrand_1, integrand_1) * dx, gf.space.mesh, element_wise=True
    )
    eta_2 = Integrate(
        InnerProduct(integrand_2, integrand_2) * dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )

    eta = np.sqrt(eta_1.NumPy().real + eta_2.NumPy().real)
    etas = {"eta_1": eta_1, "eta_2": eta_2}
    max_etas = {
        "max_eta_1": np.max(eta_1.NumPy()),
        "max_eta_2": np.max(eta_2.NumPy()),
    }
    return eta, etas, max_etas


# Marking
def mark(mesh, eta, theta=0.5):
    """
    Mark the elements for refinement
    """
    for element in mesh.Elements():
        mesh.SetRefinementFlag(element, eta[element.nr] > theta * np.max(eta))


# Full adaptivity routine
# TODO: Add error decresing
def adaptivity(order: int = 1, maxiter: int = 5, theta: float = 0.9):
    """
    Adaptivity routine for a given geometry, setup, estimator and parameters.
    It refines w.r.t. the error estimator for the landscape problem.
    """
    assert 0 < theta < 1, "Theta must be in (0, 1)"
    assert maxiter > 0, "Maxiter must be positive"
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.3))
    a, _, f, fes = get_forms(
        mesh,
        matrix_coeff=matrix,
        vector_coeff=vector,
        scalar_coeff=scalar,
        order=order,
        is_complex=True,
    )
    iteration = 0
    etas = []
    ndofs = []
    while iteration < maxiter:
        iteration += 1
        Draw(mesh)
        # input("Press any key...")
        # Solve
        u = solve(a, f, fes)
        # Estimate
        eta, _, _ = error_estimator_landscape(
            u, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
        )
        # Save info
        etas.append(eta)
        ndofs.append(fes.ndof)
        # Mark
        mark(mesh, eta, theta=theta)
        # Refine
        mesh.Refine()
        # Update the forms
        a, _, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=order,
            is_complex=True,
        )
    Draw(u, mesh, name="Solution")
    input("Press any key to continue...")
    info = {
        "etas": etas,
        "ndofs": ndofs,
    }
    return u, mesh, info


def adaptivity_eigenvalue(
    order: int = 1, maxiter: int = 5, theta: float = 0.9, **kwargs
):
    """
    Adaptivity routine for a given geometry, setup, estimator and parameters.
    It refines w.r.t. the error estimator for the eigenvalue problem.
    """
    assert 0 < theta < 1, "Theta must be in (0, 1)"
    assert maxiter > 0, "Maxiter must be positive"
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.3))
    iteration = 0
    etas = []
    error = []
    ndofs = []
    lambdas = [0.0]
    center = kwargs.get("center")
    kwargs.pop("center")
    while iteration < maxiter:
        iteration += 1
        Draw(mesh)
        # Update the forms
        a, m, _, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=order,
            is_complex=True,
        )
        # input("Press any key...")
        # Solve
        right, left, evals = solve_eigenvalue(a, m, fes, center=center, **kwargs)
        # Estimate
        estimators = []
        for i in range(right.m):
            u = right[i]
            v = left[i]
            eta_r, _, _ = error_estimator_eigenfunction(
                u,
                evals[i],
                matrix_coeff=matrix,
                vector_coeff=vector,
                scalar_coeff=scalar,
            )
            eta_l, _, _ = error_estimator_eigenfunction(
                v,
                evals[i],
                matrix_coeff=matrix,
                vector_coeff=vector,
                scalar_coeff=scalar,
            )
            estimators.append(eta_r)
            estimators.append(eta_l)
        # Take the maximum error estimator
        eta = np.max(estimators, axis=0)
        # Save info
        etas.append(eta)
        lambdas.append(np.mean(evals))
        error.append(np.absolute(lambdas[-1] - lambdas[-2]))
        ndofs.append(fes.ndof)
        # Update the center
        center = lambdas[-1]
        # Mark
        mark(mesh, eta, theta=theta)
        # Refine
        if iteration < maxiter:
            mesh.Refine()
    print(f"Eigenvalues: {evals}")
    info = {
        "etas": etas,
        "ndofs": ndofs,
        "lambdas": lambdas[1:],
        "error": error,
    }
    return right, left, evals, mesh, info


def to_file(info, filename):
    """
    Use pandas to save the info to a file.
    """
    df = pd.DataFrame(info)
    df.to_csv(filename)


#########################################################################################
# Tests #################################################################################
#########################################################################################

ORDER = 3
MAXITER = 30
THETA = 0.9

CENTER = 5 * np.pi**2 + 325
RADIUS = 1.0
NPTS = 6
NSPAN = 2


def test_adaptivity_landscape():
    """
    Test the adaptivity routine for the landscape problem, with the error estimator.
    """
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    _, mesh, info = adaptivity(order=ORDER, maxiter=MAXITER, theta=THETA)
    to_file(info, "landscape_info.csv")
    a, m, _, fes = get_forms(
        mesh, order=ORDER, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
    )
    right, left, evals = solve_eigenvalue(
        a, m, fes, center=CENTER, radius=RADIUS, npts=NPTS, nspan=NSPAN
    )
    for k in range(right.m):
        Draw(right[k], mesh, name=f"Eigenfunction {k}")
        Draw(left[k], mesh, name=f"Left Eigenfunction {k}")
        input(f"\tPress any key to continue... Eigenvalue {k}: {evals[k]}")
    print(f"Eigenvalues: {evals}")
    print(f"Number of eigenvalues: {len(evals)}")
    print("Done!")


def test_adaptivity_eigenvalue():
    """
    Test the adaptivity routine for the eigenvalue problem, with the error estimator.
    """
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)
    right, left, evals, mesh, info = adaptivity_eigenvalue(
        order=ORDER,
        maxiter=MAXITER,
        theta=THETA,
        center=CENTER,
        radius=RADIUS,
        npts=NPTS,
        nspan=NSPAN,
    )
    to_file(info, "eigenvalue_info.csv")
    for k in range(right.m):
        Draw(right[k], mesh, name=f"Eigenfunction {k}")
        Draw(left[k], mesh, name=f"Left Eigenfunction {k}")
        input(f"\tPress any key to continue... Eigenvalue {k}: {evals[k]}")
    a, _, f, fes = get_forms(
        mesh, order=ORDER, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
    )
    u = solve(a, f, fes)
    Draw(u, mesh, name="Landscape solution")
    print(f"Eigenvalues: {evals}")
    print(f"Number of eigenvalues: {len(evals)}")
    print("Done!")


def test_adaptivity_landscape_ev():
    """
    Test the adaptivity routine for the landscape problem, with the error estimator.
    We compute the eigenvalues and eigenfunctions of the landscape problem, per each iteration.
    """
    matrix = CoefficientFunction((1, 0, 0, 1), dims=(2, 2))
    vector = CoefficientFunction((20, 30))
    scalar = CoefficientFunction(0.0)

    mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.2))

    iteration = 0
    etas_max = []
    etas_l2 = []
    etas = []
    ndofs = []
    avg_evals = []
    avg_evals_real = []
    avg_evals_imag = []
    error_avg = []
    error = []
    center = CENTER
    while iteration < MAXITER:
        iteration += 1
        # Update the forms
        a, m, f, fes = get_forms(
            mesh,
            matrix_coeff=matrix,
            vector_coeff=vector,
            scalar_coeff=scalar,
            order=ORDER,
            is_complex=True,
        )
        Draw(mesh)
        # Solve
        u = solve(a, f, fes)
        # Solve eigenvalue problem
        _, _, evals = solve_eigenvalue(a, m, fes, center=center)
        # Estimate
        eta, _, _ = error_estimator_landscape(
            u, matrix_coeff=matrix, vector_coeff=vector, scalar_coeff=scalar
        )
        # Save info
        etas.append(eta)
        etas_max.append(np.max(eta))
        etas_l2.append(np.linalg.norm(eta))
        ndofs.append(fes.ndof)
        avg_evals.append(np.mean(evals))
        avg_evals_real.append(np.real(np.mean(evals)))
        avg_evals_imag.append(np.imag(np.mean(evals)))
        error_avg.append(np.absolute(avg_evals[-1] - CENTER))
        error.append(np.absolute(np.array(evals) - CENTER))
        # Update the center
        center = avg_evals[-1]
        # Mark
        mark(mesh, eta, theta=THETA)
        # Refine
        mesh.Refine()
    Draw(u, mesh, name="Solution")
    input("Press any key to continue...")
    info = {
        # "etas": etas,
        "etas_max": etas_max,
        "etas_l2": etas_l2,
        "ndofs": ndofs,
        "avg_evals": avg_evals,
        "avg_evals_real": avg_evals_real,
        "avg_evals_imag": avg_evals_imag,
        "error_avg": error_avg,
        # "error": error,
    }
    to_file(info, "lands_ev_info.csv")


if __name__ == "__main__":
    # test_adaptivity_landscape()
    # print("\n\n\nLandscape done!\n\n\n")
    # test_adaptivity_eigenvalue()
    # print("\n\n\nEigenvalue done!\n\n\n")
    test_adaptivity_landscape_ev()
    print("\n\n\nLandscape and eigenvalues done!\n\n\n")
    print("All tests done!")
