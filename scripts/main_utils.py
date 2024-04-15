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
    BilinearForm,
    ArnoldiSolver,
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
    exp,
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
    source_coeff=None,
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
    if source_coeff is not None:
        f += source_coeff * v * dx
    else:
        f += 1.0 * v * dx
    assemble(a, m, f)
    return a, m, f, fes


def get_exp_potential(tuple_values):
    """
    Get the exponential potential
    """
    return CoefficientFunction(
        exp(0.5 * tuple_values[0] * x + 0.5 * tuple_values[1] * y)
    )


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


def solve_eigenvalue_arnoldi(a, m, fes, **kwargs) -> GridFunction:
    """
    Solve for the eigenvalue problem using Arnoldi iteration
    """
    defaults = get_default_values_feast(**kwargs)
    u = GridFunction(
        fes, multidim=defaults["nspan"], autoupdate=True, name="Eigenfunctions"
    )
    with TaskManager():
        evals = ArnoldiSolver(
            a.mat, m.mat, fes.FreeDofs(), list(u.vecs), shift=defaults["center"]
        )
    return u, evals


def get_default_values_feast(**kwargs):
    """
    Get default values for FEAST
    """
    defaults = {
        "center": 1.0,
        "radius": 0.01,
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
    gf, matrix_coeff=None, vector_coeff=None, scalar_coeff=None, source_coeff=None
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

    if source_coeff is not None:
        integrand_1 = -div_grad_gf + vec_grad_gf + scalar_coeff * gf - source_coeff
    else:
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


def error_estimator_grad_landscape(
    gf, matrix_coeff=None, vector_coeff=None, scalar_coeff=None
):
    """
    Compute the gradient of the landscape error estimator.
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert vector_coeff is not None, "Vector coefficient must be provided"
    assert scalar_coeff is not None, "Scalar coefficient must be provided"
    h = specialcf.mesh_size
    n = specialcf.normal(gf.space.mesh.dim)
    if gf.space.mesh.dim == 2:
        x_vec = CoefficientFunction((x / 2, y / 2))
    else:
        x_vec = CoefficientFunction((x / 3, y / 3, z / 3))

    grad_gf = grad(gf)
    mat_grad_gf = matrix_coeff * grad_gf
    vec_grad_gf = vector_coeff * grad_gf

    integrand_1 = mat_grad_gf + x_vec
    integrand_2 = 0.5**0.5 * h**0.5 * (x_vec - x_vec.Other()) * n
    integrand_3 = h * (vec_grad_gf + scalar_coeff * gf)

    eta_1 = Integrate(
        InnerProduct(integrand_1, integrand_1) * dx, gf.space.mesh, element_wise=True
    )
    eta_2 = Integrate(
        InnerProduct(integrand_2, integrand_2) * dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )
    eta_3 = Integrate(
        InnerProduct(integrand_3, integrand_3) * dx, gf.space.mesh, element_wise=True
    )

    eta = np.sqrt(eta_1.NumPy().real + eta_2.NumPy().real + eta_3.NumPy().real)
    etas = {"eta_1": eta_1, "eta_2": eta_2, "eta_3": eta_3}
    max_etas = {
        "max_eta_1": np.max(eta_1.NumPy()),
        "max_eta_2": np.max(eta_2.NumPy()),
        "max_eta_3": np.max(eta_3.NumPy()),
    }
    return eta, etas, max_etas


# Marking
def mark(mesh, eta, theta=0.5):
    """
    Mark the elements for refinement
    """
    for element in mesh.Elements():
        mesh.SetRefinementFlag(element, eta[element.nr] > theta * np.max(eta))


# File I/O
def to_file(info, filename):
    """
    Use pandas to save the info to a file.
    """
    df = pd.DataFrame(info)
    df.to_csv(filename)


def from_file(filename):
    """
    Use pandas to read the info from a file.
    """
    return pd.read_csv(filename)


def append_to_dict(dictionary, **kwargs):
    """
    Append value to the list of values in the dictionary
    """
    for key, value in kwargs.items():
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]


if __name__ == "__main__":
    keys = ["ndofs", "rel_error", "eta_max", "eta_avg", "eta_l2", "error"]
