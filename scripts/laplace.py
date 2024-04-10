"""
Laplace equation in 2D on a unit square domain with Dirichlet boundary conditions.
"""

import numpy as np
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


def get_forms(mesh, order: int = 1, is_complex: bool = True):
    """
    Get the forms and finite element space for the Laplace equation
    """
    fes = H1(
        mesh, order=order, complex=is_complex, dirichlet="boundary", autoupdate=True
    )
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
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
        "radius": 1.0,
        "nspan": 4,
        "npts": 4,
        "checks": False,
        "seed": 1,
        "hermitian": False,
        "stop_tol": 1e-12,
        "cut_tol": 1e-14,
        "nrestarts": 10,
        "niterations": 100,
        "verbose": True,
    }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    return kwargs


# Error estimator
def error_estimator(gf):
    """
    Compute the landscape error estimator.
    """
    h = specialcf.mesh_size
    n = specialcf.normal(gf.space.mesh.dim)
    xs = [x, y, z][: gf.space.mesh.dim]

    grad_gf = grad(gf)
    lap_gf = sum((grad_gf[i].Diff(xs[i]) for i in range(gf.space.mesh.dim)))

    integrand_1 = lap_gf - 1.0
    integrand_2 = 0.5**0.5 * h**0.5 * (grad_gf - grad_gf.Other()) * n

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
def adaptivity(order: int = 1, maxiter: int = 5, theta: float = 0.9):
    """
    Adaptivity routine for a given geometry, setup, estimator and parameters.
    It refines w.r.t. the error estimator for the landscape problem.
    """
    assert 0 < theta < 1, "Theta must be in (0, 1)"
    assert maxiter > 0, "Maxiter must be positive"
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.3))
    # mesh = Mesh(make_l_shape().GenerateMesh(maxh=0.3))
    a, _, f, fes = get_forms(mesh, order=order, is_complex=True)
    iteration = 0
    while iteration < maxiter:
        iteration += 1
        Draw(mesh)
        input("Press any key...")
        # Solve
        u = solve(a, f, fes)
        # Estimate
        eta, _, _ = error_estimator(u)
        # Mark
        mark(mesh, eta, theta=theta)
        # Refine
        mesh.Refine()
        # Update the forms
        a, _, f, fes = get_forms(mesh, order=order, is_complex=True)
    Draw(u, mesh, name="Solution")
    return u, mesh


def adaptivity_eigenvalue(
    order: int = 1, maxiter: int = 5, theta: float = 0.9, **kwargs
):
    """
    TBA
    """


ORDER = 3
MAXITER = 30
THETA = 0.9

CENTER = 5 * np.pi**2 + 325
RADIUS = 1.0
NPTS = 4
NSPAN = 4

if __name__ == "__main__":
    _, mesh = adaptivity(order=ORDER, maxiter=MAXITER, theta=THETA)
    a, m, _, fes = get_forms(mesh, order=ORDER)
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
