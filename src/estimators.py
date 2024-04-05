"""
Error estimators implementation.
"""

from ngsolve import (
    GridFunction,
    Integrate,
    specialcf,
    InnerProduct,
    grad,
    x,
    y,
    z,
    dx,
    Mesh,
)
import numpy as np
from numpy import sqrt

from geo2d import make_unit_square
from setup import setup_laplace
from solver import solve_landscape


def compute_avr_ngvecs(ngvecs, name="avg") -> GridFunction:
    """
    Compute the average of the norm of the eigenvectors.
    """
    gf = GridFunction(ngvecs.fes, name=name, autoupdate=ngvecs.fes.autoupdate)
    for i in range(ngvecs.m):
        gf.vec.data += ngvecs._mv[i].data
    gf.vec.data /= ngvecs.m
    return gf


def landscape_error_estimator_dg(
    gf, order=None, potential=None, matrix_coeff=None, gamma=None
):
    """
    Compute the landscape error estimator.
    """
    assert order is not None, "Order must be specified"
    assert potential is not None, "Potential must be specified"
    assert matrix_coeff is not None, "Matrix coefficient must be specified"
    assert gamma is not None, "Gamma must be specified"

    # gf = compute_avr_ngvecs(ngvecs)

    h = specialcf.mesh_size
    n = specialcf.normal(gf.space.mesh.dim)
    p = order
    xs = [x, y, z][: gf.space.mesh.dim]

    grad_gf = grad(gf)
    a_grad_gf = matrix_coeff * grad_gf
    div_a_grad_gf = sum((a_grad_gf[i].Diff(xs[i]) for i in range(gf.space.mesh.dim)))
    v_gf = potential * gf

    # TODO: Up to a factor related to the matrix coefficient
    integrand_1 = h / p * (1.0 + div_a_grad_gf - v_gf)
    integrand_2 = 0.5**0.5 * (h / p) ** 0.5 * (a_grad_gf - a_grad_gf.Other()) * n
    integrand_3 = (
        0.5**0.5 * (h / p + gamma**2 * p**3 / h) ** 0.5 * (gf - gf.Other()) * n
    )
    integrand_4 = (h / p + gamma**2 * p**3 / h) ** 0.5 * gf

    eta_1 = Integrate(
        InnerProduct(integrand_1, integrand_1) * dx, gf.space.mesh, element_wise=True
    )
    eta_2 = Integrate(
        InnerProduct(integrand_2, integrand_2) * dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )
    eta_3 = Integrate(
        InnerProduct(integrand_3, integrand_3) * dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )
    eta_4 = Integrate(
        InnerProduct(integrand_4, integrand_4) * dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )

    eta = sqrt(eta_1.NumPy() + eta_2.NumPy() + eta_3.NumPy() + eta_4.NumPy())
    etas = {"eta_1": eta_1, "eta_2": eta_2, "eta_3": eta_3, "eta_4": eta_4}
    max_etas = {
        "max_eta_1": np.max(eta_1.NumPy()),
        "max_eta_2": np.max(eta_2.NumPy()),
        "max_eta_3": np.max(eta_3.NumPy()),
        "max_eta_4": np.max(eta_4.NumPy()),
    }
    return eta, etas, max_etas


if __name__ == "__main__":
    print("Generating the mesh...\n")
    ex_mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    print("Solving for the Laplace problem...\n")
    matrix, mass, rhs, space = setup_laplace(ex_mesh, order=5)
    print("Solving for the landscape function...\n")
    landscape = solve_landscape(matrix, rhs, space)
    print("Computing the landscape error estimator...\n")
    # TODO: Do I need gamma here? Not using DG
    eta, etas, max_etas = landscape_error_estimator_dg(
        landscape, order=5, potential=0.0, matrix_coeff=1.0, gamma=1.0
    )
    print(f"Error estimator: {eta}")
    print(f"Max error estimator components: {max_etas}")
