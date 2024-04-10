"""
Set some basic adaptivity routines
"""

import numpy as np
from ngsolve import Mesh, CoefficientFunction, GridFunction, Draw, Redraw
from pyeigfeast import NGvecs, SpectralProjNG
from geo2d import make_unit_square, make_l_shape, make_unit_circle
from setup import setup_laplace, setup_helmholtz, setup_adv_diff
from solver import solve_landscape, solve_eigenvalue
from estimators import landscape_error_estimator, compute_avr_ngvecs
from solver import get_default_values_feast


def adaptivity_laplace(order: int = 1, maxiter: int = 50, theta: float = 0.5):
    """
    Adaptivity routine for a given geometry, setup, estimator and parameters.
    It refines w.r.t. the error estimator for the landscape problem.
    """
    assert 0 < theta < 1, "Theta must be in (0, 1)"
    assert maxiter > 0, "Maxiter must be positive"
    assert order > 0, "Order must be positive"
    # mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.3))
    mesh = Mesh(make_l_shape().GenerateMesh(maxh=0.3))
    a, _, f, fes = setup_laplace(mesh, order=order, is_complex=True)
    iteration = 0
    while iteration < maxiter:
        iteration += 1
        Draw(mesh)
        input("Press any key...")
        # Solve
        u = solve_landscape(a, f, fes)
        # Estimate
        eta, _, _ = landscape_error_estimator(u, potential=0.0, matrix_coeff=1.0)
        # Mark
        for element in mesh.Elements():
            mesh.SetRefinementFlag(element, eta[element.nr] > theta * np.max(eta))
        # Refine
        mesh.Refine()
        # Deep-copy the mesh
        ngmesh = mesh.ngmesh.Copy()
        mesh = Mesh(ngmesh)
        # Update the forms
        a, _, f, fes = setup_laplace(mesh, order=order, is_complex=True)
    return u


if __name__ == "__main__":
    u = adaptivity_laplace()
    Draw(u)
    input("Press any key...")
