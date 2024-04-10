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


def make_adaptivity(
    geom=None,
    setup=None,
    estimator_landscape=None,
    estimator_eig=None,
    params: dict = None,
):
    """
    Decorator-like function to create adaptivity routines
    """
    assert geom is not None, "Geometry must be provided"
    assert setup is not None, "Setup function must be provided"
    assert (
        estimator_landscape is not None
    ), "Estimator for landscape problem must be provided"
    assert (
        estimator_eig is not None
    ), "Estimator for eigenvalue problem must be provided"
    assert params is not None, "Parameters must be provided"

    assert "maxh" in params, "maxh must be provided"
    assert "maxiter" in params, "maxiter must be provided"
    assert "theta" in params, "theta must be provided"

    def adaptivity_landscape():
        """
        Adaptivity routine for a given geometry, setup, estimator and parameters.
        It refines w.r.t. the error estimator for the landscape problem.
        """
        mesh = Mesh(geom().GenerateMesh(maxh=params["maxh"]))
        a, _, f, fes = setup(mesh, **params)
        iteration = 0
        while iteration < params["maxiter"]:
            iteration += 1
            # Solve
            u = solve_landscape(a, f, fes)
            # Estimate
            eta, _, _ = estimator_landscape(u, **params)
            # Mark
            for element in mesh.Elements():
                mesh.SetRefinementFlag(
                    element, eta[element.nr] > params["theta"] * np.max(eta)
                )
            # Refine
            mesh.Refine()
            # Deep-copy the mesh
            ngmesh = mesh.ngmesh.Copy()
            mesh = Mesh(ngmesh)
            # Update the forms
            a, _, f, fes = setup(mesh, **params)
        return u

    def adaptivity_eig():
        """
        Adaptivity routine for a given geometry, setup, estimator and parameters
        It refines w.r.t. the error estimator for the eigenvalue problem.
        """
        mesh = Mesh(geom().GenerateMesh(maxh=params["maxh"]))
        a, m, _, fes = setup(mesh, **params)
        iteration = 0
        while iteration < params["maxiter"]:
            iteration += 1
            # Solve
            right_ev, left_ev, lambdas = solve_eigenvalue(a, m, fes, **params)
            # Estimate
            eta, _, _ = estimator_eig(right_ev, left_ev, lambdas, **params)
            # Mark
            for element in mesh.Elements():
                mesh.SetRefinementFlag(
                    element, eta[element.nr] > params["theta"] * np.max(eta)
                )
            # Refine
            mesh.Refine()
            # Deep-copy the mesh
            ngmesh = mesh.ngmesh.Copy()
            mesh = Mesh(ngmesh)
            # Update the forms
            a, m, _, fes = setup(mesh, **params)
        return right_ev, left_ev, lambdas

    return adaptivity_landscape, adaptivity_eig


def null_estimator(*args, **kwargs):
    """
    Null estimator that does nothing
    """
    raise NotImplementedError("Null estimator! Check the implementation!")


def adaptivity_laplace():
    """
    Adaptivity routine for the Laplace problem
    """
    adaptivity_landscape, adaptivity_eig = make_adaptivity(
        geom=make_unit_square,
        setup=setup_laplace,
        estimator_landscape=landscape_error_estimator,
        estimator_eig=null_estimator,
        params={
            "maxh": 0.1,
            "maxiter": 5,
            "theta": 0.5,
            "matrix_coeff": 1.0,
            "potential": 0.0,
            "order": 5,
        },
    )
    return adaptivity_landscape, adaptivity_eig


if __name__ == "__main__":
    adaptivity_landscape, adaptivity_eig = adaptivity_laplace()
    u = adaptivity_landscape()
    Draw(u)
    print("Done!")
