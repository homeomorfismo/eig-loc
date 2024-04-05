"""
Solved for the different PDE problems
"""

from ngsolve import GridFunction, Draw, Mesh
from pyeigfeast import NGvecs, SpectralProjNG
from geo2d import make_unit_square
from setup import setup_laplace, setup_helmholtz


def solve_landscape(a, f, fes) -> GridFunction:
    """
    Solve for the landscape function
    """
    fun = GridFunction(fes)
    fun.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
    return fun


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


if __name__ == "__main__":
    print("Generating the mesh...\n")
    ex_mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    Draw(ex_mesh)
    input("Press any key to continue...")

    print("Solving for the Laplace problem...\n")
    matrix, mass, rhs, space = setup_laplace(ex_mesh, order=5)
    sol = solve_landscape(matrix, rhs, space)
    Draw(sol, ex_mesh, name="Laplace")
    input("Press any key to continue...")

    print("Solving for the eigenvalue problem for the Laplace problem...\n")
    right_ev, left_ev, lambdas = solve_eigenvalue(
        matrix, mass, space, center=9.0, radius=0.1
    )
    for i, lam in enumerate(lambdas):
        print(f"Eigenvalue {i}: {lam}")
        Draw(right_ev[i], ex_mesh, name=f"Right Eigenvector Laplace {i}")
        Draw(left_ev[i], ex_mesh, name=f"Left Eigenvector Laplace {i}")
    input("Press any key to continue...")

    print("Solving for the Helmholtz problem...\n")
    matrix, mass, rhs, space = setup_helmholtz(ex_mesh, 1.0, order=5)
    sol = solve_landscape(matrix, rhs, space)
    Draw(sol, ex_mesh, name="Helmholtz")
    input("Press any key to continue...")

    print("Solving for the eigenvalue problem for the Helmholtz problem...\n")
    right_ev, left_ev, lambdas = solve_eigenvalue(
        matrix, mass, space, center=9.0, radius=0.1
    )
    for i, lam in enumerate(lambdas):
        print(f"\tEigenvalue {i}: {lam}")
        Draw(right_ev[i], ex_mesh, name=f"Right Eigenvector Helmholtz {i}")
        Draw(left_ev[i], ex_mesh, name=f"Left Eigenvector Helmholtz {i}")
    input("Press any key to continue...")
