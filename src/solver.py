"""
Solved for the different PDE problems
"""
from ngsolve import GridFunction, Draw, Mesh, x, y
from ngsolve import H1, dx, BilinearForm, LinearForm, grad
from geo2d import make_unit_square
from setup import setup_laplace, setup_helmholtz


def solve_landscape(a, f, fes) -> GridFunction:
    """
    Solve for the landscape function
    """
    fun = GridFunction(fes)
    fun.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
    return fun


def solve_eigenvalue(a, m, fes) -> GridFunction:
    """
    Solve for the eigenvalue problem
    """
    raise NotImplementedError("Eigenvalue problem not implemented yet")


if __name__ == "__main__":
    ex_mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    Draw(ex_mesh)
    input("Press any key to continue...")

    # Example: Solve for the Laplace problem
    matrix, mass, rhs, space = setup_laplace(ex_mesh, order=5)
    sol = solve_landscape(matrix, rhs, space)
    Draw(sol, ex_mesh, name="Laplace")
    input("Press any key to continue...")

    # Example: Solve for the Helmholtz problem
    matrix, mass, rhs, space = setup_helmholtz(ex_mesh, 1.0, order=5)
    sol = solve_landscape(matrix, rhs, space)
    Draw(sol, ex_mesh, name="Helmholtz")
    input("Press any key to continue...")
