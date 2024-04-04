"""
Set up different systems of PDEs
"""
from ngsolve import (
        H1, grad, dx, Mesh, Draw, BilinearForm, LinearForm)
from geo2d import make_unit_square


def setup_laplace(
        mesh,
        order: int = 1,
        is_complex: bool = True):
    """
    Set up the Laplace problem
    """
    # RHS
    fes = H1(mesh, order=order, complex=is_complex, dirichlet='boundary')
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    # LHS (for eigenvalue problem)
    m = BilinearForm(fes)
    m += u * v * dx
    # Linear form (for the landscape function)
    f = LinearForm(fes)
    f += 1.0 * v * dx
    assemble(a, m, f)
    return a, m, f, fes


def setup_helmholtz(
        mesh,
        coeff: float,
        order: int = 1,
        is_complex: bool = True):
    """
    Set up the Helmholtz problem
    """
    # RHS
    fes = H1(mesh, order=order, complex=is_complex, dirichlet='boundary')
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + coeff * u * v) * dx
    # LHS (for eigenvalue problem)
    m = BilinearForm(fes)
    m += u * v * dx
    # Linear form (for the landscape function)
    f = LinearForm(fes)
    f += v * dx
    assemble(a, m, f)
    return a, m, f, fes


def assemble(*args) -> None:
    """
    Assemble the forms
    """
    for form in args:
        try:
            form.Assemble()
        except AttributeError:
            # TODO larger heap size
            print(f'Unable to assemble {form}')


if __name__ == "__main__":
    ex_mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.3))
    Draw(ex_mesh)
    input('Press Enter to continue...')
    matrix, mass, rhs, space = setup_laplace(ex_mesh)
    print(f'Number of DOFs: {space.ndof}'
          f'Bilinear form a: {matrix.mat}'
          f'Linear form f: {rhs.vec}'
          f'Mass form m: {mass.mat}')
    input('Press Enter to continue...')
    matrix, mass, rhs, space = setup_helmholtz(ex_mesh, 1.0)
    print(f'Number of DOFs: {space.ndof}'
          f'Bilinear form a: {matrix.mat}'
          f'Linear form f: {rhs.vec}'
          f'Mass form m: {mass.mat}')
    input('Press Enter to continue...')
