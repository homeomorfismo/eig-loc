"""
Set up different systems of PDEs
"""

from ngsolve import (
    H1,
    grad,
    dx,
    Mesh,
    Draw,
    BilinearForm,
    LinearForm,
    L2,
    specialcf,
    TaskManager,
    SetHeapSize,
)
from geo2d import make_unit_square


def setup_laplace(mesh, order: int = 1, is_complex: bool = True):
    """
    Set up the Laplace problem
    """
    # RHS
    fes = H1(
        mesh, order=order, complex=is_complex, dirichlet="boundary", autoupdate=True
    )
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


def setup_helmholtz(mesh, coeff: float, order: int = 1, is_complex: bool = True):
    """
    Set up the Helmholtz problem
    """
    # RHS
    fes = H1(
        mesh, order=order, complex=is_complex, dirichlet="boundary", autoupdate=True
    )
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


def setup_adv_diff(
    mesh, matrix_coeff=None, potential=None, order: int = 1, is_complex: bool = True
):
    """
    Set up the advection-diffusion problem
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert potential is not None, "Potential must be provided"
    # RHS
    fes = H1(
        mesh, order=order, complex=is_complex, dirichlet="boundary", autoupdate=True
    )
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += ((matrix_coeff * grad(u)) * grad(v) + potential * u * v) * dx
    # LHS (for eigenvalue problem)
    m = BilinearForm(fes)
    m += u * v * dx
    # Linear form (for the landscape function)
    f = LinearForm(fes)
    f += v * dx
    assemble(a, m, f)
    return a, m, f, fes


def setup_adv_diff_dg(
    mesh,
    matrix_coeff=None,
    potential=None,
    gamma=None,
    order: int = 1,
    is_complex: bool = True,
):
    """
    Set up the advection-diffusion problem with discontinuous Galerkin
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert potential is not None, "Potential must be provided"
    assert gamma is not None, "Jump penalty must be provided"

    fes = L2(mesh, order=order, complex=is_complex, dgjumps=True)
    u, v = fes.TnT()
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size

    jump_u = u - u.Other()
    jump_v = v - v.Other()
    mean_grad_u = 0.5 * n * (matrix_coeff * grad(u) + matrix_coeff * grad(u.Other()))
    mean_grad_v = 0.5 * n * (matrix_coeff * grad(v) + matrix_coeff * grad(v.Other()))

    a = BilinearForm(fes)
    a += matrix_coeff * grad(u) * grad(v) * dx
    a += potential * u * v * dx
    # TODO: Define c
    a += order**2 * gamma / h * jump_u * jump_v * dx(skeleton=True)
    a += -mean_grad_u * jump_v * dx(skeleton=True)
    a += -mean_grad_v * jump_u * dx(skeleton=True)
    # TODO: These terms are in the tutorial
    # a += order**2 * gamma / h * u * v * dx(skeleton=True)
    a += (-n * matrix_coeff * grad(u) * v - n * matrix_coeff * grad(v) * u) * dx(
        skeleton=True
    )

    m = BilinearForm(fes)
    m += u * v * dx

    f = LinearForm(fes)
    f += v * dx
    assemble(a, m, f)
    return a, m, f, fes


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


if __name__ == "__main__":
    ex_mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.3))
    Draw(ex_mesh)
    input("Press Enter to continue...")

    matrix, mass, rhs, space = setup_laplace(ex_mesh)
    print(
        f"Number of DOFs: {space.ndof}"
        f"Bilinear form a: {matrix.mat}"
        f"Linear form f: {rhs.vec}"
        f"Mass form m: {mass.mat}"
    )
    input("Press Enter to continue...")

    matrix, mass, rhs, space = setup_helmholtz(ex_mesh, 1.0)
    print(
        f"Number of DOFs: {space.ndof}"
        f"Bilinear form a: {matrix.mat}"
        f"Linear form f: {rhs.vec}"
        f"Mass form m: {mass.mat}"
    )
    input("Press Enter to continue...")

    matrix, mass, rhs, space = setup_adv_diff(ex_mesh, 1.0, 1.0)
    print(
        f"Number of DOFs: {space.ndof}"
        f"Bilinear form a: {matrix.mat}"
        f"Linear form f: {rhs.vec}"
        f"Mass form m: {mass.mat}"
    )
    input("Press Enter to continue...")

    matrix, mass, rhs, space = setup_adv_diff_dg(ex_mesh, 1.0, 1.0, 1.0)
    print(
        f"Number of DOFs: {space.ndof}"
        f"Bilinear form a: {matrix.mat}"
        f"Linear form f: {rhs.vec}"
        f"Mass form m: {mass.mat}"
    )
    input("Press Enter to continue...")
