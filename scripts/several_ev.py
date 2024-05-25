"""
This script is used to compute the eigenvalues of the L-shape domain with multiple eigenvalues.
"""

import gc
import time
import numpy as np
import pandas as pd
import ngsolve as ng
import ngsolve.internal as ngi
from pyeigfeast import NGvecs, SpectralProjNG
from netgen.geom2d import SplineGeometry
import plotly.express as px
import matplotlib.pyplot as plt

TIME_DELAY = 0.5  # seconds

ORDER = 2
IS_COMPLEX = True
MAX_ITER = 50
# MAX_ITER = 5
THETA = 0.75

# FEAST parameters
NSPAN = 10
NPTS = 10
CHECKS = False
RADIUS = 10.0
CENTER = 50 * np.pi**2

FEAST_PARAMS = {
    "hermitian": True,
    "stop_tol": 1.0e-10,
    "cut_tol": 1.0e-10,
    "eta_tol": 1e-10,
    "nrestarts": 5,
    "niterations": 100,
}


def make_l_shape():
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


def reset_mesh():
    """
    Reset the mesh to the initial state.
    """
    return ng.Mesh(make_l_shape().GenerateMesh(maxh=0.3))


def error_estimator_landscape(gf):
    """
    Compute the landscape error estimator.
    """
    h = ng.specialcf.mesh_size
    n = ng.specialcf.normal(gf.space.mesh.dim)

    temp_vec_space = ng.VectorL2(gf.space.mesh, order=ORDER + 1, complex=IS_COMPLEX)
    grad_gf = ng.GridFunction(temp_vec_space)

    grad_gf.Set(ng.grad(gf))
    grad_grad_gf = ng.Grad(grad_gf)
    div_grad_gf = grad_grad_gf[0, 0] + grad_grad_gf[1, 1]

    integrand_1 = -div_grad_gf - 1.0
    integrand_2 = (grad_gf - grad_gf.Other()) * n

    eta_1 = ng.Integrate(
        h**2 * ng.InnerProduct(integrand_1, integrand_1) * ng.dx,
        gf.space.mesh,
        element_wise=True,
    )
    eta_2 = ng.Integrate(
        0.5
        * h
        * ng.InnerProduct(integrand_2, integrand_2)
        * ng.dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )

    eta_ls = np.sqrt(eta_1.NumPy().real + eta_2.NumPy().real)
    etas = {
        "eta_1": eta_1.NumPy().real,
        "eta_2": eta_2.NumPy().real,
    }

    return eta_ls, etas


def error_estimator_ev(gf, ev):
    """
    Compute the landscape error estimator.
    """
    h = ng.specialcf.mesh_size
    n = ng.specialcf.normal(gf.space.mesh.dim)

    temp_vec_space = ng.VectorL2(gf.space.mesh, order=ORDER + 1, complex=IS_COMPLEX)
    grad_gf = ng.GridFunction(temp_vec_space)

    grad_gf.Set(ng.grad(gf))
    grad_grad_gf = ng.Grad(grad_gf)
    div_grad_gf = grad_grad_gf[0, 0] + grad_grad_gf[1, 1]

    integrand_1 = -div_grad_gf - ev * gf
    integrand_2 = (grad_gf - grad_gf.Other()) * n

    eta_1 = ng.Integrate(
        h**2 * ng.InnerProduct(integrand_1, integrand_1) * ng.dx,
        gf.space.mesh,
        element_wise=True,
    )
    eta_2 = ng.Integrate(
        0.5
        * h
        * ng.InnerProduct(integrand_2, integrand_2)
        * ng.dx(element_boundary=True),
        gf.space.mesh,
        element_wise=True,
    )

    eta_ev = np.sqrt(eta_1.NumPy().real + eta_2.NumPy().real)
    etas = {
        "eta_1": eta_1.NumPy().real,
        "eta_2": eta_2.NumPy().real,
    }

    return eta_ev, etas


def assemble(*args):
    """
    Assemble the forms.
    """
    with ng.TaskManager():
        for form in args:
            try:
                form.Assemble()
            except Exception as e:
                print(f"Unable to assemble {form}, increasing heap size\nError: {e}")
                ng.SetHeapSize(int(1e9))
                form.Assemble()
            finally:
                pass


if __name__ == "__main__":
    mesh = reset_mesh()
    fes = ng.H1(
        mesh, order=ORDER, complex=IS_COMPLEX, dirichlet="boundary", autoupdate=True
    )

    u, v = fes.TnT()

    a = ng.BilinearForm(fes)
    a += ng.grad(u) * ng.grad(v) * ng.dx

    m = ng.BilinearForm(fes)
    m += u * v * ng.dx

    f = ng.LinearForm(fes)
    f += 1.0 * v * ng.dx

    assemble(a, m, f)

    iteration = 0
    sol = ng.GridFunction(fes, name="Landscape", autoupdate=True)

    # --- Lists ---
    Ndofs = []
    Eta = []
    Eta_ev = []
    Evals = []

    # Main loop
    input("Press Enter to start the loop")
    while iteration < MAX_ITER:
        # Draw mesh
        ng.Draw(mesh, name="Mesh")
        time.sleep(TIME_DELAY)
        ngi.SnapShot(f"mesh_{iteration}.png")
        # Update
        assemble(a, m, f)
        iteration += 1
        # Solve
        sol.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
        # --- Init  call FEAST ---
        # Set clean eigenspace (required for adaptivity)
        right = NGvecs(fes, NSPAN)
        right.setrandom()
        left = NGvecs(fes, NSPAN)
        left.setrandom()
        # Set spectral projector
        projector = SpectralProjNG(
            fes,
            a.mat,
            m.mat,
            checks=CHECKS,
            radius=RADIUS,
            center=CENTER,
            npts=NPTS,
            verbose=False,
        )
        # Call FEAST
        evalues, right, history, _ = projector.feast(right, **FEAST_PARAMS)
        assert history[-1], "FEAST did not converge"
        # --- End  call  FEAST ---
        # ---   Estimate   ---
        eta_ls, _ = error_estimator_landscape(sol)
        max_eta_ls = np.max(eta_ls)
        etas = []
        for i in range(right.m):
            eta, _ = error_estimator_ev(right.gridfun(i=i), evalues[i])
            etas.append(eta)
        etas = np.stack(etas, axis=0)
        eta_ev = np.max(etas, axis=1)
        # --- End Estimate ---
        # Mark
        mesh.ngmesh.Elements2D().NumPy()["refine"] = eta_ls > THETA * max_eta_ls
        # Store
        Ndofs.append(fes.ndof)
        Eta.append(max_eta_ls)
        Eta_ev.append(eta_ev)
        Evals.append(np.array(evalues))
        # Refine
        if iteration == MAX_ITER:
            break
        mesh.Refine()
        del right, left, projector, history, evalues, etas
        gc.collect()

    ng.Draw(sol, mesh, name="Landscape")
    ng.Draw(right.gridfun(i=None), mesh, name="evs")

    # --- Dataframes ---
    Ndofs = np.array(Ndofs)
    long_ndofs = []
    long_evals = []
    for ndof, evs in zip(np.array(Ndofs), Evals):
        long_ndofs.extend([ndof] * len(evs))
        long_evals.extend(list(evs.real))
    long_evals = np.array(long_evals)
    long_ndofs = np.array(long_ndofs)

    df = pd.DataFrame(
        {
            "Ndofs": long_ndofs,
            "Evals": long_evals,
        }
    )
    df.to_csv("eigenvalues.csv", index=False)

    res_ndofs = Ndofs[Ndofs > 2.0 * 1e4]
    slice_index = list(Ndofs).index(res_ndofs[0])
    res_evs = Evals[slice_index:]
    error = np.array(
        [np.abs(res_evs[i] - res_evs[-1]) for i in range(0, len(res_ndofs) - 1)]
    )

    df = pd.DataFrame(
        {
            "Ndofs": res_ndofs[1:],
            "Error Eigenvalue 0": error[:, 0],
            "Error Eigenvalue 1": error[:, 1],
            "Error Eigenvalue 2": error[:, 2],
            "Error Eigenvalue 3": error[:, 3],
            "Error Eigenvalue 4": error[:, 4],
            "Error Eigenvalue 5": error[:, 5],
            "Error Eigenvalue 6": error[:, 6],
        }
    )
    df.to_csv("error_eigenvalues.csv", index=False)

    res_eta = np.array(Eta[slice_index:])
    res_eta_ev = np.array(Eta_ev[slice_index:])

    df = pd.DataFrame(
        {
            "Ndofs": res_ndofs,
            "Eta": res_eta,
            "Eta_ev": res_eta_ev,
        }
    )
    df.to_csv("error_estimator.csv", index=False)
