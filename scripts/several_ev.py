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
# import plotly.express as px
# import matplotlib.pyplot as plt

TIME_DELAY = 0.5  # seconds

ORDER = 2
IS_COMPLEX = True
MAX_ITER = 45
THETA = 0.75

# For mixed strategy
MAX_NDOFS_LANDSCAPE = 8e3
MAX_NDOFS_EIGENVALUES = 1e5
# For landscape and eigenvalues refinement
MAX_NDOFS = 1e5

# FEAST parameters
NSPAN = 10
NPTS = 10
CHECKS = False
RADIUS = 10.0
CENTER = 50 * np.pi**2
REF_EIGS = [485.71752463708, 490.15998172598, 493.48022005447, 493.48022005447, 493.48022005447, 499.24106145290, 502.30119419396]

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


def adaptivity_by_landscape():
    """
    Perform adaptivity by the landscape error estimator.
    """
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
    current_ndofs = fes.ndof
    sol = ng.GridFunction(fes, name="Landscape", autoupdate=True)

    # --- Lists ---
    Ndofs = []
    Eta = []
    Eta_ev = []
    Evals = []

    # Main loop
    input("DESCRIPTION: Adaptivity by landscape")
    while current_ndofs < MAX_NDOFS:
        # Draw mesh
        ng.Draw(mesh, name="Mesh")
        time.sleep(TIME_DELAY)
        ngi.SnapShot(f"mesh_landscape_{iteration}.png")
        # Update
        assemble(a, m, f)
        iteration += 1
        current_ndofs = fes.ndof
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
        # Here we *monitor* the individual error estimators for each eigenvalue
        # so eta_ev = (max_T(eta_1), max_T(eta_2), ..., max_T(eta_N))
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
        Ndofs.append(current_ndofs)
        Eta.append(max_eta_ls)
        Eta_ev.append(eta_ev)
        Evals.append(np.array(evalues))
        # Refine
        if current_ndofs >= MAX_NDOFS:
            break
        mesh.Refine()
        del right, left, projector, history, evalues, etas
        gc.collect()

    ng.Draw(sol, mesh, name="Landscape")
    ng.Draw(right.gridfun(i=None), mesh, name="evs")
    input("\tGet pictures! Then press Enter to continue")

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
    df.to_csv("eigenvalues_landscape.csv", index=False)

    res_ndofs = Ndofs[Ndofs > 2.0 * 1e4]
    slice_index = list(Ndofs).index(res_ndofs[0])
    res_evs = Evals[slice_index:]
    error = np.array(
        [np.abs(res_evs[i] - REF_EIGS) for i in range(1, len(res_ndofs))]
    )
    # error = np.array(
    #     [np.abs(res_evs[i] - res_evs[-1]) for i in range(0, len(res_ndofs) - 1)]
    # )

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
    df.to_csv("error_eigenvalues_landscape.csv", index=False)

    res_eta = np.array(Eta[slice_index:])
    res_eta_ev = np.array(Eta_ev[slice_index:])

    df = pd.DataFrame(
        {
            "Ndofs": res_ndofs,
            "Eta": res_eta,
            "Eta Eigenvalue 0": res_eta_ev[:, 0],
            "Eta Eigenvalue 1": res_eta_ev[:, 1],
            "Eta Eigenvalue 2": res_eta_ev[:, 2],
            "Eta Eigenvalue 3": res_eta_ev[:, 3],
            "Eta Eigenvalue 4": res_eta_ev[:, 4],
            "Eta Eigenvalue 5": res_eta_ev[:, 5],
            "Eta Eigenvalue 6": res_eta_ev[:, 6],
        }
    )
    df.to_csv("error_estimator_landscape.csv", index=False)


def adaptivity_by_eigenvalues():
    """
    Perform adaptivity by the eigenvalues error estimators.
    """
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
    current_ndofs = fes.ndof
    sol = ng.GridFunction(fes, name="Landscape", autoupdate=True)

    # --- Lists ---
    Ndofs = []
    Eta = []
    Eta_ev = []
    Evals = []

    # Main loop
    input("DESCRIPTION: Adaptivity by eigenvalues")
    while current_ndofs < MAX_NDOFS:
        # Draw mesh
        ng.Draw(mesh, name="Mesh")
        time.sleep(TIME_DELAY)
        ngi.SnapShot(f"mesh_eigenvalues_{iteration}.png")
        # Update
        assemble(a, m, f)
        iteration += 1
        current_ndofs = fes.ndof
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
        # Here we *use* a greedy strategy to select the maximum error estimator
        # so eta_ev = (max_i(eta_i(T))_T, i.e., a element-indexed vector
        etas = []
        for i in range(right.m):
            eta, _ = error_estimator_ev(right.gridfun(i=i), evalues[i])
            etas.append(eta)
        etas = np.stack(etas, axis=0)
        eta_ev_monitored = np.max(etas, axis=1)
        eta_ev = np.max(etas, axis=0)
        max_eta_ev = np.max(eta_ev)
        # --- End Estimate ---
        # Mark
        mesh.ngmesh.Elements2D().NumPy()["refine"] = eta_ev > THETA * max_eta_ev
        # Store
        Ndofs.append(current_ndofs)
        Eta.append(max_eta_ls)
        # TODO
        Eta_ev.append(eta_ev_monitored)
        Evals.append(np.array(evalues))
        # Refine
        if current_ndofs >= MAX_NDOFS:
            break
        mesh.Refine()
        del right, left, projector, history, evalues, etas
        gc.collect()

    ng.Draw(sol, mesh, name="Landscape")
    ng.Draw(right.gridfun(i=None), mesh, name="evs")
    input("\tGet pictures! Then press Enter to continue")

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
    df.to_csv("eigenvalues_eigenvalues.csv", index=False)

    res_ndofs = Ndofs[Ndofs > 2.0 * 1e4]
    slice_index = list(Ndofs).index(res_ndofs[0])
    res_evs = Evals[slice_index:]
    error = np.array(
        [np.abs(res_evs[i] - REF_EIGS) for i in range(1, len(res_ndofs))]
    )
    # error = np.array(
    #     [np.abs(res_evs[i] - res_evs[-1]) for i in range(0, len(res_ndofs) - 1)]
    # )

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
    df.to_csv("error_eigenvalues_eigenvalues.csv", index=False)

    res_eta = np.array(Eta[slice_index:])
    res_eta_ev = np.array(Eta_ev[slice_index:])

    df = pd.DataFrame(
        {
            "Ndofs": res_ndofs,
            "Eta": res_eta,
            "Eta Eigenvalue 0": res_eta_ev[:, 0],
            "Eta Eigenvalue 1": res_eta_ev[:, 1],
            "Eta Eigenvalue 2": res_eta_ev[:, 2],
            "Eta Eigenvalue 3": res_eta_ev[:, 3],
            "Eta Eigenvalue 4": res_eta_ev[:, 4],
            "Eta Eigenvalue 5": res_eta_ev[:, 5],
            "Eta Eigenvalue 6": res_eta_ev[:, 6],
        }
    )
    df.to_csv("error_estimator_eigenvalues.csv", index=False)


def adaptivity_by_mixed_strategy():
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
    current_ndofs = fes.ndof
    sol = ng.GridFunction(fes, name="Landscape", autoupdate=True)

    # --- Lists ---
    # - We track the eivenvalues in one list
    Evals = []
    # - We make two lists for the Ndofs: one for landscape and one for eigenvalues
    Ndofs_ev = []
    Ndofs_ls = []
    # - Then we make a new list that concatenates the two ndofs lists
    # Ndofs = [] # TBA later
    # - We make a list for the landscape error estimator and one for the eigenvalues error estimator
    #   These lists need a companion lists to store the monitored estimators (i.e., if using landscape,
    #   we monitor the eigenvalues error estimators; if using eigenvalues, we monitor the landscape error
    #   estimators).
    Eta_ls = []
    Eta_ls_ev = []

    Eta_ev = []
    Eta_ev_ls = []

    # Main loop
    input("DESCRIPTION: Adaptivity by mixed strategy")
    # FIRST LOOP: LANDSCAPE
    while current_ndofs < MAX_NDOFS_LANDSCAPE:
        # Draw mesh
        ng.Draw(mesh, name="Mesh")
        time.sleep(TIME_DELAY)
        ngi.SnapShot(f"mesh_mixed_{iteration}.png")
        # Update
        assemble(a, m, f)
        iteration += 1
        current_ndofs = fes.ndof
        # Message
        print("LANDSCAPE strategy")
        print(f"\tCurrent ndofs: {current_ndofs}")
        print(f"\tIteration: {iteration}")
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
        # Here we *monitor* the individual error estimators for each eigenvalue
        # so eta_ev = (max_T(eta_1), max_T(eta_2), ..., max_T(eta_N))
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
        Ndofs_ls.append(current_ndofs)
        Eta_ls.append(max_eta_ls)
        Eta_ls_ev.append(eta_ev)
        Evals.append(np.array(evalues))

        # Refine
        if current_ndofs >= MAX_NDOFS_LANDSCAPE:
            break
        mesh.Refine()
        del right, left, projector, history, evalues, etas
        gc.collect()

    # SECOND LOOP: EIGENVALUES
    while current_ndofs < MAX_NDOFS_EIGENVALUES:
        # Draw mesh
        ng.Draw(mesh, name="Mesh")
        time.sleep(TIME_DELAY)
        ngi.SnapShot(f"mesh_mixed_{iteration}.png")
        # Update
        assemble(a, m, f)
        iteration += 1
        current_ndofs = fes.ndof
        # Message
        print("EIGENVALUES strategy")
        print(f"\tCurrent ndofs: {current_ndofs}")
        print(f"\tIteration: {iteration}")
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
        # Here we *use* a greedy strategy to select the maximum error estimator
        # so eta_ev = (max_i(eta_i(T))_T, i.e., a element-indexed vector
        etas = []
        for i in range(right.m):
            eta, _ = error_estimator_ev(right.gridfun(i=i), evalues[i])
            etas.append(eta)
        etas = np.stack(etas, axis=0)
        eta_ev_monitored = np.max(etas, axis=1)
        eta_ev = np.max(etas, axis=0)
        max_eta_ev = np.max(eta_ev)
        # --- End Estimate ---
        # Mark
        mesh.ngmesh.Elements2D().NumPy()["refine"] = eta_ev > THETA * max_eta_ev
        # Store
        # Ndofs.append(fes.ndof)
        # Eta.append(max_eta_ls)
        # Eta_ev.append(eta_ev_monitored)
        # Evals.append(np.array(evalues))
        Ndofs_ev.append(current_ndofs)
        Eta_ev.append(max_eta_ev)
        Eta_ev_ls.append(eta_ev)
        Evals.append(np.array(evalues))
        # Refine
        if current_ndofs >= MAX_NDOFS_EIGENVALUES:
            break
        mesh.Refine()
        del right, left, projector, history, evalues, etas
        gc.collect()

    ng.Draw(sol, mesh, name="Landscape")
    ng.Draw(right.gridfun(i=None), mesh, name="evs")
    input("\tGet pictures! Then press Enter to continue")

    # --- Dataframes ---
    Ndofs = Ndofs_ls + Ndofs_ev
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
    df.to_csv("eigenvalues_mixed.csv", index=False)

    # Store landscape error estimators
    df = pd.DataFrame(
        {
            "Ndofs": Ndofs_ls,
            "Eta": Eta_ls,
        }
    )
    df.to_csv("error_estimator_landscape_mixed.csv", index=False)

    # Store eigenvalues error estimators
    df = pd.DataFrame(
        {
            "Ndofs": Ndofs_ev,
            "Eta": Eta_ev,
        }
    )
    df.to_csv("error_estimator_eigenvalues_mixed.csv", index=False)

    # Store post-asympotic error estimators
    res_ndofs = Ndofs[Ndofs > 2.0 * 1e4]
    slice_index = list(Ndofs).index(res_ndofs[0])
    res_evs = Evals[slice_index:]
    error = np.array(
        [np.abs(res_evs[i] - REF_EIGS) for i in range(1, len(res_ndofs))]
    )
    # error = np.array(
    #     [np.abs(res_evs[i] - res_evs[-1]) for i in range(0, len(res_ndofs) - 1)]
    # )

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
    df.to_csv("error_eigenvalues_mixed.csv", index=False)


if __name__ == "__main__":
    adaptivity_by_landscape()
    adaptivity_by_eigenvalues()
    # adaptivity_by_mixed_strategy()
