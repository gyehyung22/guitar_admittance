"""
fenics_modal_admittance.py
--------------------------
Modal eigensolve backend for guitar bridge admittance Y(omega).

This is an ADDITIVE backend.  The full harmonic solver in fenics_admittance.py
(solve_harmonic_petsc) is left completely untouched; this module reuses its
mesh loading, orthotropic material tensor, K/M assembly, weak-spring sizing and
bridge-DOF helpers, and adds:

  1. SLEPc generalized eigensolve            K phi_n = lambda_n M phi_n
     (Krylov-Schur, GHEP, shift-and-invert, MUMPS LU).
  2. Mass-normalized eigenvectors            phi_n^T M phi_n = 1   (GHEP B-norm).
  3. Rayleigh-equivalent modal damping        zeta_n = alpha/(2 w_n) + beta w_n/2
     (constant zeta is available as an option, but Rayleigh is the default so the
      comparison against the harmonic solver is apples-to-apples).
  4. Modal-superposition driving-point admittance

         Y_b(w) = j w * sum_n  phi_n(b)^2 / (w_n^2 - w^2 + j 2 zeta_n w_n w)

Because K and M do NOT depend on the bridge location, a single eigensolve
reconstructs the admittance for an arbitrary list of bridge points (bridge-point
batching).  Rigid-body / weak-spring modes (f_n < freq_min) are excluded from the
reconstruction.

Output format
-------------
For a single bridge point the saved admittance.npz keeps the SAME keys as the
harmonic path (frequencies, admittance) so existing consumers (dataset_gen,
run_pipeline, validate_admittance) keep working.  Extra modal data is saved
alongside (and, with save_modes, in a separate modes_*.npz).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Solid structural-modal solver revision — bumped when the eigenbasis handling
# changes (recorded in metadata so a dataset can be traced to the exact solver).
SOLID_MODAL_SOLVER_REVISION = "structural-modal-real-basis-v2"

# Reuse everything we can from the harmonic module (no duplication, no edits there).
from fenics_admittance import (
    engineering_to_C,
    load_mesh,
    assemble_KM_petsc,
    weak_spring_for_rigid_freq,
    count_admittance_peaks,
    _RIGID_FREQ_HZ,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception as _mpl_err:           # pragma: no cover - plotting is optional
    _MPL_OK = False
    print(f"[warn] matplotlib unavailable: {_mpl_err}")


# ---------------------------------------------------------------------------
# Bridge-node resolution (nearest-node, multi-point)
# ---------------------------------------------------------------------------

def find_bridge_nodes(dolfin_mesh, bridge_points_xyz) -> list[dict]:
    """Snap a list of requested bridge coordinates [mm] to nearest mesh nodes.

    Parameters
    ----------
    dolfin_mesh        : dolfinx mesh (coordinates in mm at call time)
    bridge_points_xyz  : iterable of (x, y, z) in mm

    Returns
    -------
    list of dicts, one per requested point:
        node_idx, bridge_requested_xyz, bridge_snapped_xyz, snap_distance_mm
    """
    coords = dolfin_mesh.geometry.x          # (N, 3) in mm
    out = []
    for p in bridge_points_xyz:
        req = np.asarray(p, dtype=float)
        d2 = np.sum((coords - req) ** 2, axis=1)
        idx = int(np.argmin(d2))
        snapped = coords[idx].astype(float)
        out.append({
            "node_idx": idx,
            "bridge_requested_xyz": req.tolist(),
            "bridge_snapped_xyz": snapped.tolist(),
            "snap_distance_mm": float(np.sqrt(d2[idx])),
        })
    return out


# ---------------------------------------------------------------------------
# Generalized eigensolve  K phi = lambda M phi
# ---------------------------------------------------------------------------

def _petsc_to_scipy(A):
    """PETSc Mat -> scipy CSR (real part taken by the realifier)."""
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())


def solve_modal_eigen(
    K_petsc,
    M_petsc,
    dof_indices: list[int],
    n_modes: int = 400,
    sigma_hz: float = 1.0,
    tol: float = 1e-9,
    max_it: int = 0,
) -> dict:
    """Solve the lowest `n_modes` of K phi = lambda M phi with SLEPc.

    The eigenvectors are B(=M)-orthonormalized by SLEPc for GHEP, i.e.
    phi_n^T M phi_n = 1 (mass normalization) — exactly what the modal
    superposition formula requires.

    Parameters
    ----------
    K_petsc, M_petsc : assembled PETSc matrices (K already includes the weak
                       spring on its diagonal — same as the harmonic path).
    dof_indices      : global DOF indices whose modal amplitudes phi_n(dof) we
                       keep (the bridge Z-DOFs).  Only these are extracted, so
                       memory stays small regardless of n_modes.
    n_modes          : number of (lowest) eigenpairs requested.
    sigma_hz         : shift-and-invert target frequency [Hz]; lowest modes are
                       found around here.  Use the rigid-body frequency so the
                       solver locks onto the bottom of the spectrum.

    Returns
    -------
    dict with:
        omega_n      : (n_conv,) angular eigenfrequencies [rad/s] (>=0)
        freqs_hz     : (n_conv,) eigenfrequencies [Hz]
        phi_at_dofs  : (n_dof_kept, n_conv) modal amplitudes at dof_indices
        n_conv       : number of converged eigenpairs
        eig_time     : wall time of eps.solve() [s]
    """
    from slepc4py import SLEPc

    eps = SLEPc.EPS().create()
    eps.setOperators(K_petsc, M_petsc)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_modes)
    if tol and tol > 0:
        eps.setTolerances(tol=tol, max_it=(max_it or SLEPc.DECIDE))

    sigma = (2.0 * np.pi * sigma_hz) ** 2
    eps.setTarget(sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    for solver in ("mumps", "superlu_dist", "superlu"):
        try:
            pc.setFactorSolverType(solver)
            break
        except Exception:
            pass

    eps.setFromOptions()
    print(f"  SLEPc EPS: Krylov-Schur / GHEP / shift-invert "
          f"(target {sigma_hz:.3g} Hz), nev={n_modes}")

    # Everything from eps.solve() on is guarded so the EPS + work Vec are ALWAYS
    # released, even if realification / residual checks raise (otherwise a failed
    # case would leak the SLEPc factorization and its MUMPS workspace).
    vr = None
    dof_arr = np.asarray(dof_indices, dtype=int)
    try:
        t0 = time.time()
        eps.solve()
        eig_time = time.time() - t0

        n_conv = eps.getConverged()
        print(f"  Converged eigenpairs: {n_conv}  ({eig_time:.1f} s)")

        # STREAMING realification (bounded memory).  In a complex PETSc/SLEPc build
        # a real eigenmode r comes back as v = e^{iθ} r; taking Re(v)/sqrt(vᴴMv) (the
        # old code) scales participation by cos²θ — the phase bug fixed in the
        # modal-coupled solver.  We reuse the shared degeneracy-safe realifier, but
        # process ONE eigenvalue cluster at a time and keep only the bridge-DOF rows,
        # so we never materialise all n_modes full vectors (that was
        # ~n_modes × n_dof complex = >1GB on a large solid mesh).
        lam_raw = [eps.getEigenvalue(i).real for i in range(n_conv)]   # scalars only
        if not lam_raw:
            return {"omega_n": np.zeros(0), "freqs_hz": np.zeros(0),
                    "phi_at_dofs": np.zeros((len(dof_arr), 0)), "n_conv": 0,
                    "eig_time": float(eig_time),
                    "solver_revision": SOLID_MODAL_SOLVER_REVISION,
                    "eigenbasis_diagnostics": {},
                    "timing": {"eigensolve_s": float(eig_time), "convert_s": 0.0,
                               "realify_s": 0.0, "eigenbasis_total_s": float(eig_time)}}

        t_conv = time.time()
        K_sp = _petsc_to_scipy(K_petsc)
        M_sp = _petsc_to_scipy(M_petsc)
        convert_time = time.time() - t_conv

        vr = K_petsc.createVecRight()

        def _fetch(i):
            eps.getEigenpair(int(i), vr)
            return vr.getArray().copy()           # one cluster's vectors at a time

        from modal_eigenbasis import (realify_bridge_participation,
                                       DEFAULT_RESIDUAL_TOL)
        t_re = time.time()
        lam, phi_at_dofs, diag = realify_bridge_participation(
            np.asarray(lam_raw, float), _fetch, M_sp, K_sp, dof_arr)
        realify_time = time.time() - t_re
    finally:
        if vr is not None:
            vr.destroy()
        eps.destroy()

    order = np.argsort(lam)
    lam = np.maximum(lam[order], 0.0)
    phi_at_dofs = phi_at_dofs[:, order]

    res = diag.get("max_eigen_residual")
    print(f"  [solid-modal] {phi_at_dofs.shape[1]} real modes from {n_conv} pairs, "
          f"max eigen residual={res}, max ||Im||/||Re||_M="
          f"{diag.get('max_imag_to_real_mnorm', 0.0):.2e} "
          f"(convert {convert_time:.1f}s, realify {realify_time:.1f}s)")
    if res is not None and res > DEFAULT_RESIDUAL_TOL:
        raise RuntimeError(f"[solid-modal] eigen residual too large ({res:.2e} > "
                           f"{DEFAULT_RESIDUAL_TOL:.1e}) after realification; "
                           "basis is unreliable.")

    omega_n = np.sqrt(lam)
    total = eig_time + convert_time + realify_time
    timing = {"eigensolve_s": float(eig_time), "convert_s": float(convert_time),
              "realify_s": float(realify_time), "eigenbasis_total_s": float(total),
              "total_s": float(total)}
    return {
        "omega_n": omega_n,
        "freqs_hz": omega_n / (2.0 * np.pi),
        "phi_at_dofs": phi_at_dofs,
        "n_conv": int(n_conv),
        "eig_time": float(eig_time),
        "timing": timing,
        "solver_revision": SOLID_MODAL_SOLVER_REVISION,
        "eigenbasis_diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Modal damping
# ---------------------------------------------------------------------------

def rayleigh_modal_damping(omega_n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Rayleigh-equivalent modal damping ratio zeta_n = alpha/(2 w_n) + beta w_n/2.

    This is exactly the modal damping implied by the Rayleigh damping matrix
    C = alpha M + beta K used in the harmonic solver, so the two methods see the
    same physical damping.
    """
    omega_n = np.asarray(omega_n, dtype=float)
    safe = np.where(omega_n > 0, omega_n, np.inf)
    return alpha / (2.0 * safe) + beta * omega_n / 2.0


def modal_damping(omega_n: np.ndarray, damping: str, alpha: float, beta: float,
                  zeta_const: float) -> np.ndarray:
    """Per-mode damping ratios for the chosen model."""
    if damping == "constant":
        return np.full_like(np.asarray(omega_n, dtype=float), float(zeta_const))
    if damping == "rayleigh":
        return rayleigh_modal_damping(omega_n, alpha, beta)
    raise ValueError(f"Unknown damping model '{damping}' (use 'rayleigh' or 'constant')")


# ---------------------------------------------------------------------------
# Modal superposition reconstruction
# ---------------------------------------------------------------------------

def reconstruct_admittance(freqs: np.ndarray, omega_n: np.ndarray,
                           phi_b: np.ndarray, zeta_n: np.ndarray) -> np.ndarray:
    """Driving-point admittance from mass-normalized modes (one bridge point).

        Y_b(w) = j w * sum_n phi_n(b)^2 / (w_n^2 - w^2 + j 2 zeta_n w_n w)

    Parameters
    ----------
    freqs   : (F,) target frequencies [Hz]
    omega_n : (N,) modal angular frequencies [rad/s]
    phi_b   : (N,) modal amplitudes at the bridge DOF (mass-normalized)
    zeta_n  : (N,) modal damping ratios

    Returns
    -------
    (F,) complex admittance [m/s/N]
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)        # (F,)
    wn = np.asarray(omega_n, dtype=float)                       # (N,)
    phi_b = np.asarray(phi_b, dtype=float)
    zeta_n = np.asarray(zeta_n, dtype=float)

    # (F, N) denominator; done in chunks-free vectorized form.
    denom = (wn[None, :] ** 2 - omega[:, None] ** 2
             + 1j * 2.0 * zeta_n[None, :] * wn[None, :] * omega[:, None])
    terms = (phi_b[None, :] ** 2) / denom
    Y = 1j * omega * terms.sum(axis=1)
    return Y


def reconstruct_modal_model(model_path, freqs: np.ndarray) -> np.ndarray:
    """Reconstruct every bridge mobility from a persisted compact modal model."""
    with np.load(model_path, allow_pickle=False) as model:
        if str(np.asarray(model["schema_version"]).item()) != "structural-modal-model-v1":
            raise ValueError("unsupported structural modal-model schema")
        omega_n = np.asarray(model["omega_n_rad_s"], float)
        residues = np.asarray(model["bridge_residue_inv_kg"], float)
        zeta = np.asarray(model["zeta"], float)
        residual = (np.asarray(model["residual_compliance_m_per_n"], float)
                    if "residual_compliance_m_per_n" in model.files
                    else np.zeros(residues.shape[0], float))
    freqs = np.asarray(freqs, float)
    omega = 2.0 * np.pi * freqs
    denom = (omega_n[None, :] ** 2 - omega[:, None] ** 2
             + 1j * 2.0 * zeta[None, :] * omega_n[None, :] * omega[:, None])
    response = 1j * omega[None, :] * np.sum(
        residues[:, None, :] / denom[None, :, :], axis=2)
    response += 1j * omega[None, :] * residual[:, None]
    return response


def residual_admittance(freqs: np.ndarray, H_static: float,
                        omega_n: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """High-mode truncation correction (mode-acceleration / residual flexibility).

    EXPERIMENTAL / KNOWN-BROKEN for the current free-free setup — kept for
    ablation only, NOT used by default (residual_flexibility=False).

        R   = H_static_full - sum_{n in basis} phi_n(b)^2 / w_n^2
        Y_residual(w) = i w R

    Why it is broken here: `H_static = e_bᵀ K_ws⁻¹ e_b` is dominated by the huge
    rigid-body compliance of the ~1 Hz weak-spring modes (~1e-1 m/N), while the
    true elastic residual is ~1e-8.  R is then a catastrophic-cancellation
    difference of two nearly-equal large numbers, so tiny rigid-mode inaccuracies
    leave R at rigid scale and iωR explodes at kHz (observed +80 dB).

    REQUIRED REDESIGN (future): do NOT use K_ws⁻¹.  Compute an ELASTIC-ONLY
    static residual with the rigid-body / nullspace modes DEFLATED, e.g.
        H_elastic = e_bᵀ (P K P)⁺ e_b ,  P = I - M Φ_r Φ_rᵀ  (Φ_r = rigid modes)
    and sum only over retained ELASTIC modes.  Both terms are then small and the
    subtraction is well-conditioned.  This needs the full rigid eigenvectors
    (return them from solve_modal_eigen), not just bridge-DOF amplitudes.
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    wn = np.asarray(omega_n, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    modal_static = np.sum(phi_b ** 2 / np.where(wn > 0, wn, np.inf) ** 2)
    R = float(H_static) - float(modal_static)
    return 1j * omega * R


def solve_static_compliance(K_ws_petsc, dof_list) -> np.ndarray:
    """Driving-point static compliance H_static[b] = e_bᵀ K_ws⁻¹ e_b per bridge.

    One direct solve per bridge DOF using the SAME weak-spring stiffness K_ws the
    harmonic solver uses (so the rigid-body part is consistent).  MUMPS LU is
    factorized once and reused across bridges.
    """
    from petsc4py import PETSc

    ksp = PETSc.KSP().create()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    for solver in ("mumps", "superlu_dist", "superlu"):
        try:
            pc.setFactorSolverType(solver); break
        except Exception:
            pass
    ksp.setOperators(K_ws_petsc)

    H = []
    for dof in dof_list:
        e = K_ws_petsc.createVecRight(); e.zeroEntries()
        e.setValue(dof, 1.0); e.assemblyBegin(); e.assemblyEnd()
        x = K_ws_petsc.createVecRight()
        ksp.solve(e, x)
        H.append(float(x.getValue(dof).real))
        e.destroy(); x.destroy()
    ksp.destroy()
    return np.array(H)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_admittance_modal(
    msh_path,
    material: dict,
    bridge_coords,
    freq_min: float = 20.0,
    freq_max: float = 5000.0,
    freq_points: int = 500,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 5e-6,
    output_dir="results/",
    force_n: float = 1.0,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
    modal_fmax: float = 7500.0,
    n_modes: int = 400,
    damping: str = "rayleigh",
    zeta_const: float = 0.01,
    bridge_points=None,
    save_modes: bool = False,
    include_rigid_modes: bool = True,
    residual_flexibility: bool = False,
    freqs=None,
    bulk: bool = False,
) -> dict:
    """Compute bridge admittance via modal eigensolve + superposition.

    Parameters mirror fenics_admittance.compute_admittance where they overlap.
    Extra parameters:

    modal_fmax   : compute/keep modes up to this frequency [Hz] (>= freq_max so
                   out-of-band modal tails inside the band are captured).
    n_modes      : number of lowest eigenpairs requested from SLEPc.
    damping      : 'rayleigh' (default, equivalent to harmonic C=aM+bK) or
                   'constant' (uses zeta_const for every mode).
    zeta_const   : constant modal damping ratio (only if damping='constant').
    bridge_points: optional list of (x,y,z) [mm].  If given, the SINGLE eigensolve
                   is reused to reconstruct admittance at every point (batching).
                   If None, the single `bridge_coords` point is used.
    save_modes   : also write modes_*.npz (eigenfreqs, omegas, participation).

    Returns
    -------
    dict with:
        freqs              : (F,) [Hz]
        Y                  : (F,) complex admittance for the FIRST bridge point
        Y_list             : list of (F,) complex, one per bridge point
        bridge_meta        : list of snap-metadata dicts (per bridge point)
        modal_freqs_hz     : (N_band,) eigenfrequencies used in reconstruction
        n_modes_retained   : int
        modal_fmax         : float
        timing             : dict with mesh/assembly/eigensolve/reconstruct [s]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This solver (and the bulk dataset worker that drives it) is serial: the
    # streaming realifier fetches whole eigenvectors on rank 0 only.  Fail loud on
    # a multi-rank launch instead of silently producing a rank-local partial basis.
    from mpi4py import MPI
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            f"compute_admittance_modal is serial (MPI size must be 1, got "
            f"{MPI.COMM_WORLD.size}); launch the per-case worker without mpirun.")

    # Resolve the bridge point list (first point doubles as the load_mesh nominal).
    if bridge_points is None:
        pts = [np.asarray(bridge_coords, dtype=float)]
    else:
        pts = [np.asarray(p, dtype=float) for p in bridge_points]
    if not pts:
        raise ValueError("No bridge points supplied")

    alpha = material.get("alpha", rayleigh_alpha)
    beta = material.get("beta", rayleigh_beta)

    timing: dict[str, float] = {}

    # 1. Mesh (tag=100 gives the exact embedded node for pts[0]) ----------------
    t0 = time.time()
    dolfin_mesh, bridge_idx0 = load_mesh(Path(msh_path), pts[0])
    timing["mesh"] = time.time() - t0

    # Resolve all bridge nodes (nearest-node, with snap metadata).
    bridge_meta = find_bridge_nodes(dolfin_mesh, pts)
    # Trust the tag=100 embedded node for the first point.
    bridge_meta[0]["node_idx"] = bridge_idx0
    bridge_meta[0]["bridge_snapped_xyz"] = dolfin_mesh.geometry.x[bridge_idx0].astype(float).tolist()

    # 2. Assembly ---------------------------------------------------------------
    C_pa = engineering_to_C(material)
    print(f"\nC diagonal (GPa): {np.diag(C_pa) / 1e9}")
    t0 = time.time()
    K_petsc, M_petsc, V = assemble_KM_petsc(dolfin_mesh, C_pa, material["density"])
    timing["assembly"] = time.time() - t0

    bs = V.dofmap.index_map_bs          # 3
    dof_list = [m["node_idx"] * bs + 2 for m in bridge_meta]    # Z DOF per bridge

    # Weak spring on K diagonal (same mesh-independent sizing as harmonic path).
    weak_k = weak_spring_for_rigid_freq(M_petsc, rigid_freq_hz)
    kd = K_petsc.getDiagonal()
    kd.array += weak_k
    K_petsc.setDiagonal(kd)
    kd.destroy()
    K_petsc.assemble()

    # 3. Eigensolve -------------------------------------------------------------
    # K/M are released in `finally` so a failing eigensolve/realification/static
    # solve never leaks the assembled operators (the user's explicit requirement).
    print(f"\nModal eigensolve: up to {n_modes} modes, target modal_fmax={modal_fmax} Hz")
    H_static = None
    try:
        eig = solve_modal_eigen(K_petsc, M_petsc, dof_list,
                                n_modes=n_modes, sigma_hz=rigid_freq_hz)
        timing["eigensolve"] = eig["eig_time"]
        timing["eigenbasis"] = eig.get("timing", {})

        # Static driving-point compliance (for residual flexibility) — solved with
        # the SAME weak-spring K before it is destroyed.
        if residual_flexibility:
            t0 = time.time()
            H_static = solve_static_compliance(K_petsc, dof_list)
            timing["static_solve"] = time.time() - t0
    finally:
        K_petsc.destroy()
        M_petsc.destroy()

    omega_all = eig["omega_n"]
    freqs_all = eig["freqs_hz"]
    phi_all = eig["phi_at_dofs"]          # (n_bridge, n_modes_conv)

    # Coverage sanity: warn if we did not reach modal_fmax.
    eig_freq_max_hz = float(freqs_all.max()) if len(freqs_all) else 0.0
    # A solved eigenvalue must lie strictly beyond the retention boundary.  Merely
    # landing on it cannot prove that a degenerate boundary eigenspace is complete.
    coverage_boundary_rtol = 1e-6
    coverage_ok = bool(
        eig_freq_max_hz > float(modal_fmax) * (1.0 + coverage_boundary_rtol))
    if len(freqs_all):
        print(f"  Eigenfreq range: {freqs_all.min():.1f} - {eig_freq_max_hz:.1f} Hz "
              f"({len(freqs_all)} modes)")
        if not coverage_ok:
            print(f"  WARNING: highest converged mode {eig_freq_max_hz:.0f} Hz < "
                  f"modal_fmax {modal_fmax:.0f} Hz — increase n_modes for full "
                  f"coverage (reconstruction is truncated).")
    else:
        print("  WARNING: no positive eigenvalues found.")

    # 4. Reconstruction ---------------------------------------------------------
    # Keep modes up to modal_fmax.  By default the rigid / weak-spring modes
    # (f_n < freq_min) ARE included: for a free-free structure their 1/ω inertial
    # tail is the dominant low-frequency mobility baseline that the full harmonic
    # solver retains (dropping them collapses the 20-800 Hz baseline, badly for
    # stiff/solid bodies).  Set include_rigid_modes=False to revert to the old
    # elastic-only behaviour for ablation.
    if include_rigid_modes:
        keep = (freqs_all <= modal_fmax)
        band_lo = 0.0
    else:
        keep = (freqs_all >= freq_min) & (freqs_all <= modal_fmax)
        band_lo = freq_min
    omega_band = omega_all[keep]
    phi_band = phi_all[:, keep]
    print(f"  Modes retained for reconstruction [{band_lo:.0f}, {modal_fmax:.0f}] Hz: "
          f"{keep.sum()}  (rigid included={include_rigid_modes}, "
          f"residual={residual_flexibility})")

    zeta_band = modal_damping(omega_band, damping, alpha, beta, zeta_const)

    # Frequency grid: explicit array (dataset common-grid contract) overrides the
    # default log-spaced sweep; must be strictly increasing.  Default preserved.
    if freqs is not None:
        freqs = np.asarray(freqs, dtype=float)
        if (freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs))
                or freqs[0] <= 0.0 or np.any(np.diff(freqs) <= 0)):
            raise ValueError(
                "explicit `freqs` must be finite, positive, and strictly increasing")
    else:
        freqs = np.geomspace(freq_min, freq_max, freq_points)
    omega_vec = 2.0 * np.pi * freqs
    t0 = time.time()
    Y_list = []
    residual_diag = []          # debug metadata per bridge (blowup auto-detect)
    for b, meta in enumerate(bridge_meta):
        # Admittance Y = j w u_z / F is a transfer function, independent of the
        # excitation magnitude (matches the harmonic solver, which divides u_z by
        # force_n).  So force_n is NOT applied here.
        Y_modal_b = reconstruct_admittance(freqs, omega_band, phi_band[b, :], zeta_band)
        Yb = Y_modal_b
        diag = None
        # High-mode truncation correction (EXPERIMENTAL — see residual_admittance).
        if residual_flexibility and H_static is not None:
            modal_static = float(np.sum(phi_band[b, :] ** 2
                                        / np.where(omega_band > 0, omega_band, np.inf) ** 2))
            R_b = float(H_static[b]) - modal_static
            Y_res = 1j * omega_vec * R_b
            max_iwR = float(np.max(np.abs(Y_res)))
            max_Ymodal = float(np.max(np.abs(Y_modal_b))) + 1e-30
            blowup = max_iwR > 5.0 * max_Ymodal      # residual dwarfs the response
            diag = {
                "H_static": float(H_static[b]),
                "modal_static": modal_static,
                "R": R_b,
                "max_abs_iwR": max_iwR,
                "max_abs_Y_modal": max_Ymodal,
                "residual_applied": bool(not blowup),
            }
            if blowup:
                print(f"  WARNING [residual blowup] bridge {b}: "
                      f"max|iwR|={max_iwR:.3e} >> max|Y_modal|={max_Ymodal:.3e} "
                      f"(R={R_b:.3e}). Residual NOT applied — free-free rigid/"
                      f"nullspace not deflated; see residual_admittance docstring.")
            else:
                Yb = Y_modal_b + Y_res
        residual_diag.append(diag)
        Y_list.append(Yb)
    timing["reconstruct"] = time.time() - t0

    Y = Y_list[0]

    residual_terms = np.asarray([
        (float(d["R"]) if d and d.get("residual_applied") else 0.0)
        for d in residual_diag
    ], float)
    omega = 2.0 * np.pi * np.asarray(freqs, float)
    denom = (omega_band[None, :] ** 2 - omega[:, None] ** 2
             + 1j * 2.0 * zeta_band[None, :] * omega_band[None, :] * omega[:, None])
    model_Y = 1j * omega[None, :] * np.sum(
        (phi_band ** 2)[:, None, :] / denom[None, :, :], axis=2)
    model_Y += 1j * omega[None, :] * residual_terms[:, None]
    Y_matrix = np.asarray(Y_list, complex)
    model_scale = max(float(np.max(np.abs(Y_matrix))), 1e-30)
    model_reconstruction_error = float(
        np.max(np.abs(model_Y - Y_matrix)) / model_scale)
    if not np.isfinite(model_reconstruction_error) or model_reconstruction_error > 1e-10:
        raise RuntimeError(
            "compact structural modal model does not reconstruct solver response "
            f"(relative error {model_reconstruction_error:.3e})")

    # Shared metadata (saved for BOTH single- and multi-bridge cases).
    n_modes_retained = int(keep.sum())
    coverage_meta = dict(
        modal_fmax=float(modal_fmax),
        n_modes_retained=n_modes_retained,
        n_eig_converged=int(eig["n_conv"]),
        eig_freq_max_hz=eig_freq_max_hz,
        coverage_ok=coverage_ok,
        damping=damping,
        zeta_const=float(zeta_const) if damping == "constant" else None,
        freq_min=float(freqs[0]),
        freq_max=float(freqs[-1]),
        freq_points=int(freqs.size),
        coverage_boundary_rtol=float(coverage_boundary_rtol),
        include_rigid_modes=bool(include_rigid_modes),
        residual_flexibility=bool(residual_flexibility),
        solver_revision=eig.get("solver_revision"),
        eigenbasis_diagnostics=eig.get("eigenbasis_diagnostics", {}),
        eigenbasis_timing=eig.get("timing", {}),
        model_reconstruction_max_rel_error=model_reconstruction_error,
    )

    # 5. Save (single-point file keeps harmonic-compatible keys; metadata added) -
    npz_path = output_dir / "admittance.npz"
    np.savez(str(npz_path),
             frequencies=freqs, admittance=Y,
             # --- snapped-bridge metadata (first point) ---
             bridge_requested_xyz=np.array(bridge_meta[0]["bridge_requested_xyz"]),
             bridge_snapped_xyz=np.array(bridge_meta[0]["bridge_snapped_xyz"]),
             snap_distance_mm=np.array(bridge_meta[0]["snap_distance_mm"]),
             # --- modal coverage metadata ---
             modal_fmax=np.array(modal_fmax),
             n_modes_retained=np.array(n_modes_retained),
             eig_freq_max_hz=np.array(eig_freq_max_hz),
             coverage_ok=np.array(coverage_ok))
    print(f"\nSaved: {npz_path}")

    # Human-readable metadata sidecar (always written).
    meta_json = {
        "bridge": [
            {**m, "dof_z": int(dof_list[i])} for i, m in enumerate(bridge_meta)
        ],
        "coverage": coverage_meta,
        "timing": timing,
        "n_peaks_first_bridge": int(count_admittance_peaks(freqs, Y)),
        # Residual-flexibility debug metadata (None unless residual_flexibility=True).
        # H_static, modal_static, R, max|iwR| per bridge — lets a QC step auto-detect
        # the free-free residual blowup instead of silently producing garbage.
        "residual_debug": residual_diag,
    }
    (output_dir / "admittance_meta.json").write_text(json.dumps(meta_json, indent=2))
    if not coverage_ok:
        print(f"  [coverage] WARNING saved: eig_freq_max={eig_freq_max_hz:.0f} Hz "
              f"< modal_fmax={modal_fmax:.0f} Hz")

    # Always write the batched contract, including for a one-bridge configuration.
    # The dataset worker consumes this filename uniformly.
    np.savez(str(output_dir / "admittance_modal_multi.npz"),
             frequencies=freqs,
             admittance=np.array(Y_list),
             bridge_requested=np.array([m["bridge_requested_xyz"] for m in bridge_meta]),
             bridge_snapped=np.array([m["bridge_snapped_xyz"] for m in bridge_meta]),
             snap_distance_mm=np.array([m["snap_distance_mm"] for m in bridge_meta]),
             modal_fmax=np.array(modal_fmax),
             n_modes_retained=np.array(n_modes_retained),
             eig_freq_max_hz=np.array(eig_freq_max_hz),
             coverage_ok=np.array(coverage_ok))
    print(f"Saved multi-bridge: {output_dir / 'admittance_modal_multi.npz'}")

    # Canonical compact raw model.  This is intentionally saved in bulk mode:
    # unlike full FEM eigenvectors it is tiny (modes x bridges) and permits an
    # arbitrary peak-resolved frequency sweep without repeating the eigensolve.
    np.savez_compressed(
        str(output_dir / "modal_model.npz"),
        schema_version=np.array("structural-modal-model-v1"),
        omega_n_rad_s=np.asarray(omega_band, float),
        bridge_residue_inv_kg=np.asarray(phi_band ** 2, float),
        zeta=np.asarray(zeta_band, float),
        residual_compliance_m_per_n=residual_terms,
        bridge_requested_xyz_mm=np.asarray(
            [m["bridge_requested_xyz"] for m in bridge_meta], float),
        bridge_snapped_xyz_mm=np.asarray(
            [m["bridge_snapped_xyz"] for m in bridge_meta], float),
        force_n=np.array(float(force_n)),
        # Persist the effective coefficients, including material-level overrides.
        rayleigh_alpha=np.array(float(alpha)),
        rayleigh_beta=np.array(float(beta)),
        time_convention=np.array("exp(+i omega t)"),
        response_units=np.array("m/s/N"),
    )

    if save_modes:
        modes_path = output_dir / "modes.npz"
        np.savez(str(modes_path),
                 modal_freqs_hz=freqs_all,
                 modal_omegas=omega_all,
                 modal_freqs_band_hz=freqs_all[keep],
                 bridge_participation=phi_band ** 2,    # (n_bridge, n_band) phi(b)^2
                 zeta_band=zeta_band,
                 n_modes_retained=int(keep.sum()),
                 modal_fmax=float(modal_fmax),
                 damping=damping)
        print(f"Saved modes: {modes_path}")

    # 6. Plot (first bridge point) — skipped for bulk dataset generation --------
    if _MPL_OK and not bulk:
        mag_db = 20.0 * np.log10(np.abs(Y) + 1e-30)
        n_peaks = count_admittance_peaks(freqs, Y)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(freqs, mag_db, linewidth=1.2, color="darkorange")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|Y(omega)| [dB re 1 m/s/N]")
        ax.set_title(f"Modal admittance | N_peaks={n_peaks}, "
                     f"modes={keep.sum()}, fmax={modal_fmax:.0f}, damping={damping}")
        ax.grid(True, which="both", alpha=0.4)
        ax.set_xlim(freq_min, freq_max)
        fig.tight_layout()
        plot_path = output_dir / "admittance.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"Plot saved: {plot_path}")

    return {
        "freqs": freqs,
        "Y": Y,
        "Y_list": Y_list,
        "bridge_meta": bridge_meta,
        "modal_freqs_hz": freqs_all[keep],
        "n_modes_retained": n_modes_retained,
        "modal_fmax": float(modal_fmax),
        "eig_freq_max_hz": eig_freq_max_hz,
        "coverage_ok": coverage_ok,
        "solver_revision": eig.get("solver_revision"),
        "eigenbasis_diagnostics": eig.get("eigenbasis_diagnostics", {}),
        "model_reconstruction_max_rel_error": model_reconstruction_error,
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# CLI (standalone modal run; mirrors fenics_admittance.py argument style)
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Modal eigensolve bridge admittance")
    parser.add_argument("--msh", type=str, required=True, help="Path to .msh file")
    parser.add_argument("--bridge", type=float, nargs=3, default=[0.0, -225.0, 100.0],
                        metavar=("X", "Y", "Z"), help="Bridge coords [mm]")
    parser.add_argument("--bridge-points-json", type=str, default=None,
                        help="JSON list of [x,y,z] bridge points (batched reconstruction)")
    parser.add_argument("--material", type=str, default=None)
    parser.add_argument("--material-json", type=str, default=None)
    parser.add_argument("--freq-min", type=float, default=20.0)
    parser.add_argument("--freq-max", type=float, default=5000.0)
    parser.add_argument("--freq-points", type=int, default=500)
    parser.add_argument("--rayleigh-alpha", type=float, default=0.0)
    parser.add_argument("--rayleigh-beta", type=float, default=5e-6)
    parser.add_argument("--modal-fmax", type=float, default=7500.0)
    parser.add_argument("--modal-nmodes", type=int, default=400)
    parser.add_argument("--damping", type=str, default="rayleigh",
                        choices=["rayleigh", "constant"])
    parser.add_argument("--zeta", type=float, default=0.01)
    parser.add_argument("--save-modes", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/")
    args = parser.parse_args()

    if args.material_json:
        mat = json.loads(args.material_json)
    elif args.material:
        sys.path.insert(0, str(Path(__file__).parent))
        from materials import get_material
        mat = get_material(args.material)
    else:
        from fenics_admittance import _MATERIAL
        mat = _MATERIAL

    bridge_points = None
    if args.bridge_points_json:
        p = Path(args.bridge_points_json)
        raw = p.read_text() if p.exists() else args.bridge_points_json
        bridge_points = json.loads(raw)

    compute_admittance_modal(
        msh_path=args.msh,
        material=mat,
        bridge_coords=tuple(args.bridge),
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_points=args.freq_points,
        rayleigh_alpha=args.rayleigh_alpha,
        rayleigh_beta=args.rayleigh_beta,
        output_dir=args.output_dir,
        modal_fmax=args.modal_fmax,
        n_modes=args.modal_nmodes,
        damping=args.damping,
        zeta_const=args.zeta,
        bridge_points=bridge_points,
        save_modes=args.save_modes,
    )


if __name__ == "__main__":
    _main()
