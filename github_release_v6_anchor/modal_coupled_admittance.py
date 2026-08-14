"""
modal_coupled_admittance.py
---------------------------
Reduced (modal) structure-air coupled bridge admittance — backend "E".

This is an ADDITIVE backend.  It does NOT replace the full FEM-FEM coupled solver
(`fenics_admittance_coupled.compute_coupled_admittance`, "D"), which stays the
reference.  E must be validated against D before being used for anything.

Formulation
-----------
E projects the SAME Everstine unsymmetric u-p block system that D solves
(docs/air_coupling_theory.md, exp(+iωt)) onto a reduced basis.  Nothing is
re-derived: the operators come from `fenics_admittance_coupled.build_coupled_blocks`,
so restriction, weak spring, G, port vector b and area S are bit-identical to D.

Full system (D):

    ⎡ Z_s(ω)      -Gᵀ                  ⎤ ⎡u⎤ = ⎡F⎤
    ⎣ -ρ0 ω² G     Z_a(ω) + κ(ω) b bᵀ  ⎦ ⎣p⎦   ⎣0⎦

    Z_s = K_s(1 + iωβ) - ω² M_s + iωα M_s      (K_s already has the weak spring)
    Z_a = K_a - (ω²/c²) M_a
    κ(ω) = iω ρ0 / (S² Z_h),   Z_h = R_rad(ω) + iω M_h

Reduced bases (as implemented; see AIR_COUPLING_NOTES.md "Phase 2")
------------------------------------------------------------------
ACOUSTIC (Ψ) — rigid-walled (all-Neumann) cavity basis; the soundhole port is NOT a
boundary condition of the basis, it enters as the same rank-1 κ b bᵀ term as in D:

    ψ_0 = 1/√V      ANALYTIC constant-pressure mode (cavity compliance), forced in;
                    β_0 = ψ_0ᵀ b = S/√V exactly.  K_a is singular, so an eigensolver
                    returns a polluted vector for this mode — we do not use it.
    ψ_j             elastic modes K_a ψ = (ω_a²/c²) M_a ψ, M_a-orthogonalized against ψ_0.

Without ψ_0 there is no cavity compliance and therefore no A0.

STRUCTURAL (Q) — normal modes, optionally enriched (default `basis='craig-bampton'`):

    Φ               K_s φ = ω_s² M_s φ (K_s includes the weak spring), φᵀM_sφ = 1
    + attachment    K_s⁻¹ e_bridge  and  K_s⁻¹ Gᵀψ_j, each deflated of its retained-
                    mode content (residual attachment modes), then M_s-orthonormalized.

The enrichment exists because the first ELASTIC structural mode is far above the A0
band: in 20-800 Hz the plates respond quasi-statically, and a truncated normal-mode
basis cannot represent that static compliance.  `basis='modal'` (normal modes only)
is kept as an ABLATION.

PROJECTION (u = Q q, p = Ψ a; rows left-multiplied by Qᵀ and Ψᵀ):

    ⎡ K_r(1+iωβ) - ω²M_r + iωα M_r   -G_rᵀ                          ⎤ ⎡q⎤ = ⎡Q_bᵀ F⎤
    ⎣ -ρ0 ω² G_r                      K_ar - (ω²/c²)M_ar + κ β βᵀ   ⎦ ⎣a⎦   ⎣  0   ⎦

    K_r = QᵀK_sQ,  M_r = QᵀM_sQ,  K_ar = ΨᵀK_aΨ,  M_ar = ΨᵀM_aΨ
    G_r = Ψᵀ G Q,  β = Ψᵀ b,  Q_b = Q[bridge_uz, :]

No diagonality is assumed (the enriched basis is not an eigenbasis).  Rayleigh damping
projects exactly: C_r = α M_r + β K_r.  The off-diagonal blocks are the projections of
the SAME -Gᵀ and -ρ0ω²G — the unsymmetry of the Everstine form is preserved
(-G_rᵀ vs -ρ0ω²G_r), it is NOT symmetrized.

Outputs match D:  Y = iω u_z(bridge)/F,  p_bar = (bᵀp)/S = (βᵀa)/S,  U_h = p_bar/Z_h.

Analytic sanity check (this is why the port sign/coefficient is right, not tuned):
with G = 0 and only the ω_a=0 mode (ψ_0 = 1/√V, β_0 = S/√V), the acoustic row is

    (-ω²/c²) a_0 + κ β_0² a_0 = 0,   κ = iωρ0/(S²·iωM_h) = 1/(S L_eff)
    -> ω² = c² S/(V L_eff)  ->  f = (c/2π)√(S/(V L_eff))  =  f_H  exactly.

So the reduced acoustic block + port reproduces the Helmholtz formula analytically.
`test_modal_coupled_math.py` asserts this (and the sign conventions) with no FEM.

Cost model
----------
Two eigensolves (structure ~n_u, air ~n_p) + a dense (m_u+m_p) solve per frequency.
The per-frequency cost drops from a ~96k sparse LU (D: ~85 s/freq) to a few-hundred
dense solve (ms).  The eigensolves are paid ONCE and are reused across every
frequency (and, via `bridge_points`, across bridge points).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fenics_admittance import _RIGID_FREQ_HZ
from acoustic_helmholtz import AIR_C, AIR_RHO0, radiation_resistance
import fenics_admittance_coupled as fac


SOLVER_REVISION = "modal-coupled-real-basis-v2"
RUN_STATUS_FILENAME = "run_status.json"


def _write_run_status(output_dir, status: str, **extra):
    payload = {
        "status": status,
        "solver_revision": SOLVER_REVISION,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    (Path(output_dir) / RUN_STATUS_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def _load_json_arg(value):
    """Load a CLI JSON argument that may be either a JSON string or a file path."""
    text = str(value).strip()
    if text.startswith(("[", "{")):
        return json.loads(text)
    try:
        p = Path(text)
        if p.exists():
            return json.loads(p.read_text())
    except OSError:
        pass
    return json.loads(text)


# ---------------------------------------------------------------------------
# Eigensolvers (full eigenvectors — needed to project G)
# ---------------------------------------------------------------------------

def _petsc_from_scipy(A):
    """scipy sparse -> PETSc AIJ.  Takes the real part: K/M/G are real operators
    (they are only complex-typed because dolfinx is built with complex scalars)."""
    from petsc4py import PETSc
    A = sp.csr_matrix(A)
    data = np.asarray(A.data)
    imag = float(np.max(np.abs(data.imag))) if np.iscomplexobj(data) else 0.0
    real = float(np.max(np.abs(data.real))) + 1e-300
    if imag / real > 1e-10:
        raise RuntimeError(f"Operator has a non-negligible imaginary part "
                           f"(max|imag|/max|real| = {imag/real:.2e}); the modal "
                           f"reduction assumes real symmetric K/M.")
    Am = PETSc.Mat().createAIJ(
        size=A.shape,
        csr=(A.indptr.astype(PETSc.IntType),
             A.indices.astype(PETSc.IntType),
             np.ascontiguousarray(data.real).astype(PETSc.ScalarType)))
    Am.assemble()
    return Am


from modal_eigenbasis import (  # shared, audited realification (see module)
    symmetry_residual as _symmetry_residual,
    eigenvalue_group_slices as _eigenvalue_group_slices,
    realify_eigen_group as _realify_eigen_group,
    realify_eigenbasis as _realify_eigenbasis,
    DEFAULT_RESIDUAL_TOL as _RESIDUAL_TOL,
    DEFAULT_GROUP_RTOL as _GROUP_RTOL,
)


def _mass_gram_blocked(Phi, M, block_cols: int = 16):
    """Form Phi.T M Phi without allocating a second full-size M@Phi array."""
    n_cols = Phi.shape[1]
    gram = np.empty((n_cols, n_cols), dtype=float)
    for start in range(0, n_cols, block_cols):
        stop = min(start + block_cols, n_cols)
        gram[:, start:stop] = Phi.T @ (M @ Phi[:, start:stop])
    return gram


def solve_eigen_full(K, M, n_modes: int, sigma: float, label: str,
                     tol: float = 1e-9):
    """Lowest `n_modes` of  K φ = λ M φ  with SLEPc (Krylov-Schur / GHEP /
    shift-invert at `sigma`), returning FULL M-orthonormalized eigenvectors.

    Returns (lam, Phi, eig_time, diagnostics):
        lam  : (m,) eigenvalues, ascending, clipped at 0 (rigid/constant modes
               can come back as tiny negatives; they are physical here).
        Phi  : (n, m) with Φᵀ M Φ = I enforced explicitly (not trusting SLEPc).

    `sigma` is in eigenvalue units (structure: ω², acoustic: ω²/c²).  A shift
    BELOW the bottom of the spectrum is required for the acoustic problem, whose
    lowest eigenvalue is exactly 0 (the constant-pressure cavity mode) — K_a is
    singular, so sigma = 0 would factorize a singular matrix.
    """
    from slepc4py import SLEPc
    total_t0 = time.time()

    sym_K, sym_M = _symmetry_residual(K), _symmetry_residual(M)
    if max(sym_K, sym_M) > 1e-8:
        raise RuntimeError(f"[{label}] operators are not symmetric "
                           f"(K {sym_K:.2e}, M {sym_M:.2e}); GHEP is invalid.")

    Kp, Mp = _petsc_from_scipy(K), _petsc_from_scipy(M)
    eps = SLEPc.EPS().create()
    eps.setOperators(Kp, Mp)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_modes)
    eps.setTolerances(tol=tol)
    eps.setTarget(sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP(); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu")
    for solver in ("mumps", "superlu_dist", "superlu"):
        try:
            pc.setFactorSolverType(solver)
            break
        except Exception:
            pass
    eps.setFromOptions()
    print(f"[modal-coupled] {label}: SLEPc GHEP shift-invert (sigma={sigma:.4g}), nev={n_modes}")

    setup_time = time.time() - total_t0
    solve_t0 = time.time()
    eps.solve()
    solve_time = time.time() - solve_t0
    n_conv = eps.getConverged()
    print(f"[modal-coupled] {label}: converged {n_conv}  ({solve_time:.1f} s)")
    if n_conv == 0:
        eps.destroy(); Kp.destroy(); Mp.destroy()
        raise RuntimeError(f"[{label}] SLEPc converged 0 eigenpairs.")

    K_real = sp.csr_matrix(K).real
    M_real = sp.csr_matrix(M).real
    operator_scales = (float(np.linalg.norm(K_real.data)),
                       float(np.linalg.norm(M_real.data)))
    vr = Kp.createVecRight()
    try:
        # First collect only eigenvalues.  Eigenvectors are then fetched and discarded
        # one spectral group at a time, avoiding an n_dof x n_mode complex Python copy.
        lam_raw = np.empty(n_conv, dtype=float)
        for idx in range(n_conv):
            lam_raw[idx] = eps.getEigenpair(idx, vr).real
        raw_order = np.argsort(lam_raw)
        lam_sorted = lam_raw[raw_order]
        groups = list(_eigenvalue_group_slices(lam_sorted, group_rtol=_GROUP_RTOL))

        Phi = np.empty((K.shape[0], n_conv), dtype=float)
        lam = np.empty(n_conv, dtype=float)
        out_col = 0
        group_diags = []
        for group in groups:
            raw_ids = raw_order[group]
            vecs_g = []
            for raw_idx in raw_ids:
                eps.getEigenpair(int(raw_idx), vr)
                vecs_g.append(vr.getArray().copy())
            lam_g, phi_g, diag_g = _realify_eigen_group(
                lam_sorted[group], vecs_g, M_real, K=K_real,
                operator_scales=operator_scales,
                drop_tol=1e-8, residual_tol=_RESIDUAL_TOL)
            n_group = group.stop - group.start
            if phi_g.shape[1] != n_group:
                raise RuntimeError(
                    f"[{label}] real mode count {phi_g.shape[1]} differs from "
                    f"complex pair count {n_group}")
            Phi[:, out_col:out_col + n_group] = phi_g
            lam[out_col:out_col + n_group] = lam_g
            out_col += n_group
            group_diags.append(diag_g)
    finally:
        vr.destroy(); eps.destroy(); Kp.destroy(); Mp.destroy()

    if out_col != n_conv:
        raise RuntimeError(f"[{label}] recovered {out_col} of {n_conv} requested real modes")
    if np.any(np.diff(lam) < -1e-10 * np.maximum(np.abs(lam[:-1]), 1.0)):
        raise RuntimeError(f"[{label}] realified eigenvalue groups are not ordered")
    lam = np.maximum(lam, 0.0)

    # Per-group realification makes each near-degenerate CLUSTER M-orthonormal, but
    # modes SPLIT across groups (dense thin top-plate spectra) can retain ~1e-3..1e-4
    # cross-group M-overlap.  A final GLOBAL Rayleigh-Ritz over the union of realified
    # modes restores M-orthonormality to ~machine precision AND keeps the eigenpairs
    # consistent (it re-diagonalises K in the SAME span; the residual can only shrink).
    # Cheap: it works on the n already-real columns, not the 2n complex candidates.
    Q, _kept = _m_orthonormalize(Phi, M_real)
    K_small = Q.T @ (K_real @ Q)
    K_small = 0.5 * (K_small + K_small.T)
    lam_rr, rot = np.linalg.eigh(K_small)
    order = np.argsort(lam_rr)
    Phi = Q @ rot[:, order]
    lam = np.maximum(lam_rr[order], 0.0)

    gram_t0 = time.time()
    gram = _mass_gram_blocked(Phi, M_real)
    dev = float(np.max(np.abs(gram - np.eye(gram.shape[0])))) if gram.size else 0.0
    gram_time = time.time() - gram_t0
    residual_values = [d["max_eigen_residual"] for d in group_diags
                       if d["max_eigen_residual"] is not None]
    max_residual = float(max(residual_values, default=0.0))
    im_ratio = float(max((d["max_imag_to_real_mnorm"] for d in group_diags), default=0.0))
    total_time = time.time() - total_t0
    postprocess_time = total_time - setup_time - solve_time - gram_time
    print(f"[modal-coupled] {label}: {Phi.shape[1]} real modes from {n_conv} pairs, "
          f"max|Phi^T M Phi-I|={dev:.2e}, max eig residual={max_residual:.2e}, "
          f"max ||Im||_M/||Re||_M={im_ratio:.2e}")
    if not np.isfinite(dev) or not np.isfinite(max_residual) or dev > 1e-6:
        raise RuntimeError(f"[{label}] recovered eigenbasis is not M-orthonormal "
                           f"(max|Phi^T M Phi-I|={dev:.2e}); degeneracy handling failed.")
    diagnostics = {
        "n_complex_pairs": int(n_conv),
        "n_real_modes": int(Phi.shape[1]),
        "n_eigenvalue_groups": int(len(group_diags)),
        "n_small_candidates": int(sum(d["n_small_candidates"] for d in group_diags)),
        "n_dependent_candidates": int(sum(d["n_dependent_candidates"] for d in group_diags)),
        "n_rank_capped_candidates": int(sum(d["n_rank_capped_candidates"] for d in group_diags)),
        "mass_orthonormality_max_dev": dev,
        "max_eigen_residual": max_residual,
        "max_imag_to_real_mnorm": im_ratio,
        "operator_setup_s": float(setup_time),
        "eigensolve_s": float(solve_time),
        "realification_and_extract_s": float(postprocess_time),
        "mass_gram_check_s": float(gram_time),
        "total_eigenbasis_s": float(total_time),
    }
    return lam, Phi, float(total_time), diagnostics


# ---------------------------------------------------------------------------
# Reduced coupled sweep (pure numpy — unit-testable without FEM)
# ---------------------------------------------------------------------------

def reduced_coupled_sweep(
    freqs: np.ndarray,
    omega_s: np.ndarray,       # (m_u,) structural modal angular freqs [rad/s]
    omega_a: np.ndarray,       # (m_p,) acoustic modal angular freqs [rad/s]
    G_m: np.ndarray,           # (m_p, m_u) = Psi^T G Phi
    beta_m: np.ndarray,        # (m_p,)     = Psi^T b
    phi_b: np.ndarray,         # (m_u,)     = Phi[bridge_uz, :]
    S: float,
    M_h: float,
    alpha: float,
    beta_ray: float,
    c: float = AIR_C,
    rho0: float = AIR_RHO0,
    force_n: float = 1.0,
) -> dict:
    """Solve the reduced Everstine block system over `freqs`.

    This is the projection of D's block system — same sign convention, same port
    term.  Pure numpy: no FEM, no PETSc, so it can be unit-tested locally against
    the analytic Helmholtz / SDOF limits (test_modal_coupled_math.py).
    """
    freqs = np.asarray(freqs, float)
    m_u, m_p = omega_s.size, omega_a.size
    Y = np.zeros(freqs.size, dtype=complex)
    p_bar = np.zeros(freqs.size, dtype=complex)
    U_h = np.zeros(freqs.size, dtype=complex)
    per_freq_s = np.zeros(freqs.size)

    ws2 = omega_s ** 2
    wa2 = omega_a ** 2
    rhs = np.zeros(m_u + m_p, dtype=complex)

    for i, f in enumerate(freqs):
        t_f = time.time()
        w = 2.0 * np.pi * f

        # structural diagonal: phi^T [K_s(1+iwb) - w^2 M_s + i w a M_s] phi
        d_s = ws2 * (1.0 + 1j * w * beta_ray) - w ** 2 + 1j * w * alpha
        # acoustic diagonal:   psi^T [K_a - (w^2/c^2) M_a] psi
        d_a = (wa2 - w ** 2) / c ** 2

        R = radiation_resistance(w, S, c, rho0)
        Z_h = R + 1j * w * M_h
        kappa = 1j * w * rho0 / (S ** 2 * Z_h)

        A = np.zeros((m_u + m_p, m_u + m_p), dtype=complex)
        A[:m_u, :m_u] = np.diag(d_s)
        A[:m_u, m_u:] = -G_m.T                       # -G^T  (projected)
        A[m_u:, :m_u] = (-rho0 * w ** 2) * G_m       # -rho0 w^2 G  (projected)
        A[m_u:, m_u:] = np.diag(d_a) + kappa * np.outer(beta_m, beta_m)

        rhs[:] = 0.0
        rhs[:m_u] = phi_b * force_n                  # Phi^T (F e_bridge)

        x = np.linalg.solve(A, rhs)
        q, a = x[:m_u], x[m_u:]

        u_b = phi_b @ q                              # u_z at the bridge
        Y[i] = 1j * w * u_b / force_n
        p_bar[i] = (beta_m @ a) / S
        U_h[i] = p_bar[i] / Z_h
        per_freq_s[i] = time.time() - t_f

    return {"freqs": freqs, "Y": Y, "p_bar": p_bar, "U_h": U_h,
            "per_freq_s": per_freq_s}


# ---------------------------------------------------------------------------
# GENERAL reduced sweep (no diagonal assumption)
#
# The enriched (Craig-Bampton) structural basis is NOT a set of eigenvectors, so
# Q^T K_s Q is a full dense matrix, not diag(omega^2).  Same on the acoustic side
# once the analytic constant-pressure mode is forced into the basis.  This routine
# takes the projected operators directly and is therefore the general form of
# `reduced_coupled_sweep` (which remains as the diagonal / pure-modal ablation).
# ---------------------------------------------------------------------------

def reduced_coupled_sweep_general(
    freqs: np.ndarray,
    K_r: np.ndarray,           # (m_u, m_u) = Q^T K_s Q
    M_r: np.ndarray,           # (m_u, m_u) = Q^T M_s Q
    Ka_r: np.ndarray,          # (m_p, m_p) = Psi^T K_a Psi
    Ma_r: np.ndarray,          # (m_p, m_p) = Psi^T M_a Psi
    G_r: np.ndarray,           # (m_p, m_u) = Psi^T G Q
    beta_m: np.ndarray,        # (m_p,)     = Psi^T b
    q_b: np.ndarray,           # (n_b, m_u) = Q[bridge_uz_dofs, :]
    S: float,
    M_h: float,
    alpha: float,
    beta_ray: float,
    c: float = AIR_C,
    rho0: float = AIR_RHO0,
    force_n: float = 1.0,
) -> dict:
    """Projection of D's block system onto ARBITRARY reduced bases (Q, Psi):

        [[ K_r(1+i w beta) - w^2 M_r + i w alpha M_r,  -G_r^T                   ],
         [ -rho0 w^2 G_r,                              Ka_r - (w^2/c^2) Ma_r
                                                        + kappa beta beta^T     ]]

    C_r = alpha M_r + beta K_r is the EXACT projection of C = alpha M_s + beta K_s
    (Rayleigh damping is basis-independent under projection), so no damping model is
    invented here.  Multi-bridge: q_b holds one row per bridge; the reduced matrix is
    bridge-independent, so all bridges are solved from one LU per frequency.
    """
    freqs = np.asarray(freqs, float)
    q_b = np.atleast_2d(q_b)
    n_b, m_u = q_b.shape
    m_p = beta_m.size
    Y = np.zeros((n_b, freqs.size), dtype=complex)
    p_bar = np.zeros((n_b, freqs.size), dtype=complex)
    U_h = np.zeros((n_b, freqs.size), dtype=complex)
    per_freq_s = np.zeros(freqs.size)
    port = np.outer(beta_m, beta_m)

    for i, f in enumerate(freqs):
        t_f = time.time()
        w = 2.0 * np.pi * f
        Zs_r = K_r * (1.0 + 1j * w * beta_ray) - (w ** 2) * M_r + (1j * w * alpha) * M_r
        Za_r = Ka_r - (w ** 2 / c ** 2) * Ma_r

        R = radiation_resistance(w, S, c, rho0)
        Z_h = R + 1j * w * M_h
        kappa = 1j * w * rho0 / (S ** 2 * Z_h)

        A = np.zeros((m_u + m_p, m_u + m_p), dtype=complex)
        A[:m_u, :m_u] = Zs_r
        A[:m_u, m_u:] = -G_r.T
        A[m_u:, :m_u] = (-rho0 * w ** 2) * G_r
        A[m_u:, m_u:] = Za_r + kappa * port

        RHS = np.zeros((m_u + m_p, n_b), dtype=complex)
        RHS[:m_u, :] = (q_b * force_n).T          # Q^T (F e_bridge) for each bridge
        X = np.linalg.solve(A, RHS)

        for ib in range(n_b):
            q, a = X[:m_u, ib], X[m_u:, ib]
            Y[ib, i] = 1j * w * (q_b[ib] @ q) / force_n
            p_bar[ib, i] = (beta_m @ a) / S
            U_h[ib, i] = p_bar[ib, i] / Z_h
        per_freq_s[i] = time.time() - t_f

    return {"freqs": freqs, "Y": Y, "p_bar": p_bar, "U_h": U_h,
            "per_freq_s": per_freq_s}


def reconstruct_reduced_model(model_path, freqs: np.ndarray) -> dict:
    """Resample a persisted air-coupled reduced model without any FEM objects."""
    with np.load(model_path, allow_pickle=False) as model:
        if str(np.asarray(model["schema_version"]).item()) != "air-coupled-reduced-model-v1":
            raise ValueError("unsupported air-coupled reduced-model schema")
        arrays = {key: np.asarray(model[key]).copy() for key in (
            "K_r", "M_r", "Ka_r", "Ma_r", "G_r", "beta_m", "q_b")}
        scalars = {key: float(np.asarray(model[key]).item()) for key in (
            "soundhole_area_m2", "soundhole_inertance_kg_m4",
            "air_speed_m_s", "air_density_kg_m3", "rayleigh_alpha",
            "rayleigh_beta", "force_n")}
    return reduced_coupled_sweep_general(
        np.asarray(freqs, float), arrays["K_r"], arrays["M_r"],
        arrays["Ka_r"], arrays["Ma_r"], arrays["G_r"], arrays["beta_m"],
        arrays["q_b"], scalars["soundhole_area_m2"],
        scalars["soundhole_inertance_kg_m4"], scalars["rayleigh_alpha"],
        scalars["rayleigh_beta"], c=scalars["air_speed_m_s"],
        rho0=scalars["air_density_kg_m3"], force_n=scalars["force_n"])


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def _static_solve(Ks_real, F, solver: str = "auto") -> np.ndarray:
    """Solve K_s X = F for the attachment/static vectors (multi-RHS).

    K_s is ~n_u x n_u (86k on the box mesh).  scipy's serial SuperLU (`splu`) runs out
    of memory at that size — it died on the server — while MUMPS factorizes the LARGER
    ~96k complex coupled block in ~1.6 s.  So the default is PETSc/MUMPS (the same
    factorization backend D uses), with scipy as a fallback for small problems / no
    PETSc (the local pure-numpy tests take that path).

    K_s is real SPD (weak spring), but the FEniCSx build is complex-scalar, so the
    PETSc path solves in complex and takes the real part; the imaginary residual is
    checked, not assumed.
    """
    F = np.asarray(F, float)
    if solver in ("auto", "petsc"):
        try:
            lu = fac._PetscLU(sp.csr_matrix(Ks_real).astype(complex))
            t0 = time.time()
            Xc = lu.solve(F.astype(complex))
            lu.destroy()
            imag = float(np.max(np.abs(Xc.imag)))
            real = float(np.max(np.abs(Xc.real))) + 1e-300
            if imag / real > 1e-8:
                raise RuntimeError(f"static solve returned a complex solution "
                                   f"(|imag|/|real| = {imag/real:.2e}); K_s is real SPD, "
                                   f"so this indicates a corrupted operator.")
            print(f"[modal-coupled] attachment static solve: PETSc/MUMPS, "
                  f"{F.shape[1]} RHS ({time.time() - t0:.1f}s)")
            return np.ascontiguousarray(Xc.real)
        except Exception as exc:
            if solver == "petsc":
                raise
            print(f"[modal-coupled] PETSc static solve unavailable ({exc}); "
                  f"falling back to scipy splu.")
    lu = spla.splu(sp.csc_matrix(Ks_real))
    return lu.solve(F)


def _m_orthonormalize(Q, M, tol: float = 1e-10):
    """Modified Gram-Schmidt in the M inner product.  Drops columns whose residual
    norm collapses (linearly dependent), returning (Q_orth, kept_indices).

    The M-products of the accepted columns are CACHED: the naive version recomputes
    `M @ v` inside the inner loop, i.e. O(k^2) sparse mat-vecs, which is unusable at
    FEM scale (hundreds of vectors x 86k dofs).  Here each new column costs 2 mat-vecs
    plus O(k) dot products.
    """
    cols, mcols, kept = [], [], []
    for j in range(Q.shape[1]):
        v = np.asarray(Q[:, j], float).copy()
        Mv = M @ v
        nrm0 = np.sqrt(max(float(v @ Mv), 0.0))
        if nrm0 <= 0.0 or not np.isfinite(nrm0):
            continue
        for _ in range(2):                       # twice for numerical stability
            for u, Mu in zip(cols, mcols):
                v -= float(Mu @ v) * u           # (u^T M v) u, M symmetric -> (M u)^T v
        Mv = M @ v
        nrm = np.sqrt(max(float(v @ Mv), 0.0))
        if nrm / nrm0 < tol:                     # dependent on what we already have
            continue
        cols.append(v / nrm)
        mcols.append(Mv / nrm)
        kept.append(j)
    if not cols:
        raise RuntimeError("M-orthonormalization dropped every basis vector.")
    return np.column_stack(cols), kept


def _m_orthonormalize_against(X, Phi, M, tol: float = 1e-10):
    """M-orthonormalize the columns of X against an ALREADY M-orthonormal Phi (and
    against each other).  Returns (X_orth, kept_indices_into_X).

    Phi comes from the eigensolver with Phi^T M Phi = I enforced, so re-orthogonalizing
    it against itself would be hundreds of wasted mat-vecs and a second 86k x 600 copy
    in memory.  Only the handful of attachment vectors need work here.
    """
    cols, mcols, kept = [], [], []
    for j in range(X.shape[1]):
        v = np.asarray(X[:, j], float).copy()
        Mv = M @ v
        nrm0 = np.sqrt(max(float(v @ Mv), 0.0))
        if nrm0 <= 0.0 or not np.isfinite(nrm0):
            continue
        for _ in range(2):
            v -= Phi @ (Phi.T @ (M @ v))         # project out the retained-mode space
            for u, Mu in zip(cols, mcols):
                v -= float(Mu @ v) * u
        Mv = M @ v
        nrm = np.sqrt(max(float(v @ Mv), 0.0))
        if nrm / nrm0 < tol:
            continue
        cols.append(v / nrm)
        mcols.append(Mv / nrm)
        kept.append(j)
    if not cols:
        return np.zeros((X.shape[0], 0)), []
    return np.column_stack(cols), kept


def _m_project_out(X, Phi, M, MPhi=None):
    """Remove span(Phi) from X in the M inner product, independent of scaling."""
    MPhi = M @ Phi if MPhi is None else MPhi
    gram = Phi.T @ MPhi
    gram = 0.5 * (gram + gram.T)
    coeff = np.linalg.solve(gram, MPhi.T @ X)
    return X - Phi @ coeff


def build_acoustic_basis(Ka, Ma, b, S, cavity_volume, n_modes: int,
                         acoustic_fmax: float, c: float = AIR_C,
                         port_attachment: bool = True,
                         attachment_solver: str = "auto") -> dict:
    """SLEPc eigensolve + `assemble_acoustic_reduced` (see there for the theory)."""
    t_total = time.time()
    n_p = Ka.shape[0]
    lam_a, Psi_e, t_eig, eig_diag = solve_eigen_full(
        Ka, Ma, n_modes=min(n_modes, n_p - 1),
        # negative shift: lambda_min = 0 exactly (K_a singular), so sigma = 0 would
        # factorize a singular matrix.
        sigma=-1.0 * (2.0 * np.pi * 50.0 / c) ** 2, label="acoustic")
    omega_a_e = c * np.sqrt(np.maximum(lam_a, 0.0))       # lam = omega^2 / c^2
    solved_freq_max = float(np.max(omega_a_e) / (2.0 * np.pi)) if omega_a_e.size else 0.0
    keep = omega_a_e <= 2.0 * np.pi * acoustic_fmax
    Psi_retained = np.ascontiguousarray(Psi_e[:, keep])
    omega_retained = omega_a_e[keep]
    del Psi_e
    out = assemble_acoustic_reduced(Ka, Ma, b, S, cavity_volume,
                                    Psi_retained, omega_retained,
                                    acoustic_fmax=np.inf,
                                    port_attachment=port_attachment,
                                    attachment_solver=attachment_solver,
                                    solved_freq_max_hz=solved_freq_max)
    out["eig_time"] = t_eig
    out["basis_prep_time"] = max(0.0, time.time() - t_total - t_eig)
    out["eigenbasis_diagnostics"] = eig_diag
    return out


def acoustic_port_attachment(Ka, Ma, b, cavity_volume, solver: str = "auto",
                             residual_tol: float = 1e-8):
    """Quasi-static pressure field driven by the soundhole port: the acoustic
    ATTACHMENT (residual) mode.  This is the acoustic analogue of the structural
    Craig-Bampton vectors, and it is what fixes A0.

    Why it is needed (measured on the box: A0 = 352 Hz with 60 acoustic modes,
    313 Hz with 120, vs 289 Hz for the full FEM): the pressure field near the hole
    is a strongly localized near-field that carries the interior added mass (the
    "interior end correction").  A rigid-walled modal basis converges to it very
    slowly, so A0 comes out too HIGH (too little added mass) and drifts down as
    modes are added.  But A0 lies far below the first cavity mode, so that field is
    quasi-STATIC — it is exactly K_a^{-1} b, and one vector captures it.

    K_a is singular (Neumann; nullspace = constants), so:
      * the load is made compatible:  f = b - (1^T b / V) M_a 1   (=> 1^T f = 0);
        the removed part is precisely the uniform compression that psi_0 already
        represents, so nothing is double-counted;
      * the constant is fixed by a bordered (Lagrange) system, which also enforces
        1^T M_a x = 0, i.e. x is M_a-orthogonal to psi_0 by construction.

            [ K_a    M_a 1 ] [x]   [f]
            [ 1^T M_a   0  ] [l] = [0]
    """
    Ka = sp.csr_matrix(Ka).real
    Ma = sp.csr_matrix(Ma).real
    n = Ka.shape[0]
    ones = np.ones(n)
    Ma1 = np.asarray(Ma @ ones).ravel()
    b = np.asarray(b).real.ravel()
    f = b - (float(ones @ b) / cavity_volume) * Ma1        # 1^T f = S - (S/V)*V = 0

    A = sp.bmat([[Ka, Ma1.reshape(-1, 1)],
                 [Ma1.reshape(1, -1), None]], format="csc")
    rhs = np.concatenate([f, [0.0]])

    # The bordered matrix is symmetric INDEFINITE (Lagrange multiplier), so only a
    # general LU is valid.  PETSc/MUMPS by default (the air block can be large on real
    # guitar meshes); scipy for small problems / no PETSc.  Never silently accept a bad
    # solve: the residual is checked below and raises.
    used = "scipy"
    if solver in ("auto", "petsc"):
        try:
            lu = fac._PetscLU(sp.csr_matrix(A).astype(complex))
            solc = lu.solve(rhs.astype(complex))
            lu.destroy()
            imag = float(np.max(np.abs(solc.imag)))
            real = float(np.max(np.abs(solc.real))) + 1e-300
            if imag / real > 1e-8:
                raise RuntimeError(f"complex solution for a real system "
                                   f"(|imag|/|real| = {imag/real:.2e})")
            sol = np.ascontiguousarray(solc.real)
            used = "petsc/mumps"
        except Exception as exc:
            if solver == "petsc":
                raise
            print(f"[modal-coupled] PETSc port-attachment solve unavailable ({exc}); "
                  f"falling back to scipy.")
            sol = spla.spsolve(A, rhs)
    else:
        sol = spla.spsolve(A, rhs)

    x = np.asarray(sol[:n], float)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("acoustic port attachment solve produced non-finite values "
                           "(bordered K_a system is singular?)")
    # FAIL LOUD: residual of the bordered system and the zero-mean (M_a-orthogonality
    # to psi_0) constraint.  A bad attachment vector would silently move A0.
    res = float(np.linalg.norm(Ka @ x + sol[n] * Ma1 - f) / (np.linalg.norm(f) + 1e-300))
    mean_dev = float(abs(Ma1 @ x) / (np.linalg.norm(Ma1) * np.linalg.norm(x) + 1e-300))
    if res > residual_tol or mean_dev > 1e-8:
        raise RuntimeError(
            f"acoustic port attachment solve failed its checks (solver={used}): "
            f"residual_rel={res:.2e} (tol {residual_tol:.0e}), "
            f"M_a-orthogonality to psi_0 = {mean_dev:.2e} (tol 1e-8). "
            f"Refusing to build a basis that would silently bias A0.")
    return x, {"residual_rel": res, "m_orthogonality_to_psi0": mean_dev, "solver": used}


def assemble_acoustic_reduced(Ka, Ma, b, S, cavity_volume, Psi_elastic, omega_a_elastic,
                              acoustic_fmax: float = np.inf,
                              port_attachment: bool = True,
                              attachment_solver: str = "auto",
                              solved_freq_max_hz: float | None = None) -> dict:
    """Acoustic reduced basis with the ANALYTIC constant-pressure mode forced in.

    Pure linear algebra (no SLEPc) so it is unit-testable locally.

    The rigid-walled (all-Neumann) cavity has an exact zero mode: p = const, i.e.
    psi_0 = 1/sqrt(V) after M_a-normalization (psi_0^T M_a psi_0 = (1/V) 1^T M_a 1 = 1).
    It carries the cavity COMPLIANCE, and its port coupling is exactly

        beta_0 = psi_0^T b = (1/sqrt(V)) * sum(b) = S / sqrt(V).

    An eigenvector computed for the (singular) K_a is only determined up to that
    nullspace and comes back polluted/rotated, which is what made beta_0 miss
    S/sqrt(V) by ~30 % in the first E run.  So the zero mode is NOT taken from the
    eigensolver: it is built analytically, the eigensolver's near-zero mode is
    dropped, and the remaining (elastic) modes are M_a-orthogonalized against psi_0.

    Ka_r / Ma_r are formed by explicit projection (no diagonality assumed).
    """
    n_p = Ka.shape[0]
    psi0 = np.ones(n_p) / np.sqrt(cavity_volume)

    omega_a_e = np.asarray(omega_a_elastic, float)
    f_a_e = omega_a_e / (2.0 * np.pi)
    f_a_solved_max = (float(solved_freq_max_hz) if solved_freq_max_hz is not None
                      else (float(f_a_e.max()) if f_a_e.size else 0.0))
    Psi_e = np.asarray(Psi_elastic, float)

    # Discard the eigensolver's own zero mode (it duplicates psi0, unreliably) and truncate.
    keep = (f_a_e > 1.0) & (f_a_e <= acoustic_fmax)
    Psi_e = Psi_e[:, keep]
    omega_a_e = omega_a_e[keep]
    f_a_retained_max = float(f_a_e[keep].max()) if np.any(keep) else 0.0

    Ma_real = sp.csr_matrix(Ma).real
    Ka_real = sp.csr_matrix(Ka).real

    # Port ATTACHMENT vector: the quasi-static near-hole field that carries the
    # interior added mass.  Placed right after psi_0, BEFORE the elastic modes, so
    # Gram-Schmidt never drops it in favour of a high-order mode.
    port_diag = {}
    cols = [psi0]
    if port_attachment:
        x_port, port_diag = acoustic_port_attachment(
            Ka_real, Ma_real, b, cavity_volume, solver=attachment_solver)
        cols.append(x_port)
    else:
        print("[modal-coupled] NOTE: acoustic port attachment is OFF - A0 will carry the "
              "acoustic modal-truncation bias (measured: 352 Hz @60 modes, 313 Hz @120 "
              "vs 289 Hz for the full FEM). |Y| is unaffected. Use "
              "--acoustic-attachment port.")
    if Psi_e.shape[1]:
        cols.append(Psi_e)
    Psi_raw = np.column_stack(cols)

    Psi, kept = _m_orthonormalize(Psi_raw, Ma_real)
    if 0 not in kept:
        raise RuntimeError("The analytic constant-pressure mode was dropped during "
                           "orthonormalization — cavity compliance would be lost.")
    n_port_att = int(1 in kept) if port_attachment else 0
    if port_attachment and not n_port_att:
        print("[modal-coupled] WARNING: the port attachment vector was dropped as "
              "linearly dependent on psi_0 — A0 will keep the modal-truncation bias.")

    Ka_r = Psi.T @ (Ka_real @ Psi)
    Ma_r = Psi.T @ (Ma_real @ Psi)
    beta_m = Psi.T @ np.asarray(b).real
    ma_r_dev = float(np.max(np.abs(Ma_r - np.eye(Ma_r.shape[0]))))
    ma_eigs = np.linalg.eigvalsh(0.5 * (Ma_r + Ma_r.T))
    ma_r_cond = float(ma_eigs.max() / max(ma_eigs.min(), 1e-300))
    if (not np.isfinite(ma_r_dev) or not np.all(np.isfinite(ma_eigs))
            or ma_r_dev > 1e-6 or ma_eigs.min() < 1e-8):
        raise RuntimeError(
            f"acoustic reduced basis is not M-orthonormal "
            f"(max deviation {ma_r_dev:.2e}, min eigenvalue {ma_eigs.min():.2e})")

    beta0_expected = S / np.sqrt(cavity_volume)
    beta0_rel = abs(abs(beta_m[0]) - beta0_expected) / beta0_expected
    # psi_0 is in the null space of K_a: the constant pressure has zero gradient.
    ka00 = float(abs(Ka_r[0, 0])) / (float(np.abs(Ka_r).max()) + 1e-300)
    print(f"[modal-coupled] acoustic basis: 1 analytic zero mode + {n_port_att} port "
          f"attachment + {Psi.shape[1] - 1 - n_port_att} elastic modes "
          f"(retained to {f_a_retained_max:.0f} Hz; eigensolve to {f_a_solved_max:.0f} Hz)")
    if port_diag:
        print(f"[modal-coupled] port attachment: solve residual "
              f"{port_diag['residual_rel']:.2e}, M_a-orthogonality to psi_0 "
              f"{port_diag['m_orthogonality_to_psi0']:.2e}")
    print(f"[modal-coupled] port projection check: |beta_0|={abs(beta_m[0]):.6e} vs "
          f"S/sqrt(V)={beta0_expected:.6e}  (rel {beta0_rel*100:.4f}%)")
    print(f"[modal-coupled] zero-mode stiffness check: |Ka_r[0,0]|/max|Ka_r| = {ka00:.2e} "
          f"(must be ~0: constant pressure has no gradient energy)")
    if beta0_rel > 1e-6:
        print("[modal-coupled] WARNING: beta_0 is not S/sqrt(V) to 1e-6 - check that "
              "sum(b) == S and that V = 1^T M_a 1 came from the same mesh.")

    return {"Psi": Psi, "Ka_r": Ka_r, "Ma_r": Ma_r, "beta_m": beta_m,
            "omega_a_elastic": omega_a_e, "eig_time": 0.0,
            "beta0_rel": float(beta0_rel), "zero_mode_stiffness_rel": ka00,
            "ma_r_max_dev": ma_r_dev, "ma_r_cond": ma_r_cond,
            "eig_freq_max_hz": f_a_solved_max,
            "retained_freq_max_hz": f_a_retained_max,
            "n_modes": int(Psi.shape[1]),
            "n_port_attachment": n_port_att, "port_attachment_diag": port_diag}


def build_structural_basis(Ks, Ms, G, Psi, dof_list, n_modes: int, struct_fmax: float,
                           enrich: bool = True, n_attach_acoustic: int = 12,
                           static_solver: str = "auto") -> dict:
    """SLEPc eigensolve + `assemble_structural_reduced` (see there for the theory)."""
    t_total = time.time()
    n_u = Ks.shape[0]
    lam_s, Phi, t_eig, eig_diag = solve_eigen_full(
        Ks, Ms, n_modes=min(n_modes, n_u - 1),
        # sigma = 0: K_s is SPD thanks to the weak spring, and a shift at the rigid
        # frequency would sit ON an eigenvalue (near-singular shifted matrix).
        sigma=0.0, label="structure")
    omega_s_all = np.sqrt(np.maximum(lam_s, 0.0))
    solved_freq_max = float(np.max(omega_s_all) / (2.0 * np.pi)) \
        if omega_s_all.size else 0.0
    keep = omega_s_all <= 2.0 * np.pi * struct_fmax
    Phi_retained = np.ascontiguousarray(Phi[:, keep])
    omega_retained = omega_s_all[keep]
    del Phi
    out = assemble_structural_reduced(
        Ks, Ms, G, Psi, dof_list, Phi_retained, omega_retained,
        struct_fmax=np.inf,
        enrich=enrich, n_attach_acoustic=n_attach_acoustic,
        static_solver=static_solver, solved_freq_max_hz=solved_freq_max)
    out["eig_time"] = t_eig
    out["basis_prep_time"] = max(
        0.0, time.time() - t_total - t_eig - out.get("attach_time", 0.0))
    out["eigenbasis_diagnostics"] = eig_diag
    return out


def assemble_structural_reduced(Ks, Ms, G, Psi, dof_list, Phi_all, omega_s_all,
                                struct_fmax: float = np.inf, enrich: bool = True,
                                n_attach_acoustic: int = 12,
                                static_solver: str = "auto",
                                solved_freq_max_hz: float | None = None) -> dict:
    """Structural reduced basis: normal modes, optionally enriched with attachment modes.

    Pure linear algebra (no SLEPc) so it is unit-testable locally.

    Why enrichment is needed (this is the physics, not a numerical trick): the guitar
    top's first *elastic* mode is ~1.16 kHz, so in 20-800 Hz the structure responds
    quasi-STATICALLY.  A truncated normal-mode basis represents that static compliance
    only through the tails of modes it does not contain, so it under-predicts the
    response — exactly the ~7-8 dB gap E showed against D.

    Attachment (Craig-Bampton style) vectors, one per physical load path:
        x_bridge_i = K_s^{-1} e_{bridge_i}     (the bridge point force)
        x_fsi_j    = K_s^{-1} (G^T psi_j)      (the pressure load of acoustic mode j,
                                                INCLUDING the constant-pressure mode --
                                                that is how a static cavity pressure
                                                pushes the plates)
    Each static solution X = K_s^{-1} F is deflated of its retained-mode content by
    an M-ORTHOGONAL projection onto span(Phi) (eigenvalue/normalization independent):

        coeff = (Phiᵀ M_s Phi)^{-1} Phiᵀ M_s X ;   X_res = X - Phi coeff

    which also removes the huge weak-spring RIGID compliance (omega_rigid ~ 1 Hz)
    that would otherwise dominate and wreck the conditioning of the basis.

    Returns Q (M_s-orthonormal), K_r = Q^T K_s Q, M_r = Q^T M_s Q (= I), q_b = Q[dofs].
    """
    n_u = Ks.shape[0]
    omega_s_all = np.asarray(omega_s_all, float)
    f_s_all = omega_s_all / (2.0 * np.pi)
    f_s_solved_max = (float(solved_freq_max_hz) if solved_freq_max_hz is not None
                      else (float(f_s_all.max()) if f_s_all.size else 0.0))

    keep = f_s_all <= struct_fmax
    Phi_input = np.asarray(Phi_all, float)
    Phi = Phi_input if np.all(keep) else Phi_input[:, keep]
    omega_s = omega_s_all[keep]
    if omega_s.size == 0:
        raise RuntimeError("No structural modes retained (struct_fmax too low?).")
    f_s_retained_max = float(omega_s.max() / (2.0 * np.pi))

    Ks_real = sp.csc_matrix(Ks).real
    Ms_real = sp.csr_matrix(Ms).real
    n_rigid = int(np.sum(omega_s < 2.0 * np.pi * 10.0))
    t_basis = time.time()

    # The production eigensolver already guarantees this.  Keeping the pure linear-
    # algebra entry point robust makes its normalization-independent projection
    # contract explicit and prevents an expensive attachment solve with a bad Phi.
    MsPhi = Ms_real @ Phi
    M_phi = Phi.T @ MsPhi
    phi_input_dev = float(np.max(np.abs(M_phi - np.eye(M_phi.shape[0]))))
    phi_reorthonormalized = False
    if phi_input_dev > 1e-8:
        Phi_orth, kept_phi = _m_orthonormalize(Phi, Ms_real)
        if len(kept_phi) != Phi.shape[1]:
            raise RuntimeError(
                f"retained structural mode basis is rank-deficient "
                f"({len(kept_phi)}/{Phi.shape[1]} independent columns)")
        Phi = Phi_orth
        MsPhi = Ms_real @ Phi
        M_phi = Phi.T @ MsPhi
        phi_reorthonormalized = True
    phi_dev = float(np.max(np.abs(M_phi - np.eye(M_phi.shape[0]))))
    if phi_dev > 1e-6:
        raise RuntimeError(
            f"retained structural modes are not M-orthonormal "
            f"(max deviation {phi_dev:.2e})")
    basis_prep_time = time.time() - t_basis

    if not enrich:
        Q = Phi
        K_r = Q.T @ (Ks_real @ Q)
        M_r = Q.T @ (Ms_real @ Q)
        eigvals_Mr = np.linalg.eigvalsh(0.5 * (M_r + M_r.T))
        m_r_dev = float(np.max(np.abs(M_r - np.eye(M_r.shape[0]))))
        m_r_cond = float(eigvals_Mr.max() / max(eigvals_Mr.min(), 1e-300))
        if (not np.isfinite(m_r_dev) or not np.all(np.isfinite(eigvals_Mr))
                or m_r_dev > 1e-6 or eigvals_Mr.min() < 1e-8):
            raise RuntimeError(
                f"modal structural basis is not M-orthonormal "
                f"(max deviation {m_r_dev:.2e}, min eigenvalue {eigvals_Mr.min():.2e})")
        return {"Q": Q, "K_r": K_r, "M_r": M_r, "omega_s": omega_s,
                "n_modes": int(Q.shape[1]), "n_attachment": 0, "n_rigid": n_rigid,
                "n_attachment_dropped": 0,
                "n_attach_acoustic_used": 0,
                "m_r_max_dev": m_r_dev, "m_r_cond": m_r_cond,
                "phi_input_gram_dev": phi_input_dev,
                "phi_reorthonormalized": phi_reorthonormalized,
                "eig_time": 0.0, "basis_prep_time": basis_prep_time,
                "attach_time": 0.0,
                "eig_freq_max_hz": f_s_solved_max,
                "retained_freq_max_hz": f_s_retained_max, "basis": "modal"}

    t0 = time.time()
    G_real = sp.csr_matrix(G).real
    n_att_a = min(n_attach_acoustic, Psi.shape[1])
    loads = []
    for dof in dof_list:                              # bridge static compliance
        e = np.zeros(n_u); e[dof] = 1.0
        loads.append(e)
    for j in range(n_att_a):                          # FSI pressure loads
        loads.append(G_real.T @ Psi[:, j])
    F = np.column_stack(loads)

    X = _static_solve(Ks_real, F, solver=static_solver)   # static solutions
    # Deflate the retained-mode content (residual attachment modes) by an
    # M-ORTHOGONAL projection of the static solution X onto span(Phi).  This does
    # NOT depend on each column being an exact eigenvector with a matching omega_n
    # (which can break under near-degenerate realification) nor on Phi's exact
    # normalization:
    #     coeff  = (Phiᵀ M_s Phi)^{-1} Phiᵀ M_s X ;   X_res = X - Phi coeff
    # For an M-orthonormal eigenbasis Phiᵀ M_s Phi = I and Phiᵀ M_s X = (PhiᵀF)/ω²,
    # so this reduces exactly to the old modal form — but stays correct otherwise.
    X_res = _m_project_out(X, Phi, Ms_real, MPhi=MsPhi)
    del X, MsPhi, M_phi

    # Phi is already M_s-orthonormal (enforced in solve_eigen_full), so only the
    # attachment vectors are orthogonalized — against Phi and against each other.
    X_orth, kept = _m_orthonormalize_against(X_res, Phi, Ms_real)
    n_att_kept = int(X_orth.shape[1])
    n_att_dropped = int(F.shape[1] - n_att_kept)      # dependent directions removed
    Q = np.column_stack([Phi, X_orth]) if n_att_kept else Phi
    K_r = Q.T @ (Ks_real @ Q)
    M_r = Q.T @ (Ms_real @ Q)
    # Final combined-basis orthonormality: computed from the ACTUAL Q (M_r is NEVER
    # overwritten with the identity).  With the fixed real eigenbasis + M-projection
    # deflation this should now be ~machine-eps; a large value means the basis is
    # still rank-deficient -> fail loud so bulk generation cannot use a bad basis.
    m_r_dev = float(np.max(np.abs(M_r - np.eye(M_r.shape[0]))))
    eigvals_Mr = np.linalg.eigvalsh(0.5 * (M_r + M_r.T))
    m_r_cond = float(eigvals_Mr.max() / max(eigvals_Mr.min(), 1e-300))
    if (not np.isfinite(m_r_dev) or not np.all(np.isfinite(eigvals_Mr))
            or m_r_dev > 1e-6 or eigvals_Mr.min() < 1e-8):
        raise RuntimeError(
            f"[modal-coupled] combined structural basis is not M-orthonormal "
            f"(max|M_r - I|={m_r_dev:.2e}, min eig(M_r)={eigvals_Mr.min():.2e}, "
            f"cond={m_r_cond:.2e}). Basis rank-deficient — refusing to proceed.")
    t_att = time.time() - t0

    print(f"[modal-coupled] structural basis: {Phi.shape[1]} normal modes "
          f"({n_rigid} rigid/weak-spring, to {omega_s.max()/(2*np.pi):.0f} Hz) + "
          f"{n_att_kept}/{F.shape[1]} attachment modes "
          f"({n_att_dropped} dropped dependent) -> {Q.shape[1]} vectors; "
          f"max|M_r-I|={m_r_dev:.2e} cond(M_r)={m_r_cond:.2e} ({t_att:.1f}s)")
    return {"Q": Q, "K_r": K_r, "M_r": M_r, "omega_s": omega_s,
            "n_modes": int(Q.shape[1]), "n_attachment": n_att_kept,
            "n_attachment_dropped": n_att_dropped, "n_rigid": n_rigid,
            "n_attach_acoustic_used": int(n_att_a),
            "m_r_max_dev": m_r_dev, "m_r_cond": m_r_cond,
            "phi_input_gram_dev": phi_input_dev,
            "phi_reorthonormalized": phi_reorthonormalized,
            "eig_time": 0.0, "basis_prep_time": basis_prep_time,
            "attach_time": t_att,
            "eig_freq_max_hz": f_s_solved_max,
            "retained_freq_max_hz": f_s_retained_max,
            "basis": "craig-bampton"}


# ---------------------------------------------------------------------------
# Full driver (FEM assembly + eigensolves + reduced sweep)
# ---------------------------------------------------------------------------

def _basis_coverage(solved_struct_hz: float, solved_acoustic_hz: float,
                    struct_cutoff_hz: float, acoustic_cutoff_hz: float,
                    analysis_max_hz: float, band_margin: float = 1.5,
                    boundary_rtol: float = 1e-6) -> dict:
    """Report whether both reduced bases are complete through the useful band.

    The highest retained eigenfrequency is not a completeness boundary: it can sit
    below a target that falls inside a genuine spectral gap.  The available basis
    interval is therefore limited by both the requested cutoff and the spectrum that
    was actually solved.
    """
    values = np.asarray([
        solved_struct_hz, solved_acoustic_hz, struct_cutoff_hz,
        acoustic_cutoff_hz, analysis_max_hz, band_margin,
    ], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("basis coverage inputs must all be finite")
    if np.any(values[:2] < 0.0) or np.any(values[2:] <= 0.0):
        raise ValueError("basis frequencies must be non-negative and limits positive")
    if not np.isfinite(boundary_rtol) or boundary_rtol < 0.0:
        raise ValueError("basis coverage boundary_rtol must be finite and non-negative")

    struct_limit = min(float(struct_cutoff_hz), float(solved_struct_hz))
    acoustic_limit = min(float(acoustic_cutoff_hz), float(solved_acoustic_hz))
    target = float(band_margin) * float(analysis_max_hz)
    # Require a solved eigenvalue strictly above each boundary.  Stopping exactly on
    # a degenerate eigenvalue cannot prove that every direction in that eigenspace was
    # returned by a finite-nev eigensolve.
    def beyond(value, boundary):
        return value > boundary * (1.0 + boundary_rtol)

    cutoff_reached = bool(
        beyond(solved_struct_hz, struct_cutoff_hz)
        and beyond(solved_acoustic_hz, acoustic_cutoff_hz))
    band_covered = bool(
        struct_cutoff_hz >= target and acoustic_cutoff_hz >= target
        and beyond(solved_struct_hz, target) and beyond(solved_acoustic_hz, target))
    return {
        "cutoff_reached": cutoff_reached,
        "coverage_ok": cutoff_reached,  # backward-compatible metadata meaning
        "band_covered": band_covered,
        "basis_frequency_limit_struct_hz": struct_limit,
        "basis_frequency_limit_acoustic_hz": acoustic_limit,
        "band_target_hz": target,
        "band_margin": float(band_margin),
        "boundary_rtol": float(boundary_rtol),
    }

def compute_modal_coupled_admittance(
    msh_path,
    material: dict,
    bridge_coords,
    freq_min: float = 20.0,
    freq_max: float = 800.0,
    freq_points: int = 60,
    struct_fmax: float = 3000.0,
    acoustic_fmax: float = 3000.0,
    n_struct_modes: int = 300,
    n_acoustic_modes: int = 60,
    output_dir="results/modal_coupled",
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 5e-6,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
    c: float = AIR_C,
    rho0: float = AIR_RHO0,
    t_hole_mm: float = None,
    port_end_corrections: int = 1,
    force_n: float = 1.0,
    basis: str = "craig-bampton",
    n_attach_acoustic: int = 12,
    acoustic_attachment: str = "none",   # "none" | "port"  (see CLI --acoustic-attachment)
    static_solver: str = "auto",
    bridge_points=None,
    freqs=None,
    bulk: bool = False,                  # bulk generation: skip reduced_basis.npz + PNGs
) -> dict:
    """Modal-coupled ("E") bridge admittance.  Same physics as D, reduced basis.

    `basis`:
      'craig-bampton' (DEFAULT) — normal modes + attachment/residual modes for the
          bridge force and for the FSI pressure loads G^T psi_j.  This restores the
          quasi-static compliance that a truncated normal-mode basis misses (the
          structure's first elastic mode is far above the A0 band).
      'modal' — normal modes only (the original E; kept as an ABLATION so the effect
          of enrichment can be measured, never as the recommended path).

    The acoustic basis always contains the ANALYTIC constant-pressure mode
    psi_0 = 1/sqrt(V) (cavity compliance, beta_0 = S/sqrt(V) exactly), plus the
    elastic rigid-walled cavity modes, M_a-orthonormalized.

    Multi-bridge: pass `bridge_points`; the reduced matrices do not depend on the
    bridge, so all bridges are solved from the same bases (attachment vectors are
    added per bridge).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the analysed grid before any coverage decision.  The previous code
    # accepted an explicit 20--5000 Hz grid but still evaluated basis coverage
    # against the API default freq_max=800 Hz, allowing a severely truncated basis
    # to pass bulk QC.
    if freqs is not None:
        freqs = np.asarray(freqs, dtype=float)
        if (freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs))
                or freqs[0] <= 0.0 or np.any(np.diff(freqs) <= 0)):
            raise ValueError(
                "explicit `freqs` must be finite, positive, and strictly increasing")
    else:
        if (not np.isfinite(freq_min) or not np.isfinite(freq_max)
                or freq_min <= 0 or freq_max <= freq_min or int(freq_points) < 2):
            raise ValueError("invalid frequency range/grid")
        freqs = np.linspace(freq_min, freq_max, int(freq_points))
    analysis_freq_min = float(freqs[0])
    analysis_freq_max = float(freqs[-1])
    analysis_freq_points = int(freqs.size)

    from mpi4py import MPI
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            f"compute_modal_coupled_admittance is serial (MPI size must be 1, got "
            f"{MPI.COMM_WORLD.size}); launch each dataset worker without mpirun.")
    stale_comparison = output_dir / "modal_vs_full_coupled.json"
    if stale_comparison.exists():
        stale_comparison.unlink()
    _write_run_status(output_dir, "running")
    if basis not in ("craig-bampton", "modal"):
        raise ValueError(f"basis must be 'craig-bampton' or 'modal', got '{basis}'")

    pts = bridge_points if bridge_points is not None else bridge_coords

    # 1. Same operators as D (single source of truth) -------------------------
    blk = fac.build_coupled_blocks(
        msh_path, material, pts, rigid_freq_hz=rigid_freq_hz,
        c=c, rho0=rho0, t_hole_mm=t_hole_mm,
        port_end_corrections=port_end_corrections)
    timing = dict(blk["timing"])
    Ks, Ms, Ka, Ma = blk["Ks"], blk["Ms"], blk["Ka"], blk["Ma"]
    G, b, S = blk["G"], blk["b"], blk["S"]
    n_u, n_p = blk["n_u"], blk["n_p"]
    est, M_h = blk["est"], blk["M_h"]
    V_cav = blk["cavity_volume"]
    bridges = blk["bridges"]
    dof_list = [br["uz_r"] for br in bridges]
    snap_mm = bridges[0]["snap_distance_mm"]

    alpha = material.get("alpha", rayleigh_alpha)
    beta_ray = material.get("beta", rayleigh_beta)

    # 2. Acoustic basis (analytic zero mode forced in) ------------------------
    ab = build_acoustic_basis(Ka, Ma, b, S, V_cav,
                              n_modes=min(n_acoustic_modes, n_p - 1),
                              acoustic_fmax=acoustic_fmax, c=c,
                              port_attachment=(acoustic_attachment == "port"),
                              attachment_solver=static_solver)
    Psi, Ka_r, Ma_r, beta_m = ab["Psi"], ab["Ka_r"], ab["Ma_r"], ab["beta_m"]
    timing["eig_acoustic"] = ab["eig_time"]
    acoustic_eig_diag = ab.get("eigenbasis_diagnostics", {})
    timing["eig_acoustic_setup"] = acoustic_eig_diag.get("operator_setup_s", 0.0)
    timing["eig_acoustic_solve"] = acoustic_eig_diag.get("eigensolve_s", 0.0)
    timing["eig_acoustic_postprocess"] = (
        acoustic_eig_diag.get("realification_and_extract_s", 0.0)
        + acoustic_eig_diag.get("mass_gram_check_s", 0.0))
    timing["acoustic_basis_prep"] = ab.get("basis_prep_time", 0.0)

    # 3. Structural basis (normal modes [+ attachment modes]) -----------------
    sb = build_structural_basis(Ks, Ms, G, Psi, dof_list,
                                n_modes=min(n_struct_modes, n_u - 1),
                                struct_fmax=struct_fmax,
                                enrich=(basis == "craig-bampton"),
                                n_attach_acoustic=n_attach_acoustic,
                                static_solver=static_solver)
    Q, K_r, M_r = sb["Q"], sb["K_r"], sb["M_r"]
    timing["eig_struct"] = sb["eig_time"]
    structural_eig_diag = sb.get("eigenbasis_diagnostics", {})
    timing["eig_struct_setup"] = structural_eig_diag.get("operator_setup_s", 0.0)
    timing["eig_struct_solve"] = structural_eig_diag.get("eigensolve_s", 0.0)
    timing["eig_struct_postprocess"] = (
        structural_eig_diag.get("realification_and_extract_s", 0.0)
        + structural_eig_diag.get("mass_gram_check_s", 0.0))
    timing["struct_basis_prep"] = sb.get("basis_prep_time", 0.0)
    timing["attachment"] = sb.get("attach_time", 0.0)

    # Coverage: TWO different questions, previously conflated into one false alarm.
    #   cutoff_reached : did the eigensolve reach the cutoff the USER asked for?
    #                    (a soft, informational condition — the cutoff is a knob)
    #   band_covered   : does the basis extend well above the ANALYSED band?
    #                    (the condition that actually threatens the physics)
    # Do not use the highest retained eigenfrequency as a completeness boundary: it
    # can lie below the cutoff simply because the next physical mode is above it.
    # Coverage uses the raw solved spectrum and the requested retention cutoff.
    f_s_solved = sb["eig_freq_max_hz"]
    f_a_solved = ab["eig_freq_max_hz"]
    f_s_retained = sb["retained_freq_max_hz"]
    f_a_retained = ab["retained_freq_max_hz"]
    coverage = _basis_coverage(
        f_s_solved, f_a_solved, struct_fmax, acoustic_fmax, analysis_freq_max)
    cutoff_reached = coverage["cutoff_reached"]
    coverage_ok = coverage["coverage_ok"]
    band_covered = coverage["band_covered"]
    f_s_basis_limit = coverage["basis_frequency_limit_struct_hz"]
    f_a_basis_limit = coverage["basis_frequency_limit_acoustic_hz"]
    band_target = coverage["band_target_hz"]
    band_margin = coverage["band_margin"]
    coverage_boundary_rtol = coverage["boundary_rtol"]
    if not band_covered:
        print(f"[modal-coupled] WARNING: the reduced basis does NOT cover the analysed "
              f"band with margin (available struct {f_s_basis_limit:.0f} Hz, "
              f"acoustic {f_a_basis_limit:.0f} Hz vs "
              f"{band_margin:.1f} x freq_max = {band_target:.0f} Hz). "
              f"Raise --n-struct-modes / --n-acoustic-modes: the result IS truncated.")
    elif not cutoff_reached:
        print(f"[modal-coupled] note: eigensolve stopped below the requested cutoffs "
              f"(struct {f_s_solved:.0f}/{struct_fmax:.0f} Hz, acoustic "
              f"{f_a_solved:.0f}/{acoustic_fmax:.0f} Hz) but still covers the analysed band "
              f"(<= {freq_max:.0f} Hz) with margin - not a truncation problem.")

    # 4. Projection ----------------------------------------------------------
    t0 = time.time()
    G_real = sp.csr_matrix(G).real
    G_r = Psi.T @ (G_real @ Q)                   # (m_p, m_u)
    q_b = np.asarray([Q[dof, :] for dof in dof_list])   # (n_b, m_u)
    timing["projection"] = time.time() - t0
    m_u, m_p = Q.shape[1], Psi.shape[1]

    # 5. Reduced sweep -------------------------------------------------------
    t0 = time.time()
    out = reduced_coupled_sweep_general(
        freqs, K_r, M_r, Ka_r, Ma_r, G_r, beta_m, q_b, S, M_h,
        alpha, beta_ray, c=c, rho0=rho0, force_n=force_n)
    timing["solve"] = time.time() - t0
    timing["per_freq_mean_s"] = float(out["per_freq_s"].mean())
    timing["tracked_compute_s"] = sum(float(timing.get(key, 0.0)) for key in (
        "mesh", "assembly", "eig_acoustic", "acoustic_basis_prep",
        "eig_struct", "struct_basis_prep", "attachment", "projection", "solve",
    ))
    timing["total_s"] = timing["tracked_compute_s"]  # backward-compatible key
    Y_m, p_bar_m, U_h_m = out["Y"], out["p_bar"], out["U_h"]
    Y, p_bar, U_h = Y_m[0], p_bar_m[0], U_h_m[0]

    nonfinite = {name: int(np.sum(~np.isfinite(arr)))
                 for name, arr in (("Y", Y_m), ("p_bar", p_bar_m), ("U_h", U_h_m))}
    if any(nonfinite.values()):
        raise RuntimeError(f"Modal-coupled solve produced non-finite values {nonfinite}. "
                           f"No output saved.")

    # 6. A0 - SAME detector as D (local peak on |p_bar|, cross-checked with |U_h|) ---
    a0 = fac.detect_a0(freqs, p_bar, U_h, est.estimated_helmholtz_hz)
    f_a0 = float(a0["A0_observed_hz"]) if a0["A0_detected"] else None
    rel = a0.get("rel_error_vs_f_H")
    print(f"[modal-coupled] eig {timing['eig_struct'] + timing['eig_acoustic']:.1f}s + "
          f"sweep {timing['solve']:.1f}s "
          f"({timing['per_freq_mean_s']*1e3:.1f} ms/freq)")

    # 7. Save ----------------------------------------------------------------
    np.savez(str(output_dir / "admittance_air_modal_coupled.npz"),
             frequencies=freqs, admittance=Y)
    np.savez(str(output_dir / "pressure_mean_cavity.npz"), frequencies=freqs, p_bar=p_bar)
    np.savez(str(output_dir / "soundhole_volume_velocity.npz"), frequencies=freqs, U_h=U_h)
    np.savez(str(output_dir / "admittance_air_modal_coupled_multi.npz"),
             frequencies=freqs, admittance=Y_m, p_bar=p_bar_m, U_h=U_h_m,
             bridge_requested_xyz=np.asarray(
                 [br["bridge_requested_xyz"] for br in bridges], float),
             bridge_snapped_xyz=np.asarray(
                 [br["bridge_snapped_xyz"] for br in bridges], float),
             snap_distance_mm=np.asarray(
                 [br["snap_distance_mm"] for br in bridges], float))
    # Canonical compact reduced model.  Q/Psi full FEM vectors are deliberately
    # excluded; the projected operators below are sufficient to resample the
    # exact reduced system on any dense/adaptive grid without another FEM solve.
    reduced_model_path = output_dir / "reduced_model.npz"
    np.savez_compressed(
        str(reduced_model_path),
        schema_version=np.array("air-coupled-reduced-model-v1"),
        K_r=K_r, M_r=M_r, Ka_r=Ka_r, Ma_r=Ma_r, G_r=G_r,
        beta_m=beta_m, q_b=q_b,
        omega_s_modes_rad_s=np.asarray(sb["omega_s"], float),
        omega_a_elastic_rad_s=np.asarray(ab["omega_a_elastic"], float),
        soundhole_area_m2=np.array(float(S)),
        soundhole_inertance_kg_m4=np.array(float(M_h)),
        cavity_volume_m3=np.array(float(V_cav)),
        air_speed_m_s=np.array(float(c)),
        air_density_kg_m3=np.array(float(rho0)),
        rayleigh_alpha=np.array(float(alpha)),
        rayleigh_beta=np.array(float(beta_ray)),
        force_n=np.array(float(force_n)),
        time_convention=np.array("exp(+i omega t)"),
        response_units=np.array("Y:m/s/N;p_bar:Pa;U_h:m^3/s"),
    )
    t_verify = time.time()
    audit_idx = np.unique(np.linspace(
        0, freqs.size - 1, min(3, freqs.size), dtype=int))
    reconstructed = reconstruct_reduced_model(reduced_model_path, freqs[audit_idx])
    model_reconstruction_error = 0.0
    for key, reference in (("Y", Y_m), ("p_bar", p_bar_m), ("U_h", U_h_m)):
        expected = np.asarray(reference)[:, audit_idx]
        scale = max(float(np.max(np.abs(expected))), 1e-30)
        model_reconstruction_error = max(
            model_reconstruction_error,
            float(np.max(np.abs(reconstructed[key] - expected)) / scale))
    if (not np.isfinite(model_reconstruction_error)
            or model_reconstruction_error > 1e-10):
        raise RuntimeError(
            "persisted air-coupled reduced model does not reconstruct solver response "
            f"(relative error {model_reconstruction_error:.3e})")
    timing["model_verify"] = time.time() - t_verify
    timing["tracked_compute_s"] += timing["model_verify"]
    timing["total_s"] = timing["tracked_compute_s"]
    if not bulk:
        np.savez(str(output_dir / "reduced_basis.npz"),
                 K_r=K_r, M_r=M_r, Ka_r=Ka_r, Ma_r=Ma_r, G_r=G_r, beta_m=beta_m, q_b=q_b,
                 omega_s_modes=sb["omega_s"], omega_a_elastic=ab["omega_a_elastic"])
    (output_dir / "A0_estimated_vs_observed.json").write_text(json.dumps({
        **a0,
        "A0_estimated_hz": est.estimated_helmholtz_hz,   # rigid-wall f_H (legacy key)
        "rel_error": rel,
        "cavity_volume_m3": V_cav, "soundhole_area_m2": S,
        "n_freq_points": analysis_freq_points,
        "freq_min_hz": analysis_freq_min,
        "freq_max_hz": analysis_freq_max,
    }, indent=2, allow_nan=False))
    (output_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (output_dir / "modal_coupled_metadata.json").write_text(json.dumps({
        "backend": f"E modal-coupled (reduced, basis={basis})",
        "solver_revision": SOLVER_REVISION,
        "reference_backend": "D fenics_admittance_coupled (full FEM-FEM)",
        "formulation": "Everstine unsymmetric u-p, projected onto (Q, Psi)",
        "time_convention": "exp(+i omega t)",
        "reduced_block_system": "[[K_r(1+i w beta) - w^2 M_r + i w alpha M_r, -G_r.T], "
                                "[-rho0 w^2 G_r, Ka_r - (w^2/c^2) Ma_r + kappa beta beta^T]]",
        "structural_basis": sb["basis"],
        "structural_basis_detail": (
            "normal modes (K_s phi = ws^2 M_s phi, K_s includes weak spring) + "
            "M_s-orthonormalized residual attachment modes for the bridge force and "
            "for the FSI pressure loads G^T psi_j"
            if basis == "craig-bampton" else
            "normal modes only (ABLATION: misses the quasi-static compliance)"),
        "acoustic_basis": (
            "analytic constant-pressure mode 1/sqrt(V) + "
            + ("port attachment (quasi-static K_a^-1 b) + "
               if ab["n_port_attachment"] else "")
            + "rigid-walled (Neumann) elastic modes, M_a-orthonormalized"),
        "acoustic_zero_mode": "analytic (NOT taken from SLEPc)",
        "acoustic_port_attachment": bool(ab["n_port_attachment"]),
        "acoustic_port_attachment_diag": ab.get("port_attachment_diag", {}),
        "rigid_structural_modes_kept": True,
        "n_wood_u_dofs": int(n_u), "n_air_p_dofs": int(n_p),
        "n_struct_basis_vectors": int(m_u),
        "n_struct_normal_modes": int(sb["omega_s"].size),
        "n_struct_attachment_modes": int(sb["n_attachment"]),
        "n_struct_attachment_modes_dropped": int(sb["n_attachment_dropped"]),
        "n_struct_rigid_modes": int(sb["n_rigid"]),
        "structural_mass_orthonormality_max_dev": float(sb["m_r_max_dev"]),
        "structural_reduced_mass_condition": float(sb["m_r_cond"]),
        "structural_phi_input_gram_dev": float(sb["phi_input_gram_dev"]),
        "structural_phi_reorthonormalized": bool(sb["phi_reorthonormalized"]),
        "structural_eigenbasis_diagnostics": sb.get("eigenbasis_diagnostics", {}),
        "n_acoustic_basis_vectors": int(m_p),
        "n_attach_acoustic_used": int(sb["n_attach_acoustic_used"]),
        "acoustic_eigenbasis_diagnostics": ab.get("eigenbasis_diagnostics", {}),
        "acoustic_mass_orthonormality_max_dev": float(ab["ma_r_max_dev"]),
        "acoustic_reduced_mass_condition": float(ab["ma_r_cond"]),
        "struct_fmax_hz": float(struct_fmax), "acoustic_fmax_hz": float(acoustic_fmax),
        "eig_freq_max_struct_hz": f_s_solved,
        "eig_freq_max_acoustic_hz": f_a_solved,
        "retained_freq_max_struct_hz": f_s_retained,
        "retained_freq_max_acoustic_hz": f_a_retained,
        "basis_frequency_limit_struct_hz": f_s_basis_limit,
        "basis_frequency_limit_acoustic_hz": f_a_basis_limit,
        "basis_band_target_hz": band_target,
        "basis_band_margin": band_margin,
        "basis_coverage_boundary_rtol": coverage_boundary_rtol,
        "coverage_ok": coverage_ok,          # = cutoff_reached (backward compatible)
        "cutoff_reached": cutoff_reached,     # eigensolve reached the requested --*-fmax
        "band_covered": band_covered,         # basis reaches 1.5x freq_max (the real gate)
        "beta0_vs_S_over_sqrtV_rel": float(ab["beta0_rel"]),
        "zero_mode_stiffness_rel": float(ab["zero_mode_stiffness_rel"]),
        "soundhole_area_m2": float(S), "cavity_volume_m3": float(V_cav),
        "n_bridges": int(len(bridges)), "bridges": bridges,
        "bridge_requested_mm": [float(v) for v in bridges[0]["bridge_requested_xyz"]],
        "bridge_snap_distance_mm": float(snap_mm),
        "port_end_corrections": int(port_end_corrections),
        "static_solver": str(static_solver),
        "n_attach_acoustic_requested": int(n_attach_acoustic),
        "analysis_freq_min_hz": analysis_freq_min,
        "analysis_freq_max_hz": analysis_freq_max,
        "analysis_freq_points": analysis_freq_points,
        "model_reconstruction_max_rel_error": model_reconstruction_error,
    }, indent=2))
    if not bulk:                          # PNGs are diagnostics only — skip in bulk
        fac._plot_coupled(output_dir / "admittance.png", freqs, Y, p_bar, U_h,
                          est.estimated_helmholtz_hz,
                          float("nan") if f_a0 is None else f_a0,
                          title="Modal-coupled bridge admittance - bridge 0")
        fac._plot_all_bridge_coupled(
            output_dir / "bridge_plots", freqs, Y_m, p_bar_m, U_h_m,
            est.estimated_helmholtz_hz)
    _write_run_status(output_dir, "complete", comparison_completed=False)
    print(f"[modal-coupled] saved -> {output_dir}")

    return {"freqs": freqs, "Y": Y, "p_bar": p_bar, "U_h": U_h,
            "Y_multi": Y_m, "p_bar_multi": p_bar_m, "U_h_multi": U_h_m,
            "bridges": bridges,
            "A0_detected": bool(a0["A0_detected"]),
            "A0_observed": f_a0, "A0_estimated": est.estimated_helmholtz_hz,
            "n_struct_modes": m_u, "n_acoustic_modes": m_p,
            "coverage_ok": coverage_ok, "cutoff_reached": cutoff_reached,
            "band_covered": band_covered,
            "model_reconstruction_max_rel_error": model_reconstruction_error,
            "solver_revision": SOLVER_REVISION,
            "structural_basis_diagnostics": {
                "m_r_max_dev": sb["m_r_max_dev"],
                "m_r_cond": sb["m_r_cond"],
                "n_attachment_dropped": sb["n_attachment_dropped"],
                "eigenbasis": structural_eig_diag,
            },
            "acoustic_basis_diagnostics": {"eigenbasis": acoustic_eig_diag},
            "timing": timing}


# ---------------------------------------------------------------------------
# E vs D comparison (reference check — never a substitute for D)
# ---------------------------------------------------------------------------

def _peak_features(freqs, db_ref, db_test, prominence_db: float = 3.0) -> dict:
    """Resonance / antiresonance agreement between two |Y| dB spectra on one grid.

    For each prominent peak (and trough = antiresonance) of the REFERENCE (D), find
    the nearest corresponding feature in the test (E) spectrum and report frequency
    and amplitude errors.  Antiresonances are the hardest thing for a truncated basis
    to place correctly (they come from mode/compliance cancellation), which is why
    they are reported separately.
    """
    try:
        from scipy.signal import find_peaks
    except Exception:
        return {"peak_features": "scipy.signal unavailable"}

    def _match(idx_ref, idx_test):
        if len(idx_ref) == 0 or len(idx_test) == 0:
            return [], [], []
        f_ref, f_test = freqs[idx_ref], freqs[idx_test]
        pairs = [int(np.argmin(np.abs(f_test - fr))) for fr in f_ref]
        df = [abs(f_test[p] - fr) / max(fr, 1e-9) for p, fr in zip(pairs, f_ref)]
        da = [abs(db_test[idx_test[p]] - db_ref[i]) for p, i in zip(pairs, idx_ref)]
        return f_ref.tolist(), df, da

    pk_ref, _ = find_peaks(db_ref, prominence=prominence_db)
    pk_test, _ = find_peaks(db_test, prominence=prominence_db)
    tr_ref, _ = find_peaks(-db_ref, prominence=prominence_db)
    tr_test, _ = find_peaks(-db_test, prominence=prominence_db)

    f_pk, dfp, dap = _match(pk_ref, pk_test)
    f_tr, dft, dat = _match(tr_ref, tr_test)
    med = lambda v: float(np.median(v)) if len(v) else None
    return {
        "n_peaks_full": int(len(pk_ref)), "n_peaks_modal": int(len(pk_test)),
        "peak_freq_median_rel_err": med(dfp),
        "peak_amp_median_abs_err_db": med(dap),
        "n_antires_full": int(len(tr_ref)), "n_antires_modal": int(len(tr_test)),
        "antires_freq_median_rel_err": med(dft),
        "antires_depth_median_abs_err_db": med(dat),
        "peak_freqs_full_hz": [round(v, 2) for v in f_pk],
        "antires_freqs_full_hz": [round(v, 2) for v in f_tr],
    }

def compare_with_full_coupled(modal_dir, full_dir, output_dir=None) -> dict:
    """Compare E (modal-coupled) against D (full coupled) on a common grid.

    Both directories must come from the SAME mesh/bridge/frequency band.  Metrics:
    dB RMSE and correlation on |Y|, A0 shift from the p_bar peak.  Acceptance is
    deliberately reported, not enforced — D stays the reference.
    """
    modal_dir, full_dir = Path(modal_dir), Path(full_dir)
    status_path = modal_dir / RUN_STATUS_FILENAME
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status != "complete":
            raise RuntimeError(
                f"refusing to compare incomplete E output in {modal_dir} "
                f"(run status: {status!r})")
    full_batch_status_path = full_dir / "batch_step_status.json"
    if full_batch_status_path.exists():
        try:
            full_batch_status = json.loads(
                full_batch_status_path.read_text(encoding="utf-8")).get("status")
        except Exception as exc:
            raise RuntimeError(
                f"refusing to compare D output with an invalid batch status in "
                f"{full_dir}: {exc}") from exc
        if full_batch_status != "complete":
            raise RuntimeError(
                f"refusing to compare incomplete D output in {full_dir} "
                f"(batch status: {full_batch_status!r})")
    out = Path(output_dir) if output_dir else modal_dir
    out.mkdir(parents=True, exist_ok=True)

    # Prefer the MULTI-bridge arrays: comparing only bridge 0 hides the spread across
    # driving points, and the dataset drives several bridges per shape.
    mm = modal_dir / "admittance_air_modal_coupled_multi.npz"
    fm = full_dir / "admittance_air_full_coupled_multi.npz"
    if mm.exists() != fm.exists():
        raise FileNotFoundError(
            "E/D multi-bridge outputs are inconsistent: exactly one of "
            f"{mm} and {fm} exists; refusing a single-bridge fallback")
    if mm.exists() and fm.exists():
        dm, df = np.load(str(mm)), np.load(str(fm))
        f_m, f_f = dm["frequencies"], df["frequencies"]
        Ym_all, Yf_all = np.atleast_2d(dm["admittance"]), np.atleast_2d(df["admittance"])
        pbar_m_all, pbar_f_all = np.atleast_2d(dm["p_bar"]), np.atleast_2d(df["p_bar"])
        Uh_m_all, Uh_f_all = np.atleast_2d(dm["U_h"]), np.atleast_2d(df["U_h"])
    else:                                   # legacy single-bridge outputs
        dm = np.load(str(modal_dir / "admittance_air_modal_coupled.npz"))
        df = np.load(str(full_dir / "admittance_air_full_coupled.npz"))
        pm_ = np.load(str(modal_dir / "pressure_mean_cavity.npz"))
        pf_ = np.load(str(full_dir / "pressure_mean_cavity.npz"))
        um_ = np.load(str(modal_dir / "soundhole_volume_velocity.npz"))
        uf_ = np.load(str(full_dir / "soundhole_volume_velocity.npz"))
        f_m, f_f = dm["frequencies"], df["frequencies"]
        Ym_all, Yf_all = dm["admittance"][None, :], df["admittance"][None, :]
        pbar_m_all, pbar_f_all = pm_["p_bar"][None, :], pf_["p_bar"][None, :]
        Uh_m_all, Uh_f_all = um_["U_h"][None, :], uf_["U_h"][None, :]

    def _validate_result(name, freqs, Y, pbar, Uh):
        freqs = np.asarray(freqs, float)
        if freqs.ndim != 1 or freqs.size < 3 or not np.all(np.isfinite(freqs)):
            raise ValueError(f"{name} has an invalid frequency grid")
        if np.any(np.diff(freqs) <= 0.0):
            raise ValueError(f"{name} frequency grid must be strictly increasing")
        for field, values in (("admittance", Y), ("p_bar", pbar), ("U_h", Uh)):
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[1] != freqs.size:
                raise ValueError(
                    f"{name} {field} shape {values.shape} does not match "
                    f"frequency count {freqs.size}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} {field} contains non-finite values")
        if not (Y.shape[0] == pbar.shape[0] == Uh.shape[0]) or Y.shape[0] == 0:
            raise ValueError(f"{name} bridge counts are inconsistent or empty")

    _validate_result("E", f_m, Ym_all, pbar_m_all, Uh_m_all)
    _validate_result("D", f_f, Yf_all, pbar_f_all, Uh_f_all)
    if Ym_all.shape[0] != Yf_all.shape[0]:
        raise ValueError(
            f"E/D bridge counts differ ({Ym_all.shape[0]} vs {Yf_all.shape[0]}); "
            "refusing a partial comparison")
    n_b = Ym_all.shape[0]

    band_tol = 1e-9 * max(abs(float(f_f[0])), abs(float(f_f[-1])), 1.0)
    if f_m[0] > f_f[0] + band_tol or f_m[-1] < f_f[-1] - band_tol:
        raise ValueError(
            f"E frequency band [{f_m[0]:.6g}, {f_m[-1]:.6g}] Hz does not cover "
            f"D [{f_f[0]:.6g}, {f_f[-1]:.6g}] Hz; interpolation would extrapolate")

    if mm.exists() and fm.exists():
        for coord_key in ("bridge_requested_xyz", "bridge_snapped_xyz"):
            if coord_key not in dm.files or coord_key not in df.files:
                raise ValueError(
                    f"multi-bridge comparison requires {coord_key} in both E and D")
            coords_m = np.asarray(dm[coord_key], float)
            coords_f = np.asarray(df[coord_key], float)
            if (coords_m.shape != (n_b, 3) or coords_f.shape != (n_b, 3)
                    or not np.allclose(coords_m, coords_f, rtol=0.0, atol=1e-9)):
                raise ValueError(
                    f"E/D {coord_key} values differ; results are not for the "
                    "same bridge locations")

    f_est = None
    for candidate in (full_dir / "A0_estimated_vs_observed.json",
                      modal_dir / "A0_estimated_vs_observed.json"):
        if candidate.exists():
            try:
                j = json.loads(candidate.read_text())
                f_est = j.get("f_H_rigid_wall_hz") or j.get("A0_estimated_hz")
                if f_est:
                    break
            except Exception:
                pass
    if not f_est:
        f_est = 0.5 * (float(f_f.min()) + float(f_f.max()))

    # --- Per-bridge admittance metrics, on D's grid, over the band and sub-bands ----
    sub_bands = [("20-800", 20.0, 800.0), ("20-2000", 20.0, 2000.0),
                 ("full", float(f_f.min()), float(f_f.max()))]
    per_bridge, band_stats = [], {name: {"rmse": [], "rmse_w": [], "corr": []}
                                  for name, _, _ in sub_bands}
    for b in range(n_b):
        Ym_i = np.interp(f_f, f_m, np.abs(Ym_all[b]))
        db_m_b = 20 * np.log10(Ym_i + 1e-30)
        db_f_b = 20 * np.log10(np.abs(Yf_all[b]) + 1e-30)
        row = {"bridge": b}
        for name, lo_b, hi_b in sub_bands:
            m = (f_f >= lo_b) & (f_f <= hi_b)
            if np.count_nonzero(m) < 3:
                continue
            e = db_m_b[m] - db_f_b[m]
            w = np.abs(Yf_all[b][m]) / (np.mean(np.abs(Yf_all[b][m])) + 1e-30)
            r = float(np.sqrt(np.mean(e ** 2)))
            weight_sum = float(np.sum(w))
            if weight_sum <= 0.0:
                raise ValueError(f"D admittance is identically zero for bridge {b} in {name}")
            rw = float(np.sqrt(np.sum(w * e ** 2) / weight_sum))
            c = float(np.corrcoef(db_m_b[m], db_f_b[m])[0, 1])
            if not np.all(np.isfinite([r, rw, c])):
                raise ValueError(
                    f"undefined E/D metric for bridge {b} in {name}; "
                    "the dB spectrum may be constant")
            row[name] = {"Y_dB_rmse": r, "Y_dB_rmse_weighted": rw, "Y_dB_corr": c}
            band_stats[name]["rmse"].append(r)
            band_stats[name]["rmse_w"].append(rw)
            band_stats[name]["corr"].append(c)
        per_bridge.append(row)

    med = lambda v: float(np.median(v)) if len(v) else None
    bands = {name: {"median_Y_dB_rmse": med(s["rmse"]),
                    "median_Y_dB_rmse_weighted": med(s["rmse_w"]),
                    "median_Y_dB_corr": med(s["corr"]),
                    "worst_Y_dB_rmse": (float(np.max(s["rmse"])) if s["rmse"] else None),
                    "worst_Y_dB_corr": (float(np.min(s["corr"])) if s["corr"] else None)}
             for name, s in band_stats.items()}

    # Headline numbers = MEDIAN over bridges on the full band (was: bridge 0 only).
    rmse_db = bands["full"]["median_Y_dB_rmse"]
    rmse_db_w = bands["full"]["median_Y_dB_rmse_weighted"]
    corr = bands["full"]["median_Y_dB_corr"]

    # --- A0: same local-peak detector as D/E, per bridge ---------------------------
    a0_rows, a0_errs = [], []
    for b in range(n_b):
        am = fac.detect_a0(f_m, pbar_m_all[b], Uh_m_all[b], f_est)
        af = fac.detect_a0(f_f, pbar_f_all[b], Uh_f_all[b], f_est)
        ok = bool(am["A0_detected"] and af["A0_detected"])
        err = (abs(am["A0_observed_hz"] - af["A0_observed_hz"])
               / max(af["A0_observed_hz"], 1e-9)) if ok else None
        if err is not None:
            a0_errs.append(err)
        a0_rows.append({"bridge": b,
                        "A0_modal_hz": am["A0_observed_hz"],
                        "A0_full_hz": af["A0_observed_hz"],
                        "A0_rel_error": err,
                        "modal_detected": am["A0_detected"],
                        "full_detected": af["A0_detected"],
                        "full_resolution_ok": af.get("A0_resolution_ok"),
                        "full_failure": af.get("A0_failure"),
                        "modal_failure": am.get("A0_failure")})
    a0_detected_all = all(r["modal_detected"] and r["full_detected"] for r in a0_rows)
    a0_resolved = all(r["full_resolution_ok"] for r in a0_rows if r["full_detected"])
    a0_err = med(a0_errs)
    a0_err_worst = float(np.max(a0_errs)) if a0_errs else None
    a0_m = med([r["A0_modal_hz"] for r in a0_rows if r["A0_modal_hz"]])
    a0_f = med([r["A0_full_hz"] for r in a0_rows if r["A0_full_hz"]])

    # Bridge 0 spectra for the plot / peak features.
    db_m = 20 * np.log10(np.interp(f_f, f_m, np.abs(Ym_all[0])) + 1e-30)
    db_f = 20 * np.log10(np.abs(Yf_all[0]) + 1e-30)
    feat = _peak_features(f_f, db_f, db_m)
    pm = {"p_bar": pbar_m_all[0]}
    pf = {"p_bar": pbar_f_all[0]}

    worst_rmse = bands["full"]["worst_Y_dB_rmse"]
    worst_corr = bands["full"]["worst_Y_dB_corr"]
    a0_ok = bool(a0_detected_all and a0_resolved
                 and a0_err_worst is not None and a0_err_worst < 0.03)
    adm_ok = bool(worst_rmse is not None and worst_corr is not None
                  and worst_rmse < 1.0 and worst_corr > 0.99)
    res = {
        "modal_dir": str(modal_dir), "full_dir": str(full_dir),
        "n_bridges": int(n_b),
        "n_freq_full": int(f_f.size), "n_freq_modal": int(f_m.size),
        "band_hz": [float(f_f.min()), float(f_f.max())],
        # headline = median across bridges (full band)
        "Y_dB_rmse": rmse_db, "Y_dB_rmse_weighted": rmse_db_w, "Y_dB_corr": corr,
        "bands": bands,
        "per_bridge": per_bridge,
        "A0_per_bridge": a0_rows,
        "A0_modal_hz": a0_m, "A0_full_hz": a0_f, "A0_rel_error": a0_err,
        "A0_worst_rel_error": a0_err_worst,
        "f_H_rigid_wall_hz": float(f_est),   # NOT the coupled A0 - separate quantity
        **feat,
        # TWO SEPARATE ACCEPTANCE LAYERS - do not mix them.  The bridge admittance is
        # what the dataset/NN consumes; the cavity pressure / A0 is a physics check on
        # the acoustic basis.  They fail for different reasons and have different fixes
        # (structural enrichment vs acoustic port attachment).
        "acceptance": {
            "admittance": {
                "criteria": "every bridge: |Y| dB RMSE < 1.0 and corr > 0.99",
                "median_Y_dB_rmse": rmse_db, "median_Y_dB_corr": corr,
                "worst_Y_dB_rmse": worst_rmse, "worst_Y_dB_corr": worst_corr,
                "passed": adm_ok,
            },
            "cavity_pressure_A0": {
                "criteria": "every bridge: A0 detected in BOTH, D resolves it, E within 3 %",
                "A0_modal_hz": a0_m, "A0_full_hz": a0_f, "A0_rel_error": a0_err,
                "worst_A0_rel_error": a0_err_worst,
                "A0_detected_in_both": a0_detected_all,
                "A0_resolved_by_D_grid": a0_resolved,
                "passed": a0_ok,
                "note": None if a0_ok else
                        ("A0 comparison is NOT EVALUABLE (detection failed or D's grid "
                         "is too coarse to locate A0) — run a dedicated low-band sweep; "
                         "do NOT read this as an E error."
                         if not (a0_detected_all and a0_resolved) else
                         "A0 detected and resolved, but E is off by more than 3 %."),
            },
        },
        "guideline": "every bridge: RMSE < 1 dB & corr > 0.99; A0 within 3 % of D",
        "meets_guideline": bool(adm_ok and a0_ok),
    }
    (out / "modal_vs_full_coupled.json").write_text(json.dumps(res, indent=2))
    if status_path.exists():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        status_payload.update({
            "status": "complete",
            "comparison_completed": True,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        status_path.write_text(
            json.dumps(status_payload, indent=2), encoding="utf-8")
    acc = res["acceptance"]
    print(f"[E vs D] {n_b} bridge(s), median over bridges:")
    for name in ("20-800", "20-2000", "full"):
        bb = bands.get(name) or {}
        if bb.get("median_Y_dB_rmse") is None:
            continue
        print(f"   {name:>8} Hz : RMSE {bb['median_Y_dB_rmse']:.2f} dB "
              f"(weighted {bb['median_Y_dB_rmse_weighted']:.2f}, worst "
              f"{bb['worst_Y_dB_rmse']:.2f})  corr {bb['median_Y_dB_corr']:.4f} "
              f"(worst {bb['worst_Y_dB_corr']:.4f})")
    print(f"[E vs D] ADMITTANCE (all bridges): -> {'PASS' if adm_ok else 'FAIL'}")
    if a0_detected_all and a0_resolved:
        print(f"[E vs D] CAVITY/A0  : {a0_m:.1f} vs {a0_f:.1f} Hz "
              f"(median {a0_err*100:.1f}%, worst {a0_err_worst*100:.1f}%)  "
              f"-> {'PASS' if a0_ok else 'FAIL'}")
    else:
        print(f"[E vs D] CAVITY/A0  : NOT EVALUABLE "
              f"(detected in both: {a0_detected_all}, D grid resolves A0: {a0_resolved})"
              f" - run a dedicated low-band sweep; this is not an E error.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        ax[0].plot(f_f, db_f, "k", lw=1.5, label="D full coupled (reference)")
        ax[0].plot(f_m, 20 * np.log10(np.abs(Ym_all[0]) + 1e-30), "r--", lw=1.0,
                   label="E modal-coupled")
        ax[0].set_ylabel("|Y| [dB]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[0].set_title(f"Modal-coupled (E) vs full FEM-FEM coupled (D) - bridge 0 of "
                        f"{n_b} (medians in the JSON)")
        ax[1].plot(f_f, db_m - db_f, "b", lw=1.0)
        ax[1].set_ylabel("Delta |Y| [dB] (E - D)")
        ax[1].grid(alpha=0.3)
        # Cavity pressure: this is where the acoustic-basis (A0) defect shows up, and it
        # is a SEPARATE acceptance layer from |Y| — so plot it separately.
        ax[2].plot(f_f, np.abs(pf["p_bar"]), "k", lw=1.5, label="D |p_bar|")
        ax[2].plot(f_m, np.abs(pm["p_bar"]), "r--", lw=1.0, label="E |p_bar|")
        if a0_f:
            ax[2].axvline(a0_f, color="k", ls=":", lw=0.8, label=f"A0 D {a0_f:.0f} Hz")
        if a0_m:
            ax[2].axvline(a0_m, color="r", ls=":", lw=0.8, label=f"A0 E {a0_m:.0f} Hz")
        if not a0_ok:
            ax[2].set_title("A0 comparison NOT EVALUABLE (see acceptance.note)",
                            fontsize=9, color="r")
        ax[2].set_ylabel("|p_bar| (mean cavity)"); ax[2].set_xlabel("Frequency [Hz]")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(str(out / "modal_vs_full_coupled.png"), dpi=150)
        plt.close(fig)
        print(f"[E vs D] plot: {out / 'modal_vs_full_coupled.png'}")

        plot_dir = out / "bridge_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for b in range(n_b):
            db_f_b = 20 * np.log10(np.abs(Yf_all[b]) + 1e-30)
            db_m_b = 20 * np.log10(np.interp(f_f, f_m, np.abs(Ym_all[b])) + 1e-30)
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            ax[0].plot(f_f, db_f_b, "k", lw=1.5, label="D full coupled")
            ax[0].plot(f_m, 20 * np.log10(np.abs(Ym_all[b]) + 1e-30),
                       "r--", lw=1.0, label="E modal-coupled")
            ax[0].set_ylabel("|Y| [dB]")
            ax[0].set_title(f"Modal-coupled (E) vs full coupled (D) - bridge {b}")
            ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

            ax[1].plot(f_f, db_m_b - db_f_b, "b", lw=1.0)
            ax[1].set_ylabel("Delta |Y| [dB] (E - D)")
            ax[1].grid(alpha=0.3)

            ax[2].plot(f_f, np.abs(pbar_f_all[b]), "k", lw=1.5, label="D |p_bar|")
            ax[2].plot(f_m, np.abs(pbar_m_all[b]), "r--", lw=1.0, label="E |p_bar|")
            if b < len(a0_rows):
                af = a0_rows[b].get("A0_full_hz")
                am = a0_rows[b].get("A0_modal_hz")
                if af:
                    ax[2].axvline(af, color="k", ls=":", lw=0.8, label=f"A0 D {af:.0f} Hz")
                if am:
                    ax[2].axvline(am, color="r", ls=":", lw=0.8, label=f"A0 E {am:.0f} Hz")
            ax[2].set_ylabel("|p_bar| (mean cavity)")
            ax[2].set_xlabel("Frequency [Hz]")
            ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(plot_dir / f"modal_vs_full_bridge_{b:03d}.png"), dpi=150)
            plt.close(fig)
        print(f"[E vs D] per-bridge plots: {plot_dir}")
    except Exception as exc:
        print(f"[E vs D] plot skipped ({exc})")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        description="Modal-coupled (reduced) structure-air admittance - backend E "
                    "(reduced projection of the full coupled D system)")
    ap.add_argument("--air-msh", type=str, help="conformal mesh_air.msh")
    ap.add_argument("--bridge", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="Bridge coords [mm]")
    ap.add_argument("--bridge-points-json", type=str, default=None,
                    help="JSON list (or path) of [[x,y,z],...] in mm: multi-bridge")
    ap.add_argument("--basis", type=str, default="craig-bampton",
                    choices=("craig-bampton", "modal"),
                    help="craig-bampton (default): normal modes + attachment/residual "
                         "modes (restores the quasi-static compliance). "
                         "modal: normal modes only (ABLATION).")
    ap.add_argument("--n-attach-acoustic", type=int, default=12,
                    help="how many acoustic modes contribute FSI attachment vectors "
                         "K_s^-1 G^T psi_j (the constant-pressure mode is always #0)")
    ap.add_argument("--acoustic-attachment", type=str, default="none",
                    choices=("none", "port"),
                    help="Acoustic basis enrichment. port = add the quasi-static "
                         "near-hole field K_a^-1 b (bordered solve; fixes the A0 "
                         "modal-truncation bias: A0 came out 352/313 Hz vs D's 289 Hz). "
                         "DEFAULT none = previous behaviour, until the box run validates "
                         "port; then the default flips.")
    ap.add_argument("--static-solver", type=str, default="auto",
                    choices=("auto", "petsc", "scipy"),
                    help="factorization for the attachment static solves K_s X = F: "
                         "auto (PETSc/MUMPS, fall back to scipy), petsc, or scipy "
                         "(scipy splu runs out of memory on large meshes)")
    ap.add_argument("--material", type=str, default="engelmann_spruce")
    ap.add_argument("--material-json", type=str, default=None)
    ap.add_argument("--freq-min", type=float, default=20.0)
    ap.add_argument("--freq-max", type=float, default=800.0)
    ap.add_argument("--freq-points", type=int, default=200)
    ap.add_argument("--struct-fmax", type=float, default=3000.0,
                    help="keep structural modes up to this frequency [Hz]")
    ap.add_argument("--acoustic-fmax", type=float, default=3000.0,
                    help="keep acoustic modes up to this frequency [Hz]")
    ap.add_argument("--n-struct-modes", type=int, default=300)
    ap.add_argument("--n-acoustic-modes", type=int, default=60)
    ap.add_argument("--t-hole-mm", type=float, default=None)
    ap.add_argument("--rayleigh-alpha", type=float, default=0.0,
                    help="Rayleigh alpha (mass); overridden by material['alpha'] if present")
    ap.add_argument("--rayleigh-beta", type=float, default=5e-6,
                    help="Rayleigh beta (stiffness); overridden by material['beta'] if present")
    ap.add_argument("--port-end-corrections", type=int, default=1, choices=(1, 2),
                    help="soundhole end corrections in L_eff: 1 = exterior only (default, "
                         "matches D), 2 = both. MUST match D when comparing E vs D.")
    ap.add_argument("--output-dir", type=str, default="results/modal_coupled")
    ap.add_argument("--compare-full-dir", type=str, default=None,
                    help="D full-coupled results dir; compare E against it after the sweep")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip the solve; only compare --output-dir against --compare-full-dir")
    a = ap.parse_args()

    if a.compare_only:
        if not a.compare_full_dir:
            ap.error("--compare-only needs --compare-full-dir")
        compare_with_full_coupled(a.output_dir, a.compare_full_dir)
        return

    bridge_points = None
    if a.bridge_points_json:
        bridge_points = _load_json_arg(a.bridge_points_json)
    if not a.air_msh or (a.bridge is None and bridge_points is None):
        ap.error("--air-msh and (--bridge or --bridge-points-json) are required "
                 "(unless --compare-only)")

    if a.material_json:
        mat = _load_json_arg(a.material_json)
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from materials import get_material
        mat = get_material(a.material)

    try:
        compute_modal_coupled_admittance(
            msh_path=a.air_msh, material=mat,
            bridge_coords=tuple(a.bridge) if a.bridge else None,
            bridge_points=bridge_points,
            freq_min=a.freq_min, freq_max=a.freq_max, freq_points=a.freq_points,
            struct_fmax=a.struct_fmax, acoustic_fmax=a.acoustic_fmax,
            n_struct_modes=a.n_struct_modes, n_acoustic_modes=a.n_acoustic_modes,
            t_hole_mm=a.t_hole_mm, rayleigh_alpha=a.rayleigh_alpha,
            rayleigh_beta=a.rayleigh_beta,
            port_end_corrections=a.port_end_corrections,
            basis=a.basis, n_attach_acoustic=a.n_attach_acoustic,
            acoustic_attachment=a.acoustic_attachment,
            static_solver=a.static_solver,
            output_dir=a.output_dir)

        if a.compare_full_dir:
            compare_with_full_coupled(a.output_dir, a.compare_full_dir)
        _write_run_status(
            a.output_dir, "complete", comparison_completed=bool(a.compare_full_dir))
    except BaseException as exc:
        output_path = Path(a.output_dir)
        if output_path.exists():
            _write_run_status(
                output_path, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    _main()
