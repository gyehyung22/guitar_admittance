"""
modal_eigenbasis.py
-------------------
Shared, FEM-free helpers to recover a REAL, M-orthonormal eigenbasis from the
complex eigenvectors a complex PETSc/SLEPc build returns for a real-symmetric
GHEP  K φ = λ M φ.

Why this exists: SLEPc (complex build) returns a real eigenmode r up to an
arbitrary global phase, v = e^{iθ} r.  Taking Re(v)/sqrt(vᴴMv) scales the mode by
cos²θ (lost entirely at θ→π/2), so ΦᵀMΦ ≠ I and every downstream assumption of
M-orthonormality (modal participation, attachment deflation, reduced-mass
identity) is corrupted.  These helpers recover a real M-orthonormal basis
degeneracy-safely (per eigenvalue cluster, with a small Rayleigh–Ritz solve), and
are used by BOTH the air-coupled modal solver and the structural (solid) modal
solver so the two share one audited implementation.

Two entry points:
  * `realify_eigenbasis(lams, vecs_c, M, K)` — returns the FULL real basis Phi
    (n_dof x m).  Convenient but allocates all eigenvectors.
  * `realify_bridge_participation(lams, fetch_vec, M, K, dof_indices)` — STREAMING:
    processes one eigenvalue cluster at a time, keeps only the requested DOF rows,
    and discards each cluster's full vectors immediately.  Peak memory is bounded
    by the largest degenerate cluster (usually 1–3 vectors), not by n_modes.  This
    is what the large-mesh solid/hollow bulk path uses.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Normwise backward error the realified basis must meet before it is accepted.
# 1e-4 is still an excellent basis for these ~1e4-1e5 DOF GHEPs (near the
# double-precision floor for SLEPc at a practical tol); the earlier 1e-6 was
# physically over-tight and ABORTED otherwise-good solves (observed 1.1e-6-1.3e-6
# on production geometry).  Kept in lockstep with dataset_gen_mixed.QC_EIGEN_RESIDUAL_TOL.
DEFAULT_RESIDUAL_TOL = 1e-4

# Relative eigenvalue tolerance for CLUSTERING near-degenerate modes into one group
# before realification.  Each group is M-orthonormalised INTERNALLY (enforced to
# <1e-6), but modes split across DIFFERENT groups rely on their SLEPc eigenvectors
# already being M-orthogonal — which fails for the dense near-degenerate spectra of
# thin (3 mm) top plates, giving max|Phi^T M Phi - I| ~ 1e-3.  A looser tolerance
# groups those near-twins together so they are jointly re-orthonormalised.  Widening
# a cluster is always CORRECTNESS-safe (a Rayleigh-Ritz solve re-diagonalises it);
# it only makes the per-cluster work slightly larger.  1e-6 was over-tight.
DEFAULT_GROUP_RTOL = 1e-2


def symmetry_residual(A) -> float:
    """max|A - Aᵀ| / max|A| — a cheap guard that the GHEP assumption holds."""
    A = sp.csr_matrix(A)
    if A.nnz and not np.all(np.isfinite(A.data)):
        return float("inf")
    num = abs(A - A.transpose()).max() if A.nnz else 0.0
    den = abs(A).max() if A.nnz else 1.0
    out = float(num / (den + 1e-300))
    return out if np.isfinite(out) else float("inf")


def eigenvalue_group_slices(lams, group_rtol: float):
    """Yield slices of an ascending eigenvalue array that form one spectral group."""
    i = 0
    while i < len(lams):
        j = i + 1
        while j < len(lams) and (lams[j] - lams[i]) <= group_rtol * max(
                abs(lams[i]), abs(lams[j]), 1.0):
            j += 1
        yield slice(i, j)
        i = j


def realify_eigen_group(lams, vecs_c, M, K=None, operator_scales=None,
                        drop_tol: float = 1e-8, residual_tol: float = DEFAULT_RESIDUAL_TOL):
    """Convert one degenerate eigenvalue group to a real M-orthonormal basis.

    Exactly one real direction is returned per complex input eigenpair.  This cap is
    important: without it, a tiny orthogonal imaginary round-off component can be
    normalized into a spurious extra mode.  A small Rayleigh-Ritz solve restores the
    individual eigenpairs when a group contains close but non-identical eigenvalues.
    """
    lams = np.asarray(lams, float)
    if len(lams) == 0 or len(lams) != len(vecs_c):
        raise ValueError("eigenvalue group and vector list must have the same non-zero length")
    if not np.all(np.isfinite(lams)):
        raise ValueError("eigenvalue group contains non-finite values")
    if not np.all(np.isfinite(M.data)) or (K is not None and not np.all(np.isfinite(K.data))):
        raise ValueError("eigen operators contain non-finite values")
    target_rank = len(lams)

    def mnorm(v):
        q = float(v @ (M @ v))
        if not np.isfinite(q):
            raise RuntimeError("non-finite M norm during eigenbasis realification")
        if q < -1e-12 * max(float(v @ v), 1.0):
            raise RuntimeError(f"negative M norm during eigenbasis realification ({q:g})")
        return float(np.sqrt(max(q, 0.0)))

    cands = []
    im_ratio = 0.0
    for vec in vecs_c:
        vec = np.asarray(vec)
        re = np.ascontiguousarray(vec.real, dtype=float)
        im = np.ascontiguousarray(vec.imag, dtype=float)
        nre, nim = mnorm(re), mnorm(im)
        im_ratio = max(im_ratio, nim / (nre + 1e-300))
        cands.extend((re, im))

    norms = [mnorm(v) for v in cands]
    max_n = max(norms, default=0.0)
    if max_n <= 0.0:
        raise RuntimeError("complex eigenvalue group has no non-zero real direction")

    kept_v, kept_Mv = [], []
    n_small = n_dependent = n_rank_capped = 0
    candidate_order = sorted(range(len(cands)), key=lambda idx: -norms[idx])
    for pos, idx in enumerate(candidate_order):
        if len(kept_v) == target_rank:
            n_rank_capped = len(candidate_order) - pos
            break
        if norms[idx] < drop_tol * max_n:
            n_small += 1
            continue
        v = cands[idx].copy()
        for _ in range(2):
            for u, Mu in zip(kept_v, kept_Mv):
                v -= float(Mu @ v) * u
        Mv = M @ v
        nrm = float(np.sqrt(max(float(v @ Mv), 0.0)))
        if nrm < drop_tol * max_n:
            n_dependent += 1
            continue
        kept_v.append(v / nrm)
        kept_Mv.append(Mv / nrm)

    if len(kept_v) != target_rank:
        raise RuntimeError(
            f"real eigenbasis rank {len(kept_v)} does not match the {target_rank} "
            "complex eigenpairs in its spectral group")

    U = np.column_stack(kept_v)
    MU = np.column_stack(kept_Mv)
    max_residual = None
    if K is not None:
        if operator_scales is None:
            operator_scales = (float(np.linalg.norm(K.data)),
                               float(np.linalg.norm(M.data)))
        k_scale, m_scale = operator_scales
        KU = K @ U
        K_small = U.T @ KU
        K_small = 0.5 * (K_small + K_small.T)
        lam_out, rotation = np.linalg.eigh(K_small)
        Phi = U @ rotation
        MPhi = MU @ rotation
        KPhi = KU @ rotation
        residuals = []
        for col, lam in enumerate(lam_out):
            num = np.linalg.norm(KPhi[:, col] - lam * MPhi[:, col])
            # Normwise backward error.  Unlike ||Kphi||+|lam|||Mphi||, this remains
            # meaningful for an exact zero mode where both terms nearly vanish.
            den = ((k_scale + abs(lam) * m_scale) * np.linalg.norm(Phi[:, col])
                   + 1e-300)
            residuals.append(float(num / den))
        max_residual = max(residuals, default=0.0)
        if not np.isfinite(max_residual) or max_residual > residual_tol:
            raise RuntimeError(
                f"realified eigenbasis residual {max_residual:.2e} exceeds "
                f"tolerance {residual_tol:.1e}; refusing to promote numerical noise")
    else:
        Phi = U
        MPhi = MU
        lam_out = np.full(target_rank, float(np.mean(lams)))

    gram_dev = float(np.max(np.abs(Phi.T @ MPhi - np.eye(target_rank))))
    if not np.isfinite(gram_dev) or gram_dev > 1e-6:
        raise RuntimeError(
            f"real eigenvalue group is not M-orthonormal (max deviation {gram_dev:.2e})")
    return np.asarray(lam_out, float), Phi, {
        "n_input_pairs": int(target_rank),
        "n_output_modes": int(Phi.shape[1]),
        "n_small_candidates": int(n_small),
        "n_dependent_candidates": int(n_dependent),
        "n_rank_capped_candidates": int(n_rank_capped),
        "max_imag_to_real_mnorm": float(im_ratio),
        "mass_orthonormality_max_dev": gram_dev,
        "max_eigen_residual": max_residual,
    }


def _aggregate_group_diags(group_diags, extra=None):
    residual_values = [d["max_eigen_residual"] for d in group_diags
                       if d["max_eigen_residual"] is not None]
    out = {
        "n_input_pairs": int(sum(d["n_input_pairs"] for d in group_diags)),
        "n_output_modes": int(sum(d["n_output_modes"] for d in group_diags)),
        "n_eigenvalue_groups": int(len(group_diags)),
        "n_small_candidates": int(sum(d["n_small_candidates"] for d in group_diags)),
        "n_dependent_candidates": int(sum(d["n_dependent_candidates"] for d in group_diags)),
        "n_rank_capped_candidates": int(sum(d["n_rank_capped_candidates"] for d in group_diags)),
        "max_imag_to_real_mnorm": float(max(
            (d["max_imag_to_real_mnorm"] for d in group_diags), default=0.0)),
        "mass_orthonormality_max_dev": float(max(
            (d["mass_orthonormality_max_dev"] for d in group_diags), default=0.0)),
        "max_eigen_residual": (float(max(residual_values)) if residual_values else None),
    }
    if extra:
        out.update(extra)
    return out


def realify_eigenbasis(lams, vecs_c, M, K=None, group_rtol: float = DEFAULT_GROUP_RTOL,
                       drop_tol: float = 1e-8, residual_tol: float = DEFAULT_RESIDUAL_TOL):
    """Recover the FULL real M-orthonormal basis (n_dof x m).  Allocates all
    eigenvectors — use `realify_bridge_participation` for the large-mesh path."""
    M = sp.csr_matrix(M).real
    K = sp.csr_matrix(K).real if K is not None else None
    operator_scales = ((float(np.linalg.norm(K.data)), float(np.linalg.norm(M.data)))
                       if K is not None else None)
    lams = np.asarray(lams, float)
    if len(lams) != len(vecs_c):
        raise ValueError("eigenvalue and eigenvector counts differ")
    if not np.all(np.isfinite(lams)):
        raise ValueError("eigenvalues contain non-finite values")
    if not np.all(np.isfinite(M.data)) or (K is not None and not np.all(np.isfinite(K.data))):
        raise ValueError("eigen operators contain non-finite values")
    order = np.argsort(lams)
    lams = lams[order]
    vecs_c = [vecs_c[int(i)] for i in order]

    lam_parts, phi_parts, group_diags = [], [], []
    for group in eigenvalue_group_slices(lams, group_rtol):
        lam_g, phi_g, diag_g = realify_eigen_group(
            lams[group], vecs_c[group.start:group.stop], M, K=K,
            operator_scales=operator_scales, drop_tol=drop_tol, residual_tol=residual_tol)
        lam_parts.append(lam_g)
        phi_parts.append(phi_g)
        group_diags.append(diag_g)

    if not phi_parts:
        raise RuntimeError("realification produced an empty eigenbasis")
    lam_out = np.concatenate(lam_parts)
    Phi = np.column_stack(phi_parts)
    order = np.argsort(lam_out)
    lam_out, Phi = lam_out[order], Phi[:, order]
    gram = Phi.T @ (M @ Phi)
    gram_dev = float(np.max(np.abs(gram - np.eye(Phi.shape[1]))))
    if not np.isfinite(gram_dev) or gram_dev > 1e-6:
        raise RuntimeError(
            f"real eigenbasis is not globally M-orthonormal (max deviation {gram_dev:.2e})")
    diag = _aggregate_group_diags(group_diags,
                                  {"mass_orthonormality_max_dev": gram_dev})
    return lam_out, Phi, diag


def realify_bridge_participation(lams, fetch_vec, M, K, dof_indices,
                                 group_rtol: float = DEFAULT_GROUP_RTOL, drop_tol: float = 1e-8,
                                 residual_tol: float = DEFAULT_RESIDUAL_TOL):
    """STREAMING realification that keeps only `dof_indices` rows of the basis.

    Parameters
    ----------
    lams        : (n_conv,) eigenvalues in the ORIGINAL solver order.
    fetch_vec   : callable(i) -> full complex eigenvector for original index i.
                  Called once per vector, cluster by cluster; the returned array is
                  not retained past its cluster, so peak memory is bounded by the
                  largest degenerate cluster.
    M, K        : scipy sparse (K includes the weak spring).  K enables Rayleigh-Ritz
                  refinement + eigen-residual check per cluster.
    dof_indices : DOF rows to keep (e.g. the bridge Z-DOFs).

    Returns (lam_out ascending, participation (n_dof_kept, m), diagnostics).

    Cross-cluster M-orthogonality is a mathematical property of a real GHEP with
    distinct eigenvalues and is enforced per cluster (Rayleigh-Ritz residual), so no
    global Gram over the full basis is needed — which is exactly what lets this avoid
    materialising the full Phi.
    """
    M = sp.csr_matrix(M).real
    K = sp.csr_matrix(K).real
    operator_scales = (float(np.linalg.norm(K.data)), float(np.linalg.norm(M.data)))
    dof_arr = np.asarray(dof_indices, dtype=int)
    lams = np.asarray(lams, float)
    if not np.all(np.isfinite(lams)):
        raise ValueError("eigenvalues contain non-finite values")
    if not np.all(np.isfinite(M.data)) or not np.all(np.isfinite(K.data)):
        raise ValueError("eigen operators contain non-finite values")
    order = np.argsort(lams)
    lams_sorted = lams[order]

    lam_parts, phi_dof_parts, group_diags = [], [], []
    for group in eigenvalue_group_slices(lams_sorted, group_rtol):
        orig_idx = order[group.start:group.stop]
        vecs = [np.asarray(fetch_vec(int(i))) for i in orig_idx]
        lam_g, Phi_g, diag_g = realify_eigen_group(
            lams_sorted[group], vecs, M, K=K, operator_scales=operator_scales,
            drop_tol=drop_tol, residual_tol=residual_tol)
        lam_parts.append(lam_g)
        phi_dof_parts.append(np.ascontiguousarray(Phi_g[dof_arr, :]))   # keep DOF rows
        group_diags.append(diag_g)
        del vecs, Phi_g                                                 # free cluster

    if not phi_dof_parts:
        raise RuntimeError("realification produced an empty eigenbasis")
    lam_out = np.concatenate(lam_parts)
    phi_dofs = np.column_stack(phi_dof_parts)          # (n_dof_kept, m)
    order2 = np.argsort(lam_out)
    lam_out = lam_out[order2]
    phi_dofs = phi_dofs[:, order2]
    if not np.all(np.isfinite(lam_out)) or not np.all(np.isfinite(phi_dofs)):
        raise RuntimeError("streaming realification produced non-finite output")
    diag = _aggregate_group_diags(group_diags)
    return lam_out, phi_dofs, diag
