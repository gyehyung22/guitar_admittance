"""
fenics_admittance_coupled.py
----------------------------
Full monolithic FEM-FEM structure-air coupled bridge admittance (pilot).

Solves the Everstine unsymmetric u-p block system (docs/air_coupling_theory.md),
exp(+iωt):

    ⎡ Z_s(ω)     -Gᵀ      ⎤ ⎡u⎤ = ⎡F⎤
    ⎣ -ρ0 ω² G    Z_a'(ω) ⎦ ⎣p⎦   ⎣0⎦

    Z_s = K_s - ω² M_s + iω C_s        (C_s = α M_s + β K_s, + weak spring on K_s)
    Z_a'= K_a - (ω²/c²) M_a + port(ω)  (soundhole rank-1 impedance port)
    G_{ji} = ∫_{Γ_sa} N_j^p (N_i^u·n_a) dS   (from fsi_coupling, restricted)

    Y_b(ω) = iω u_z(bridge) / F_bridge

Restriction: the structural block lives on the WOOD displacement dofs, the
acoustic block on the AIR pressure dofs (fsi_coupling._domain_scalar_dofs), so the
monolithic system has no inactive air-u / wood-p dofs (which would be singular).

This is a coarse PILOT (20-800 Hz, linear grid): it reuses the orthotropic tensor
from fenics_admittance and the acoustic/port machinery from air_acoustics /
acoustic_helmholtz.  FEniCSx imports are deferred so the module imports for static
checks where dolfinx is absent.  Heavy solve runs on the server.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fenics_admittance import engineering_to_C, _RIGID_FREQ_HZ
from acoustic_helmholtz import (
    AIR_C, AIR_RHO0, helmholtz_estimate, acoustic_inertance,
    effective_neck_length, radiation_resistance,
)
import fsi_coupling as fsi

PHYS_SOLID_WOOD = fsi.PHYS_SOLID_WOOD
PHYS_AIR_INTERNAL = fsi.PHYS_AIR_INTERNAL
PHYS_SOUNDHOLE = fsi.PHYS_SOUNDHOLE


# ---------------------------------------------------------------------------
# Subdomain assembly (structural on wood, acoustic on air)
# ---------------------------------------------------------------------------

def _to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())


def _parabolic_peak(freqs, mag):
    """Sub-grid peak of `mag(freqs)` by 3-point parabolic interpolation around the
    grid argmax.  Returns (f_peak, at_edge).  `at_edge` is True if the max is at a
    grid boundary (peak not bracketed) — then the grid frequency is returned."""
    freqs = np.asarray(freqs, float)
    mag = np.asarray(mag, float)
    i = int(np.argmax(mag))
    if i == 0 or i == len(mag) - 1:
        return float(freqs[i]), True
    ym1, y0, yp1 = mag[i - 1], mag[i], mag[i + 1]
    denom = ym1 - 2.0 * y0 + yp1
    if abs(denom) < 1e-30:
        return float(freqs[i]), False
    delta = 0.5 * (ym1 - yp1) / denom          # in grid-index units, |delta|<=0.5
    df = freqs[i + 1] - freqs[i]               # assumes near-uniform spacing
    return float(freqs[i] + delta * df), False


def _a0_peak_near_estimate(freqs, p_bar, f_est, low_factor=0.25, high_factor=1.75):
    """Find the Helmholtz/A0 peak near f_est, not the global pressure maximum.

    Broad sweeps can contain much larger higher acoustic/structural pressure
    peaks.  Using the global |p_bar| maximum then mislabels those as A0.  For A0
    reporting, search near the analytical Helmholtz estimate and report the
    global pressure peak separately as a diagnostic.

    `low_factor` is 0.25, NOT 0.45: f_est is the RIGID-WALL Helmholtz frequency, and
    compliant walls push the coupled A0 well below it (guitar shape_0005: A0 ~ 40 Hz
    vs f_est = 90.8 Hz, i.e. 0.44 f_est — the old 0.45 window put the true peak just
    OUTSIDE the search range and silently returned the window edge as "A0").  The
    at_edge flag is still returned and must be treated as a failed detection.
    """
    freqs = np.asarray(freqs, float)
    mag = np.abs(np.asarray(p_bar))
    if freqs.size == 0:
        raise ValueError("empty frequency grid")
    lo = max(float(freqs.min()), low_factor * float(f_est))
    hi = min(float(freqs.max()), high_factor * float(f_est), 1000.0)
    mask = (freqs >= lo) & (freqs <= hi)
    if np.count_nonzero(mask) < 3:
        mask = np.ones_like(freqs, dtype=bool)
        lo, hi = float(freqs.min()), float(freqs.max())
    f_win = freqs[mask]
    mag_win = mag[mask]
    f_a0, edge = _parabolic_peak(f_win, mag_win)
    f_grid = float(f_win[int(np.argmax(mag_win))])
    f_global = float(freqs[int(np.argmax(mag))])
    return f_a0, f_grid, edge, [float(lo), float(hi)], f_global


def _detect_a0_legacy(freqs, p_bar, U_h, f_est, low_factor=0.25, high_factor=1.75,
                      min_points_per_peak: int = 5) -> dict:
    """A0 detection by LOCAL PEAK, cross-checked between p_bar and U_h.

    Replaces the old "argmax inside a window" logic, which failed silently on the real
    guitar: the true A0 (~40 Hz, compliant walls) fell just outside the 0.45*f_est
    window, so the window's EDGE value (49.9 Hz) was reported as A0 for both D and E.

    Rules (all reported, none hidden):
      * search a local maximum of |p_bar| inside [low*f_est, high*f_est];
      * a monotone window (no interior local max) => DETECTION FAILED, A0 is None;
      * the same peak must appear in |U_h| (soundhole volume velocity) — A0 is the
        cavity/port breathing mode, so if the two disagree by > 5 % the detection is
        flagged as unreliable;
      * the grid must resolve the peak (>= `min_points_per_peak` points across it),
        otherwise the located frequency is not trustworthy.

    `f_est` is the RIGID-WALL Helmholtz estimate; it is only used to place the search
    window and is reported SEPARATELY from the observed coupled A0 (they are different
    physical quantities — compliant walls shift the coupled A0 well below f_est).
    """
    freqs = np.asarray(freqs, float)
    mag_p = np.abs(np.asarray(p_bar))
    mag_u = np.abs(np.asarray(U_h)) if U_h is not None else None

    lo = max(float(freqs.min()), low_factor * float(f_est))
    hi = min(float(freqs.max()), high_factor * float(f_est))
    mask = (freqs >= lo) & (freqs <= hi)
    idx = np.flatnonzero(mask)

    def _local_peaks(mag, idx):
        """Interior local maxima of `mag` restricted to `idx`, lowest frequency first."""
        out = []
        for i in idx:
            if i == 0 or i == len(mag) - 1:
                continue                       # boundary: cannot be a bracketed peak
            if mag[i] > mag[i - 1] and mag[i] >= mag[i + 1]:
                # prominence proxy: height above the higher of the two local minima
                out.append((float(mag[i] - max(mag[i - 1], mag[i + 1])), int(i)))
        return sorted(out, key=lambda item: freqs[item[1]])

    peaks_p = _local_peaks(mag_p, idx)
    res = {
        "A0_search_window_hz": [float(lo), float(hi)],
        "f_H_rigid_wall_hz": float(f_est),         # NOT the coupled A0 — separate item
        "global_pbar_peak_hz": float(freqs[int(np.argmax(mag_p))]),
        "grid_spacing_hz": float(np.median(np.diff(freqs))) if freqs.size > 1 else None,
    }

    if not peaks_p:
        res.update({
            "A0_detected": False,
            "A0_observed_hz": None,
            "A0_failure": "no interior local maximum of |p_bar| inside the search "
                          "window (the window is monotone). A0 may lie outside the "
                          "window or below freq_min; run a dedicated low-band sweep.",
        })
        print(f"[A0] DETECTION FAILED: no local |p_bar| peak in "
              f"[{lo:.1f}, {hi:.1f}] Hz. Not reporting an A0 value.")
        return res

    i_p = peaks_p[0][1]
    f_p, _ = _parabolic_peak(freqs[i_p - 1:i_p + 2], mag_p[i_p - 1:i_p + 2])
    res["A0_observed_hz"] = float(f_p)
    res["A0_observed_grid_hz"] = float(freqs[i_p])
    res["A0_detected"] = True
    res["n_candidate_peaks_in_window"] = len(peaks_p)

    # Cross-check against the soundhole volume velocity (the port breathing).
    if mag_u is not None:
        peaks_u = _local_peaks(mag_u, idx)
        if peaks_u:
            i_u = peaks_u[0][1]
            f_u, _ = _parabolic_peak(freqs[i_u - 1:i_u + 2], mag_u[i_u - 1:i_u + 2])
            res["A0_from_U_h_hz"] = float(f_u)
            dev = abs(f_u - f_p) / max(f_p, 1e-9)
            res["A0_pbar_vs_Uh_rel_diff"] = float(dev)
            if dev > 0.05:
                res["A0_detected"] = False
                # Preserve the rejected pressure peak as a diagnostic only.  Once
                # the port-flow cross-check fails there is no accepted observed A0.
                res["A0_candidate_hz"] = float(f_p)
                res["A0_observed_hz"] = None
                res["A0_failure"] = (f"|p_bar| peak ({f_p:.1f} Hz) and |U_h| peak "
                                     f"({f_u:.1f} Hz) disagree by {dev*100:.1f} % — "
                                     f"the low-frequency feature is not a clean "
                                     f"cavity/port (A0) resonance.")
                print(f"[A0] WARNING: p_bar and U_h peaks disagree "
                      f"({f_p:.1f} vs {f_u:.1f} Hz) - A0 unreliable.")
        else:
            res["A0_from_U_h_hz"] = None

    # Resolution: A0 is narrow; a coarse grid cannot locate it.
    df = res["grid_spacing_hz"] or float("nan")
    rel = df / max(f_p, 1e-9)
    res["spacing_rel_to_A0"] = float(rel)
    res["A0_resolution_ok"] = bool(rel <= 1.0 / min_points_per_peak)
    if not res["A0_resolution_ok"]:
        print(f"[A0] WARNING: grid spacing {df:.2f} Hz is {rel*100:.0f} % of A0 "
              f"({f_p:.1f} Hz) - too coarse to LOCATE A0. Run a dedicated low-band "
              f"sweep (e.g. 20-300 Hz) before quoting an A0 number.")
    res["rel_error_vs_f_H"] = float(abs(f_p - f_est) / max(f_est, 1e-9))
    label = "observed (coupled)" if res["A0_detected"] else "rejected candidate"
    print(f"[A0] {label} {f_p:.1f} Hz  |  rigid-wall f_H estimate "
          f"{f_est:.1f} Hz  ({res['rel_error_vs_f_H']*100:.1f} % apart; these are "
          f"DIFFERENT quantities - compliant walls shift A0)")
    return res


def detect_a0(freqs, p_bar, U_h, f_est, low_factor=0.25, high_factor=1.75,
              min_points_per_peak: int = 5,
              min_prominence_db: float = 3.0) -> dict:
    """Detect a resolved, prominent joint cavity-pressure/port-flow A0 peak.

    A candidate is accepted only when both signals have >= ``min_prominence_db``
    prominence, their interpolated peak frequencies agree within 5%, and the
    measured -3 dB FWHM contains at least ``min_points_per_peak`` grid intervals
    in both signals.  Rejected candidates are diagnostic-only: observed A0 is
    always ``None`` when ``A0_detected`` is false.
    """
    from scipy.signal import find_peaks

    freqs = np.asarray(freqs, float)
    mag_p = np.abs(np.asarray(p_bar))
    mag_u = np.abs(np.asarray(U_h)) if U_h is not None else None
    if (freqs.ndim != 1 or freqs.size < 3 or mag_p.shape != freqs.shape
            or mag_u is None or mag_u.shape != freqs.shape
            or not np.all(np.isfinite(freqs + mag_p + mag_u))
            or freqs[0] <= 0.0 or np.any(np.diff(freqs) <= 0.0)
            or not np.isfinite(float(f_est)) or float(f_est) <= 0.0
            or min_points_per_peak < 1
            or not np.isfinite(float(min_prominence_db))
            or min_prominence_db <= 0.0):
        raise ValueError("invalid A0 detector inputs")

    lo = max(float(freqs.min()), low_factor * float(f_est))
    hi = min(float(freqs.max()), high_factor * float(f_est))
    idx = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    result = {
        "A0_search_window_hz": [float(lo), float(hi)],
        "f_H_rigid_wall_hz": float(f_est),
        "global_pbar_peak_hz": float(freqs[int(np.argmax(mag_p))]),
        "grid_spacing_hz": float(np.median(np.diff(freqs))),
        "A0_min_prominence_db": float(min_prominence_db),
    }
    if idx.size < 3:
        result.update(A0_detected=False, A0_observed_hz=None,
                      A0_from_U_h_hz=None, A0_resolution_ok=False,
                      A0_failure="fewer than three grid points in A0 search window")
        return result

    def candidates(mag):
        db = 20.0 * np.log10(mag[idx] + 1e-300)
        local, props = find_peaks(db, prominence=float(min_prominence_db))
        return [{"idx": int(idx[j]), "prominence_db": float(prom)}
                for j, prom in zip(local, props["prominences"])]

    peaks_p = candidates(mag_p)
    peaks_u = candidates(mag_u)
    result["n_candidate_pbar_peaks_in_window"] = len(peaks_p)
    result["n_candidate_U_h_peaks_in_window"] = len(peaks_u)
    if not peaks_p or not peaks_u:
        missing = "|p_bar|" if not peaks_p else "|U_h|"
        result.update(
            A0_detected=False, A0_observed_hz=None, A0_from_U_h_hz=None,
            A0_resolution_ok=False,
            A0_failure=(f"no >= {min_prominence_db:g} dB prominent {missing} "
                        "peak inside the A0 search window"))
        print("[A0] DETECTION FAILED: no joint prominent p_bar/U_h peak.")
        return result

    pairs = []
    for pp in peaks_p:
        for pu in peaks_u:
            rel = abs(freqs[pp["idx"]] - freqs[pu["idx"]]) / freqs[pp["idx"]]
            if rel <= 0.05:
                pairs.append((min(pp["prominence_db"], pu["prominence_db"]),
                              -rel, pp, pu))
    if not pairs:
        strongest_p = max(peaks_p, key=lambda item: item["prominence_db"])
        candidate_idx = strongest_p["idx"]
        candidate_hz, _ = _parabolic_peak(
            freqs[candidate_idx - 1:candidate_idx + 2],
            mag_p[candidate_idx - 1:candidate_idx + 2])
        result.update(
            A0_detected=False, A0_observed_hz=None, A0_from_U_h_hz=None,
            A0_candidate_hz=float(candidate_hz), A0_resolution_ok=False,
            A0_failure="prominent |p_bar| and |U_h| peaks do not agree within 5%")
        print("[A0] DETECTION FAILED: p_bar/U_h prominent peaks do not match.")
        return result

    _score, _neg_rel, selected_p, selected_u = max(
        pairs, key=lambda item: (item[0], item[1], -freqs[item[2]["idx"]]))
    i_p, i_u = selected_p["idx"], selected_u["idx"]
    f_p, _ = _parabolic_peak(freqs[i_p - 1:i_p + 2], mag_p[i_p - 1:i_p + 2])
    f_u, _ = _parabolic_peak(freqs[i_u - 1:i_u + 2], mag_u[i_u - 1:i_u + 2])
    dev = abs(f_u - f_p) / max(f_p, 1e-9)
    result.update({
        "A0_observed_grid_hz": float(freqs[i_p]),
        "A0_from_U_h_hz": float(f_u),
        "A0_pbar_vs_Uh_rel_diff": float(dev),
        "A0_pbar_prominence_db": selected_p["prominence_db"],
        "A0_U_h_prominence_db": selected_u["prominence_db"],
    })

    def fwhm(mag, peak_idx):
        db = 20.0 * np.log10(mag + 1e-300)
        threshold = db[peak_idx] - 3.01029995664
        left = peak_idx - 1
        while left >= 0 and db[left] > threshold:
            left -= 1
        right = peak_idx + 1
        while right < db.size and db[right] > threshold:
            right += 1
        if left < 0 or right >= db.size:
            return {"resolved": False, "fwhm_hz": None,
                    "fwhm_left_hz": None, "fwhm_right_hz": None,
                    "n_intervals_across_fwhm": 0,
                    "max_spacing_in_fwhm_hz": None}

        def crossing(i0, i1):
            d0, d1 = db[i0], db[i1]
            if d1 == d0:
                return float((freqs[i0] + freqs[i1]) * 0.5)
            fraction = (threshold - d0) / (d1 - d0)
            return float(freqs[i0] + fraction * (freqs[i1] - freqs[i0]))

        f_left = crossing(left, left + 1)
        f_right = crossing(right - 1, right)
        overlap = (freqs[:-1] < f_right) & (freqs[1:] > f_left)
        n_intervals = int(np.count_nonzero(overlap))
        max_spacing = (float(np.max(np.diff(freqs)[overlap]))
                       if n_intervals else None)
        width = float(f_right - f_left)
        resolved = bool(n_intervals >= min_points_per_peak
                        and max_spacing is not None
                        and max_spacing <= width / min_points_per_peak)
        return {"resolved": resolved, "fwhm_hz": width,
                "fwhm_left_hz": f_left, "fwhm_right_hz": f_right,
                "n_intervals_across_fwhm": n_intervals,
                "max_spacing_in_fwhm_hz": max_spacing}

    width_p = fwhm(mag_p, i_p)
    width_u = fwhm(mag_u, i_u)
    result.update({f"A0_pbar_{key}": value for key, value in width_p.items()
                   if key != "resolved"})
    result.update({f"A0_U_h_{key}": value for key, value in width_u.items()
                   if key != "resolved"})
    result["A0_resolution_ok"] = bool(
        width_p["resolved"] and width_u["resolved"])
    result["rel_error_vs_f_H"] = float(
        abs(f_p - f_est) / max(float(f_est), 1e-9))
    result["A0_detected"] = bool(dev <= 0.05 and result["A0_resolution_ok"])
    if result["A0_detected"]:
        result["A0_observed_hz"] = float(f_p)
        print(f"[A0] observed (coupled) {f_p:.2f} Hz; "
              f"FWHM {width_p['fwhm_hz']:.3g} Hz is resolved.")
    else:
        result["A0_candidate_hz"] = float(f_p)
        result["A0_observed_hz"] = None
        result["A0_failure"] = (
            f"joint A0 candidate is not resolved by {min_points_per_peak} "
            "grid intervals across the -3 dB FWHM in both p_bar and U_h")
        print("[A0] DETECTION FAILED: joint peak is under-resolved by FWHM.")
    return result


def a0_resolution_warning(freqs, f_a0, edge, min_points_per_peak: int = 5):
    """Warn when the frequency grid is too coarse to LOCATE A0 (or the peak fell on a
    window edge).  A0 is narrow; on the guitar run D's grid was ~10 Hz while A0 sits at
    ~40 Hz, so the peak was covered by 1-2 points and its position is meaningless.

    Returns a dict describing the resolution, and prints a warning if it is inadequate.
    """
    freqs = np.asarray(freqs, float)
    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else float("nan")
    rel = df / max(f_a0, 1e-9)
    ok = bool((not edge) and rel <= 1.0 / min_points_per_peak)
    if edge:
        print(f"[A0] WARNING: the A0 peak sits on the SEARCH-WINDOW EDGE - detection "
              f"FAILED, do not report {f_a0:.1f} Hz as A0. Widen the band/window.")
    if rel > 1.0 / min_points_per_peak:
        print(f"[A0] WARNING: grid spacing {df:.2f} Hz is {rel*100:.0f} % of A0 "
              f"({f_a0:.1f} Hz). A0 is a narrow resonance; run a dedicated LOW-BAND "
              f"sweep (e.g. 20-300 Hz) before quoting an A0 number.")
    return {"grid_spacing_hz": df, "spacing_rel_to_A0": rel,
            "A0_resolution_ok": ok, "A0_peak_at_grid_edge": bool(edge)}


def assemble_structural_wood(mesh, cell_tags, V_u, C_pa, density):
    """K_s, M_s (scipy CSR, full-space) integrated over the WOOD subdomain only."""
    import dolfinx.fem as fem
    import dolfinx.fem.petsc as petsc_fem
    import ufl

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)(PHYS_SOLID_WOOD)
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    def eps_voigt(w):
        e = ufl.sym(ufl.grad(w))
        return ufl.as_vector([e[0, 0], e[1, 1], e[2, 2],
                              2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])

    from petsc4py import PETSc
    C_ufl = fem.Constant(mesh, PETSc.ScalarType(C_pa))
    rho = fem.Constant(mesh, PETSc.ScalarType(density))
    K = petsc_fem.assemble_matrix(fem.form(
        ufl.inner(ufl.dot(C_ufl, eps_voigt(u)), eps_voigt(v)) * dx)); K.assemble()
    M = petsc_fem.assemble_matrix(fem.form(
        rho * ufl.inner(u, v) * dx)); M.assemble()
    return _to_scipy(K), _to_scipy(M)


def assemble_acoustic_air(mesh, cell_tags, V_p):
    """K_a = ∫∇p·∇q, M_a = ∫ p q (scipy CSR, full-space) over the AIR subdomain."""
    import dolfinx.fem as fem
    import dolfinx.fem.petsc as petsc_fem
    import ufl

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)(PHYS_AIR_INTERNAL)
    p = ufl.TrialFunction(V_p)
    q = ufl.TestFunction(V_p)
    K = petsc_fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(p), ufl.grad(q)) * dx)); K.assemble()
    M = petsc_fem.assemble_matrix(fem.form(ufl.inner(p, q) * dx)); M.assemble()
    return _to_scipy(K), _to_scipy(M)


def soundhole_port_vector_full(mesh, facet_tags, V_p):
    """b_j = ∫_{Γ_h} N_j dS on the FULL acoustic space (scipy vector), + area S."""
    import dolfinx.fem as fem
    import dolfinx.fem.petsc as petsc_fem
    import ufl
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)(PHYS_SOUNDHOLE)
    q = ufl.TestFunction(V_p)
    bvec = petsc_fem.assemble_vector(fem.form(ufl.conj(q) * ds))
    bvec.assemble()
    b = bvec.getArray().real.copy()
    return b, float(b.sum())


# ---------------------------------------------------------------------------
# Bridge dof (wood u_z nearest node)
# ---------------------------------------------------------------------------

def _as_bridge_list(bridge_coords):
    """Accept a single (x,y,z) or a list of (x,y,z) -> always a list of points [mm]."""
    arr = np.asarray(bridge_coords, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"bridge_coords must be (x,y,z); got shape {arr.shape}")
        return [arr]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [row for row in arr]
    raise ValueError(f"bridge_coords must be (3,) or (N,3); got shape {arr.shape}")


def _load_json_arg(value):
    """Load a CLI JSON argument that may be either a JSON string or a file path.

    Check JSON-looking strings first.  Otherwise a long JSON list can be treated as
    a path and Path.exists() raises "File name too long" before json.loads runs.
    """
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


def _bridge_wood_dof(mesh, V_u, bridge_xyz_mm, wood_u_dofs):
    """Global scalar u_z dof at the WOOD node nearest the bridge (mm coords), its
    index within the restricted wood_u_dofs ordering, the snap distance AND the
    snapped node coordinates [mm].

    The snapped coordinates are returned from HERE, where the block-vs-scalar layout
    of `tabulate_dof_coordinates()` is already handled — callers must not re-derive
    them by indexing the raw dof-coordinate array (that silently gives the wrong node
    when the array is scalar-expanded).  The nearest-node search is restricted to wood
    blocks so an air-region node can never be picked.
    """
    bs = V_u.dofmap.index_map_bs
    coords = V_u.tabulate_dof_coordinates()[:, :3]           # mesh is in metres
    n_blocks = V_u.dofmap.index_map.size_local + V_u.dofmap.index_map.num_ghosts
    if len(coords) == n_blocks * bs:
        coords = coords[::bs]                                # scalar-expanded -> blocks
    wood_blocks = np.unique(wood_u_dofs // bs)               # blocks on wood
    bridge_m = np.asarray(bridge_xyz_mm, float) * 1e-3
    d2 = np.sum((coords[wood_blocks] - bridge_m) ** 2, axis=1)
    imin = int(np.argmin(d2))
    block = int(wood_blocks[imin])
    uz_global = block * bs + 2                               # z component
    snap_mm = float(np.sqrt(d2[imin])) * 1e3
    snapped_mm = (coords[block] * 1e3).astype(float)         # metres -> mm
    pos = np.where(wood_u_dofs == uz_global)[0]
    if len(pos) == 0:
        raise RuntimeError("Bridge u_z dof not in wood restriction (bridge not on wood?)")
    return uz_global, int(pos[0]), snap_mm, snapped_mm


# ---------------------------------------------------------------------------
# Linear-solver backends for the monolithic block (scipy splu | PETSc LU/MUMPS)
#
# The block A(w) = [[Zs, -G.T], [-rho0 w^2 G, Za]] is COMPLEX, NONSYMMETRIC and
# indefinite (Everstine u-p).  Only general LU is valid — no Cholesky/CG/GHEP
# assumptions anywhere on this matrix.  Both backends expose the same interface:
#
#     lu = factorize_block(A, linear_solver)     # one factorization per frequency
#     X  = lu.solve(B)                           # B: (n,) or (n, k) multi-RHS
#     lu.destroy()
#
# Multi-RHS is what makes bridge batching cheap: at a given frequency the matrix
# does not depend on the bridge point, so ONE factorization serves all bridge RHS
# vectors plus the Sherman-Morrison port vector.
# ---------------------------------------------------------------------------

class _ScipyLU:
    """scipy.sparse.linalg.splu — the original (reference) path."""

    backend = "scipy"

    def __init__(self, A):
        self.lu = spla.splu(A.tocsc())
        self.factor_solver = "superlu (scipy splu)"

    def solve(self, B):
        return self.lu.solve(np.asarray(B, dtype=complex))

    def destroy(self):
        self.lu = None


class _PetscLU:
    """PETSc KSP(preonly) + PC(lu) with MUMPS, for the same complex nonsymmetric block.

    Requires a COMPLEX PETSc build (as used by the complex dolfinx env) — a real
    build cannot represent this matrix and we fail loudly rather than silently
    dropping the imaginary part.
    """

    backend = "petsc"

    def __init__(self, A):
        from petsc4py import PETSc
        if not np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise RuntimeError(
                "PETSc is built with REAL scalars; the coupled block is complex. "
                "Use --linear-solver scipy, or run in the complex FEniCSx env.")
        self.PETSc = PETSc
        A = A.tocsr()
        self.mat = PETSc.Mat().createAIJ(
            size=A.shape,
            csr=(A.indptr.astype(PETSc.IntType),
                 A.indices.astype(PETSc.IntType),
                 A.data.astype(PETSc.ScalarType)))
        self.mat.assemble()

        ksp = PETSc.KSP().create()
        ksp.setOperators(self.mat)
        ksp.setType("preonly")                    # direct solve, no Krylov
        pc = ksp.getPC()
        pc.setType("lu")                          # general LU: matrix is nonsymmetric
        self.factor_solver = "petsc default"
        for solver in ("mumps", "superlu_dist", "superlu"):
            try:
                pc.setFactorSolverType(solver)
                self.factor_solver = solver
                break
            except Exception:
                continue
        ksp.setFromOptions()
        ksp.setUp()                               # numeric factorization happens here
        self.ksp = ksp
        self._x = self.mat.createVecRight()
        self._b = self.mat.createVecLeft()

    def solve(self, B):
        B = np.asarray(B, dtype=complex)
        one_d = (B.ndim == 1)
        Bm = B.reshape(-1, 1) if one_d else B
        out = np.zeros(Bm.shape, dtype=complex)
        for j in range(Bm.shape[1]):
            self._b.setArray(np.ascontiguousarray(Bm[:, j]))
            self.ksp.solve(self._b, self._x)
            reason = self.ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(f"PETSc KSP failed (converged reason {reason}) — "
                                   f"the coupled block may be singular.")
            out[:, j] = self._x.getArray().copy()
        return out[:, 0] if one_d else out

    def destroy(self):
        self._x.destroy(); self._b.destroy()
        self.ksp.destroy(); self.mat.destroy()


def factorize_block(A, linear_solver: str = "scipy"):
    """Factorize the monolithic block with the requested backend ('scipy'|'petsc')."""
    if linear_solver == "scipy":
        return _ScipyLU(A)
    if linear_solver == "petsc":
        return _PetscLU(A)
    raise ValueError(f"unknown linear_solver '{linear_solver}' (use scipy|petsc)")


# ---------------------------------------------------------------------------
# Block builder (shared by the FULL coupled solve and the modal-coupled backend)
# ---------------------------------------------------------------------------

def build_coupled_blocks(
    msh_path,
    material: dict,
    bridge_coords,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
    c: float = AIR_C,
    rho0: float = AIR_RHO0,
    t_hole_mm: float = None,
    port_end_corrections: int = 1,
) -> dict:
    """Assemble the frequency-INDEPENDENT pieces of the Everstine u-p system.

    This is the single source of truth for the coupled operators: the full
    monolithic solve (`compute_coupled_admittance`) and the reduced modal-coupled
    backend (`modal_coupled_admittance.py`) both call it, so they cannot drift
    apart in restriction, weak spring, port area or G.

    Returns (all restricted to active wood-u / air-p dofs):
        Ks (with weak spring), Ms, Ka, Ma  : scipy CSC
        G (n_p, n_u), b (n_p,), S          : coupling, port vector, port area
        cavity_volume, est, M_h            : Helmholtz quantities
        uz_r / uz_global / snap_mm         : bridge u_z dof (restricted / global)
        n_u, n_p, weak_k, timing
    """
    timing = {}

    # 1. Mesh + spaces + restriction ----------------------------------------
    t0 = time.time()
    mesh, cell_tags, facet_tags = fsi.load_coupled_mesh(msh_path)
    mesh.geometry.x[:] *= 1e-3                                   # mm -> m
    V_u, V_p = fsi.build_spaces(mesh)
    wood_u_dofs = fsi._domain_scalar_dofs(V_u, cell_tags.find(PHYS_SOLID_WOOD))
    air_p_dofs = fsi._domain_scalar_dofs(V_p, cell_tags.find(PHYS_AIR_INTERNAL))
    if len(wood_u_dofs) == 0:
        raise RuntimeError("No active SOLID_WOOD displacement dofs found.")
    if len(air_p_dofs) == 0:
        raise RuntimeError("No active AIR_INTERNAL pressure dofs found.")
    timing["mesh"] = time.time() - t0

    # 2. Blocks (frequency-independent) -------------------------------------
    t0 = time.time()
    C_pa = engineering_to_C(material)
    Ks_f, Ms_f = assemble_structural_wood(mesh, cell_tags, V_u, C_pa, material["density"])
    Ka_f, Ma_f = assemble_acoustic_air(mesh, cell_tags, V_p)
    b_f, S = soundhole_port_vector_full(mesh, facet_tags, V_p)
    G_full = fsi.assemble_G_custom(mesh, cell_tags, facet_tags, V_u, V_p)
    timing["assembly"] = time.time() - t0

    # Restrict to active dofs
    Ks = Ks_f[wood_u_dofs][:, wood_u_dofs].tocsc()
    Ms = Ms_f[wood_u_dofs][:, wood_u_dofs].tocsc()
    Ka = Ka_f[air_p_dofs][:, air_p_dofs].tocsc()
    Ma = Ma_f[air_p_dofs][:, air_p_dofs].tocsc()
    G = G_full.tocsr()[air_p_dofs][:, wood_u_dofs].tocsc()       # (n_p, n_u)
    b = b_f[air_p_dofs]
    n_u, n_p = Ks.shape[0], Ka.shape[0]
    if S <= 0.0 or not np.isfinite(S):
        raise RuntimeError(f"Invalid soundhole port area S={S}; check SOUNDHOLE tagging.")
    if G.nnz == 0:
        raise RuntimeError("Restricted FSI coupling matrix G has zero nnz.")
    if np.count_nonzero(b) == 0:
        raise RuntimeError("Soundhole port vector is zero; check SOUNDHOLE facets.")
    print(f"[coupled] n_u(wood)={n_u}  n_p(air)={n_p}  G {G.shape} nnz={G.nnz}  S={S*1e4:.3f} cm^2")

    # Weak spring for the free-free structure (rigid modes at rigid_freq_hz).
    # NOTE: in the RESTRICTED wood ordering the z-dofs are NOT at index 2::3; they
    # are the positions where wood_u_dofs % 3 == 2 (the unroll makes global dof
    # %3 == component).  Compute the total mass with that mask, not the default
    # weak_spring_for_rigid_freq (which assumes full block-major ordering).
    zmask = (wood_u_dofs % 3 == 2).astype(float)
    m_total = float(np.real(zmask @ (Ms @ zmask)))
    n_nodes = len(wood_u_dofs) // 3
    weak_k = (2.0 * np.pi * rigid_freq_hz) ** 2 * m_total / max(n_nodes, 1)
    if m_total <= 0.0 or not np.isfinite(m_total):
        raise RuntimeError(f"Invalid restricted wood mass M_total={m_total}.")
    print(f"[coupled] weak spring: target {rigid_freq_hz:.3g} Hz -> k={weak_k:.4g} N/m "
          f"(M_total={m_total:.4f} kg, N_nodes={n_nodes})")
    Ks = Ks + weak_k * sp.identity(n_u, format="csc")

    # Cavity volume + Helmholtz estimate + port inertance (Ma is complex in the
    # complex build; the volume ∫1 dΩ = 1ᵀ M_a 1 is real -> take the real part).
    cavity_volume = float(np.real(np.ones(n_p) @ (Ma @ np.ones(n_p))))
    if t_hole_mm is None:
        meta = Path(msh_path).parent / "air_mesh_meta.json"
        t_hole_mm = (json.loads(meta.read_text()).get("top_plate_thickness_mm", 3.0)
                     if meta.exists() else 3.0)
    t_hole = t_hole_mm * 1e-3
    est = helmholtz_estimate(cavity_volume, S, t_hole, c, rho0)
    L_eff = effective_neck_length(t_hole, S, n_open_ends=port_end_corrections)
    M_h = acoustic_inertance(L_eff, S, rho0)
    print(f"[coupled] cavity V={cavity_volume*1e3:.3f} L  f_H(est)={est.estimated_helmholtz_hz:.1f} Hz  "
          f"M_h={M_h:.4g}")

    # Bridge dof(s).  `bridge_coords` may be a single (x,y,z) or a list of points
    # (bridge batching): the block A(w) does not depend on the bridge, so one
    # factorization per frequency serves every bridge RHS.
    pts = _as_bridge_list(bridge_coords)
    bridges = []
    for p in pts:
        g, r, snap, snapped_mm = _bridge_wood_dof(mesh, V_u, p, wood_u_dofs)
        bridges.append({
            "bridge_requested_xyz": [float(v) for v in p],
            "bridge_snapped_xyz": [float(v) for v in snapped_mm],
            "snap_distance_mm": float(snap),
            "uz_global": int(g), "uz_r": int(r),
            "response_at": "nearest_wood_mesh_node_to_requested_bridge",
        })
        print(f"[coupled] bridge {np.round(p, 2).tolist()} -> u_z global={g} "
              f"restricted={r} snap={snap:.2f} mm")

    return {
        "Ks": Ks, "Ms": Ms, "Ka": Ka, "Ma": Ma, "G": G, "b": b, "S": S,
        "n_u": n_u, "n_p": n_p, "weak_k": weak_k,
        "cavity_volume": cavity_volume, "est": est, "M_h": M_h,
        "L_eff": L_eff, "t_hole_mm": t_hole_mm,
        # single-bridge keys kept for backward compatibility (= first bridge)
        "uz_global": bridges[0]["uz_global"], "uz_r": bridges[0]["uz_r"],
        "snap_mm": bridges[0]["snap_distance_mm"],
        "bridges": bridges,
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# Coupled solve
# ---------------------------------------------------------------------------

def compute_coupled_admittance(
    msh_path,
    material: dict,
    bridge_coords,
    freq_min: float = 20.0,
    freq_max: float = 800.0,
    freq_points: int = 60,
    output_dir="results/coupled",
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 5e-6,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
    c: float = AIR_C,
    rho0: float = AIR_RHO0,
    t_hole_mm: float = None,
    port_end_corrections: int = 1,
    force_n: float = 1.0,
    linear_solver: str = "scipy",
    bridge_points=None,
    freqs=None,
) -> dict:
    """Monolithic structure-air coupled bridge admittance over [freq_min, freq_max].

    `linear_solver`: 'scipy' (splu, the original reference path, DEFAULT — unchanged)
    or 'petsc' (KSP preonly + PC lu + MUMPS).  Both factorize the SAME block and use
    the SAME Sherman-Morrison port update, so they must agree to solver tolerance.

    `bridge_points`: optional list of [x,y,z] (mm).  The block A(w) is independent of
    the bridge, so all bridge RHS (plus the port vector) are solved from ONE
    factorization per frequency.  When given, it overrides `bridge_coords`.

    Returns dict with freqs, Y (coupled), p_bar (mean cavity pressure), U_h
    (soundhole volume velocity), and A0 (observed vs Helmholtz estimate).  For a
    single bridge the arrays are 1-D (backward compatible); for several bridges the
    per-bridge arrays are (n_bridges, n_freq) under the *_multi keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pts = bridge_points if bridge_points is not None else bridge_coords

    # 1.-2. Mesh, restriction, blocks (shared with the modal-coupled backend) --
    blk = build_coupled_blocks(
        msh_path, material, pts, rigid_freq_hz=rigid_freq_hz,
        c=c, rho0=rho0, t_hole_mm=t_hole_mm,
        port_end_corrections=port_end_corrections)
    timing = blk["timing"]
    Ks, Ms, Ka, Ma = blk["Ks"], blk["Ms"], blk["Ka"], blk["Ma"]
    G, b, S = blk["G"], blk["b"], blk["S"]
    n_u, n_p = blk["n_u"], blk["n_p"]
    weak_k = blk["weak_k"]
    cavity_volume, est, M_h = blk["cavity_volume"], blk["est"], blk["M_h"]
    uz_global, uz_r, snap_mm = blk["uz_global"], blk["uz_r"], blk["snap_mm"]
    bridges = blk["bridges"]
    n_b = len(bridges)
    dof_list = [br["uz_r"] for br in bridges]

    alpha = material.get("alpha", rayleigh_alpha)
    beta = material.get("beta", rayleigh_beta)

    # 3. Frequency sweep -----------------------------------------------------
    # An explicit grid is required by validation/dataset callers: regenerating a
    # rounded persisted linspace can move narrow resonance samples.  The legacy
    # min/max/count API remains unchanged when `freqs` is omitted.
    if freqs is not None:
        freqs = np.asarray(freqs, dtype=float)
        if (freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs))
                or freqs[0] <= 0.0 or np.any(np.diff(freqs) <= 0.0)):
            raise ValueError(
                "explicit `freqs` must be finite, positive, and strictly increasing")
        freq_min, freq_max, freq_points = (
            float(freqs[0]), float(freqs[-1]), int(freqs.size))
    else:
        freqs = np.linspace(freq_min, freq_max, freq_points)
    Y_m = np.zeros((n_b, freqs.size), dtype=complex)
    p_bar_m = np.zeros((n_b, freqs.size), dtype=complex)
    U_h_m = np.zeros((n_b, freqs.size), dtype=complex)

    Gt = G.transpose().tocsc()
    per_freq_s = np.zeros(freqs.size)      # per-frequency solve wall time (QC)
    per_freq_fact_s = np.zeros(freqs.size)  # factorization-only time
    factor_solver_used = None
    t0 = time.time()
    for i, f in enumerate(freqs):
        tf = time.time()
        w = 2.0 * np.pi * f
        Zs = (Ks * (1.0 + 1j * w * beta)) - (w ** 2) * Ms + (1j * w * alpha) * Ms
        # acoustic block + rank-1 soundhole impedance port term
        R = radiation_resistance(w, S, c, rho0)
        Z_h = R + 1j * w * M_h
        kappa = 1j * w * rho0 / (S ** 2 * Z_h)
        Za = (Ka - (w ** 2 / c ** 2) * Ma).tocsc()
        # block system  [[Zs, -Gt],[-rho0 w^2 G, Za + kappa b b^T]]
        # rank-1 port handled via Sherman-Morrison on the acoustic block is awkward
        # inside the coupled solve; assemble it as an explicit low-rank update.
        A = sp.bmat([[Zs, -Gt],
                     [(-rho0 * w ** 2) * G, Za]], format="csc")
        # Solve with the rank-1 port via Sherman-Morrison:
        #   (A + kappa * bb bb^T) x = rhs, where bb acts only on the p-block.
        # NOTE: plain transpose, NOT conjugate transpose — matches the D formulation.
        bb = np.zeros(n_u + n_p, dtype=complex)
        bb[n_u:] = b

        # One factorization per frequency; multi-RHS = [bridge forces | bb]
        tfac = time.time()
        lu = factorize_block(A, linear_solver)
        per_freq_fact_s[i] = time.time() - tfac
        factor_solver_used = lu.factor_solver
        RHS = np.zeros((n_u + n_p, n_b + 1), dtype=complex)
        for ib, dof in enumerate(dof_list):
            RHS[dof, ib] = force_n
        RHS[:, n_b] = bb
        X = lu.solve(RHS)
        lu.destroy()
        x0_all, y0 = X[:, :n_b], X[:, n_b]

        denom = 1.0 + kappa * (bb @ y0)
        if abs(denom) < 1e-12:
            print(f"[coupled] WARNING: small soundhole-port Sherman-Morrison denominator "
                  f"at {f:.2f} Hz: {denom:.3e}")
        for ib in range(n_b):
            x0 = x0_all[:, ib]
            x = x0 - kappa * y0 * (bb @ x0) / denom
            Y_m[ib, i] = 1j * w * x[dof_list[ib]] / force_n
            p_bar_m[ib, i] = (b @ x[n_u:]) / S
            U_h_m[ib, i] = p_bar_m[ib, i] / Z_h
        per_freq_s[i] = time.time() - tf
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{freqs.size}] {f:6.1f} Hz  |Y|={abs(Y_m[0, i]):.3e}  "
                  f"|p_bar|={abs(p_bar_m[0, i]):.3e}  ({per_freq_s[i]:.1f}s, "
                  f"fact {per_freq_fact_s[i]:.1f}s)")
    timing["solve"] = time.time() - t0
    timing["per_freq_mean_s"] = float(per_freq_s.mean())
    timing["per_freq_min_s"] = float(per_freq_s.min())
    timing["per_freq_max_s"] = float(per_freq_s.max())
    timing["per_freq_factorize_mean_s"] = float(per_freq_fact_s.mean())
    timing["linear_solver"] = linear_solver
    timing["factor_solver"] = factor_solver_used
    timing["n_bridges"] = n_b

    # First bridge = the backward-compatible 1-D result.
    Y, p_bar, U_h = Y_m[0], p_bar_m[0], U_h_m[0]

    # Finite-response check: FAIL FAST (raise before saving) so a singular /
    # unstable coupled solve never leaves "successful"-looking output files that
    # could contaminate validation data.
    nonfinite = {name: int(np.sum(~np.isfinite(arr)))
                 for name, arr in (("Y", Y_m), ("p_bar", p_bar_m), ("U_h", U_h_m))}
    if any(nonfinite.values()):
        raise RuntimeError(
            f"Coupled solve produced non-finite values {nonfinite} — unstable / "
            f"singular block (check restriction, weak spring, port). No output "
            f"saved.")

    # A0: local-peak detection on |p_bar|, cross-checked with |U_h|.  Never an argmax
    # over a window (that silently returned the window edge on the real guitar).
    a0 = detect_a0(freqs, p_bar, U_h, est.estimated_helmholtz_hz)
    f_a0 = a0["A0_observed_hz"] if a0["A0_detected"] else float("nan")
    rel = a0.get("rel_error_vs_f_H", float("nan"))
    print(f"[coupled] solve {timing['solve']:.1f}s")

    # 4. Save ----------------------------------------------------------------
    np.savez(str(output_dir / "admittance_air_full_coupled.npz"),
             frequencies=freqs, admittance=Y)
    np.savez(str(output_dir / "pressure_mean_cavity.npz"), frequencies=freqs, p_bar=p_bar)
    np.savez(str(output_dir / "soundhole_volume_velocity.npz"), frequencies=freqs, U_h=U_h)
    # Multi-bridge arrays (always written; identical content for n_b = 1).
    np.savez(str(output_dir / "admittance_air_full_coupled_multi.npz"),
             frequencies=freqs, admittance=Y_m, p_bar=p_bar_m, U_h=U_h_m,
             bridge_requested_xyz=np.asarray(
                 [br["bridge_requested_xyz"] for br in bridges], float),
             bridge_snapped_xyz=np.asarray(
                 [br["bridge_snapped_xyz"] for br in bridges], float),
             snap_distance_mm=np.asarray(
                 [br["snap_distance_mm"] for br in bridges], float),
             bridge_uz_restricted_dof=np.asarray(dof_list, int),
             linear_solver=np.asarray(linear_solver))
    (output_dir / "A0_estimated_vs_observed.json").write_text(json.dumps({
        # The rigid-wall Helmholtz estimate and the observed COUPLED A0 are different
        # physical quantities (compliant walls shift A0) — reported separately, and the
        # observed value is only present when the detection actually succeeded.
        **a0,
        "A0_estimated_hz": est.estimated_helmholtz_hz,   # = f_H_rigid_wall_hz (legacy key)
        "rel_error": rel,                                # legacy key: |A0 - f_H| / f_H
        "cavity_volume_m3": cavity_volume, "soundhole_area_m2": S,
        "port_end_corrections": port_end_corrections,
        "n_freq_points": int(freq_points),
    }, indent=2))
    (output_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (output_dir / "coupled_metadata.json").write_text(json.dumps({
        "formulation": "Everstine unsymmetric u-p",
        "time_convention": "exp(+i omega t)",
        "block_system": "[[Zs, -G.T], [-rho0*omega^2*G, Za_port]]",
        "n_wood_u_dofs": int(n_u),
        "n_air_p_dofs": int(n_p),
        "G_restricted_shape": [int(G.shape[0]), int(G.shape[1])],
        "G_restricted_nnz": int(G.nnz),
        "soundhole_area_m2": float(S),
        "cavity_volume_m3": float(cavity_volume),
        "weak_spring_k_N_per_m": float(weak_k),
        "rigid_freq_hz": float(rigid_freq_hz),
        "bridge_requested_mm": [float(v) for v in bridges[0]["bridge_requested_xyz"]],
        "bridge_uz_global_dof": int(uz_global),
        "bridge_uz_restricted_dof": int(uz_r),
        "bridge_snap_distance_mm": float(snap_mm),
        "port_end_corrections": int(port_end_corrections),
        "linear_solver": linear_solver,
        "factor_solver": factor_solver_used,
        "n_bridges": int(n_b),
        "bridges": bridges,
        "sherman_morrison": "x = x0 - kappa*y0*(bb@x0)/(1 + kappa*(bb@y0)); "
                            "plain transpose, no conjugation",
    }, indent=2))
    _plot_coupled(output_dir / "admittance.png", freqs, Y, p_bar, U_h,
                  est.estimated_helmholtz_hz, f_a0,
                  title="Coupled bridge admittance - bridge 0")
    _plot_all_bridge_coupled(
        output_dir / "bridge_plots", freqs, Y_m, p_bar_m, U_h_m,
        est.estimated_helmholtz_hz)
    print(f"[coupled] saved -> {output_dir}  "
          f"({linear_solver}/{factor_solver_used}, {n_b} bridge(s), "
          f"{timing['per_freq_mean_s']:.1f}s/freq)")

    return {"freqs": freqs, "Y": Y, "p_bar": p_bar, "U_h": U_h,
            "Y_multi": Y_m, "p_bar_multi": p_bar_m, "U_h_multi": U_h_m,
            "bridges": bridges,
            "A0_observed": f_a0, "A0_detected": a0["A0_detected"], "A0": a0,
            "A0_estimated": est.estimated_helmholtz_hz,
            "nonfinite": nonfinite, "timing": timing}


def _plot_coupled(png_path, freqs, Y, p_bar, U_h, f_H_est, f_a0, title=None):
    """Debug plot: coupled |Y|, mean cavity |p_bar|, soundhole |U_h|."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[coupled] plot skipped ({exc})")
        return
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(freqs, 20 * np.log10(np.abs(Y) + 1e-30), color="k")
    ax[0].set_ylabel("|Y| [dB re 1 m/s/N]")
    ax[0].set_title(title or "Coupled bridge admittance")
    ax[1].plot(freqs, np.abs(p_bar), color="C0")
    ax[1].axvline(f_H_est, color="g", ls="--", lw=0.8, label=f"f_H est {f_H_est:.0f}")
    if np.isfinite(f_a0):
        ax[1].axvline(f_a0, color="r", ls=":", lw=0.8, label=f"A0 obs {f_a0:.0f}")
    else:
        ax[1].set_title("A0 DETECTION FAILED (no local p_bar peak in the window)",
                        fontsize=9, color="r")
    ax[1].set_ylabel("|p_bar| (mean cavity)"); ax[1].legend(fontsize=8)
    ax[2].plot(freqs, np.abs(U_h), color="C3")
    ax[2].set_ylabel("|U_h| (soundhole vol. vel.)"); ax[2].set_xlabel("Frequency [Hz]")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(str(png_path), dpi=150)
    plt.close(fig)
    print(f"[coupled] plot: {png_path}")


def _plot_all_bridge_coupled(out_dir, freqs, Y_m, p_bar_m, U_h_m, f_H_est):
    """Write one coupled-response plot per bridge point."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_b = int(np.atleast_2d(Y_m).shape[0])
    for ib in range(n_b):
        a0 = detect_a0(freqs, p_bar_m[ib], U_h_m[ib], f_H_est)
        f_a0 = a0["A0_observed_hz"] if a0["A0_detected"] else float("nan")
        _plot_coupled(
            out_dir / f"admittance_bridge_{ib:03d}.png",
            freqs, Y_m[ib], p_bar_m[ib], U_h_m[ib], f_H_est, f_a0,
            title=f"Coupled bridge admittance - bridge {ib}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Monolithic structure-air coupled admittance (pilot)")
    ap.add_argument("--air-msh", type=str, required=True, help="conformal mesh_air.msh")
    ap.add_argument("--bridge", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Bridge coords [mm] (single bridge)")
    ap.add_argument("--bridge-points-json", type=str, default=None,
                    help="JSON list (or path to one) of [[x,y,z],...] in mm: bridge "
                         "BATCHING - one factorization per frequency serves all bridges")
    ap.add_argument("--linear-solver", type=str, default="scipy", choices=("scipy", "petsc"),
                    help="monolithic block solver: scipy splu (default, reference path) "
                         "or petsc (KSP preonly + PC lu + MUMPS). Both must agree.")
    ap.add_argument("--material", type=str, default="engelmann_spruce")
    ap.add_argument("--material-json", type=str, default=None)
    ap.add_argument("--freq-min", type=float, default=20.0)
    ap.add_argument("--freq-max", type=float, default=800.0)
    ap.add_argument("--freq-points", type=int, default=60)
    ap.add_argument("--t-hole-mm", type=float, default=None)
    ap.add_argument("--rayleigh-alpha", type=float, default=0.0,
                    help="Rayleigh alpha (mass); overridden by material['alpha'] if present")
    ap.add_argument("--rayleigh-beta", type=float, default=5e-6,
                    help="Rayleigh beta (stiffness); overridden by material['beta'] if present")
    ap.add_argument("--port-end-corrections", type=int, default=1, choices=(1, 2),
                    help="soundhole end corrections in L_eff: 1 = exterior only (default, "
                         "the 3-D cavity already carries the interior added mass), 2 = both")
    ap.add_argument("--output-dir", type=str, default="results/coupled")
    a = ap.parse_args()

    if a.material_json:
        mat = _load_json_arg(a.material_json)
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from materials import get_material
        mat = get_material(a.material)

    bridge_points = None
    if a.bridge_points_json:
        bridge_points = _load_json_arg(a.bridge_points_json)
    if bridge_points is None and a.bridge is None:
        ap.error("provide --bridge X Y Z or --bridge-points-json")

    compute_coupled_admittance(
        msh_path=a.air_msh, material=mat,
        bridge_coords=tuple(a.bridge) if a.bridge else None,
        bridge_points=bridge_points,
        freq_min=a.freq_min, freq_max=a.freq_max, freq_points=a.freq_points,
        t_hole_mm=a.t_hole_mm, rayleigh_alpha=a.rayleigh_alpha,
        rayleigh_beta=a.rayleigh_beta,
        port_end_corrections=a.port_end_corrections,
        linear_solver=a.linear_solver,
        output_dir=a.output_dir)


if __name__ == "__main__":
    _main()
