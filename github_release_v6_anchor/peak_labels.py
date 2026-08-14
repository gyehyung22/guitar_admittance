"""Peak-label utilities for the mixed admittance dataset.

The NN-facing target is magnitude-only, but the two FEM backends expose
different trustworthy information:

* solid: mass-normalized structural eigenmodes provide frequencies, damping and
  bridge residues; the full harmonic solver supplies the final peak amplitude.
* hollow: the frequency-dependent soundhole impedance makes a simple K/M pole
  solve invalid, so peaks and Q are measured from a dense reduced response.

This module keeps those extraction paths separate and shares only deterministic
selection, ordering and fixed-width padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


DB_FLOOR_MAGNITUDE = 1.0e-30
HALF_POWER_DB = 20.0 * np.log10(np.sqrt(2.0))
DEFAULT_TOP_K = 32
DEFAULT_PROMINENCE_DB = 3.0
DEFAULT_HOLLOW_SEARCH_POINTS = 4096


def magnitude_db(values, floor: float = DB_FLOOR_MAGNITUDE) -> np.ndarray:
    """Return ``20 log10(|values|)`` with a documented finite floor."""
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("magnitude input contains NaN or Inf")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("magnitude floor must be finite and positive")
    return 20.0 * np.log10(np.maximum(np.abs(values), float(floor)))


@dataclass(frozen=True)
class PeakBatch:
    frequency_hz: np.ndarray
    amplitude_db: np.ndarray
    q: np.ndarray
    mask: np.ndarray
    count_total: np.ndarray
    truncated: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "peak_frequency_hz": self.frequency_hz,
            "peak_amplitude_db": self.amplitude_db,
            "peak_q": self.q,
            "peak_mask": self.mask,
            "peak_count_total": self.count_total,
            "peak_truncated": self.truncated,
        }


def _as_bridge_matrix(values, n_candidates: int, name: str) -> np.ndarray:
    arr = np.asarray(values, float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n_candidates:
        raise ValueError(f"{name} must have shape (n_bridge, {n_candidates})")
    return arr


def pack_top_k_peaks(frequencies_hz, amplitudes_db, q_values, *,
                     top_k: int = DEFAULT_TOP_K, eligible=None) -> PeakBatch:
    """Select strongest peaks per bridge, then store them in frequency order.

    Invalid or ineligible candidates are excluded before ``count_total`` is
    computed. Padding is exact zero with ``mask=False`` so downstream masked
    losses cannot accidentally consume a sentinel as physical data.
    """
    f = np.asarray(frequencies_hz, float)
    if f.ndim != 1:
        raise ValueError("frequencies_hz must be one-dimensional")
    if isinstance(top_k, bool) or int(top_k) <= 0:
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    amp = _as_bridge_matrix(amplitudes_db, f.size, "amplitudes_db")
    q = _as_bridge_matrix(q_values, f.size, "q_values")
    if q.shape[0] == 1 and amp.shape[0] > 1:
        q = np.repeat(q, amp.shape[0], axis=0)
    if q.shape[0] != amp.shape[0]:
        raise ValueError("q_values bridge count does not match amplitudes_db")
    if eligible is None:
        allowed = np.ones_like(amp, dtype=bool)
    else:
        allowed = np.asarray(eligible, bool)
        if allowed.ndim == 1:
            allowed = allowed[None, :]
        if allowed.shape[0] == 1 and amp.shape[0] > 1:
            allowed = np.repeat(allowed, amp.shape[0], axis=0)
        if allowed.shape != amp.shape:
            raise ValueError("eligible must match amplitudes_db")

    valid_common = np.isfinite(f) & (f > 0.0)
    valid = (allowed & valid_common[None, :] & np.isfinite(amp)
             & np.isfinite(q) & (q > 0.0))
    n_bridge = amp.shape[0]
    out_f = np.zeros((n_bridge, top_k), float)
    out_a = np.zeros((n_bridge, top_k), float)
    out_q = np.zeros((n_bridge, top_k), float)
    out_m = np.zeros((n_bridge, top_k), bool)
    counts = np.sum(valid, axis=1, dtype=np.int64)

    for bridge in range(n_bridge):
        idx = np.flatnonzero(valid[bridge])
        if idx.size > top_k:
            strongest = np.argsort(amp[bridge, idx], kind="stable")[-top_k:]
            idx = idx[strongest]
        idx = idx[np.argsort(f[idx], kind="stable")]
        n = idx.size
        out_f[bridge, :n] = f[idx]
        out_a[bridge, :n] = amp[bridge, idx]
        out_q[bridge, :n] = q[bridge, idx]
        out_m[bridge, :n] = True

    return PeakBatch(out_f, out_a, out_q, out_m, counts,
                     counts > np.int64(top_k))


def group_structural_modes(frequencies_hz, zeta, bridge_residues,
                           *, rel_tol: float = 1.0e-6):
    """Combine degenerate structural modes into one observable resonance.

    Driving-point residues add within an exactly/near-degenerate eigenspace.
    Returning one grouped frequency prevents duplicate peak labels for arbitrary
    basis directions in the same physical eigenspace.
    """
    f = np.asarray(frequencies_hz, float)
    zeta = np.asarray(zeta, float)
    residues = np.asarray(bridge_residues, float)
    if residues.ndim == 1:
        residues = residues[None, :]
    if (f.ndim != 1 or zeta.shape != f.shape or residues.ndim != 2
            or residues.shape[1] != f.size):
        raise ValueError("incompatible structural-mode arrays")
    if (not np.all(np.isfinite(f)) or not np.all(np.isfinite(zeta))
            or not np.all(np.isfinite(residues)) or np.any(f <= 0.0)
            or np.any(zeta <= 0.0) or np.any(residues < 0.0)):
        raise ValueError("structural-mode arrays contain invalid values")
    if not np.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("rel_tol must be finite and non-negative")

    order = np.argsort(f, kind="stable")
    f, zeta, residues = f[order], zeta[order], residues[:, order]
    grouped_f: list[float] = []
    grouped_z: list[float] = []
    grouped_r: list[np.ndarray] = []
    start = 0
    for stop in range(1, f.size + 1):
        boundary = (stop == f.size or
                    abs(f[stop] - f[start]) > rel_tol * max(f[stop], f[start], 1.0))
        if boundary:
            sl = slice(start, stop)
            grouped_f.append(float(np.mean(f[sl])))
            grouped_z.append(float(np.mean(zeta[sl])))
            grouped_r.append(np.sum(residues[:, sl], axis=1))
            start = stop
    return (np.asarray(grouped_f), np.asarray(grouped_z),
            np.stack(grouped_r, axis=1))


def select_structural_candidate_union(frequencies_hz, zeta, bridge_residues,
                                      *, top_k: int = DEFAULT_TOP_K,
                                      multiplier: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact union of modes worth exact full-harmonic evaluation.

    The isolated-mode mobility maximum is ``residue/(2*zeta*omega)``.
    Each bridge contributes its strongest ``multiplier*top_k`` candidates; the
    union is evaluated with one batched full solve per frequency.
    """
    f = np.asarray(frequencies_hz, float)
    zeta = np.asarray(zeta, float)
    residues = np.asarray(bridge_residues, float)
    if residues.ndim == 1:
        residues = residues[None, :]
    if (zeta.shape != f.shape or residues.shape[1] != f.size
            or np.any(zeta <= 0.0) or np.any(residues < 0.0)
            or not np.all(np.isfinite(f)) or not np.all(np.isfinite(zeta))
            or not np.all(np.isfinite(residues))):
        raise ValueError("invalid structural peak candidates")
    n_pool = min(f.size, max(int(top_k), int(top_k) * int(multiplier)))
    omega = 2.0 * np.pi * f
    estimate = residues / (2.0 * zeta[None, :] * omega[None, :])
    eligible = residues > 0.0
    chosen = np.zeros_like(eligible)
    for bridge in range(residues.shape[0]):
        idx = np.flatnonzero(eligible[bridge])
        if idx.size > n_pool:
            idx = idx[np.argsort(estimate[bridge, idx], kind="stable")[-n_pool:]]
        chosen[bridge, idx] = True
    union = np.flatnonzero(np.any(chosen, axis=0))
    return union, chosen[:, union]


def structural_peak_labels(frequencies_hz, zeta, exact_admittance,
                           *, top_k: int = DEFAULT_TOP_K, eligible=None) -> PeakBatch:
    """Create solid labels from exact full response at grouped eigenfrequencies."""
    f = np.asarray(frequencies_hz, float)
    zeta = np.asarray(zeta, float)
    Y = np.asarray(exact_admittance)
    if Y.ndim == 1:
        Y = Y[None, :]
    if zeta.shape != f.shape or Y.ndim != 2 or Y.shape[1] != f.size:
        raise ValueError("incompatible solid peak arrays")
    q = 1.0 / (2.0 * zeta)
    return pack_top_k_peaks(f, magnitude_db(Y), q, top_k=top_k,
                            eligible=eligible)


def hollow_peak_search_grid(freq_min: float, freq_max: float, *,
                            n_points: int = DEFAULT_HOLLOW_SEARCH_POINTS,
                            hints_hz=()) -> np.ndarray:
    """Dense logarithmic search grid augmented with reduced-model mode hints."""
    if (not np.isfinite(freq_min) or not np.isfinite(freq_max)
            or freq_min <= 0.0 or freq_max <= freq_min or int(n_points) < 32):
        raise ValueError("invalid hollow peak-search grid")
    hints = np.asarray(hints_hz, float).ravel()
    hints = hints[np.isfinite(hints) & (hints >= freq_min) & (hints <= freq_max)]
    grid = np.geomspace(float(freq_min), float(freq_max), int(n_points))
    return np.unique(np.concatenate([grid, hints]))


def _parabolic_peak_log_frequency(freqs: np.ndarray, db: np.ndarray, index: int):
    if index <= 0 or index >= freqs.size - 1:
        return float(freqs[index]), float(db[index])
    x = np.log(freqs[index - 1:index + 2])
    y = db[index - 1:index + 2]
    try:
        a, b, c = np.polyfit(x, y, 2)
    except (ValueError, np.linalg.LinAlgError):
        return float(freqs[index]), float(db[index])
    if not np.isfinite(a) or a >= 0.0:
        return float(freqs[index]), float(db[index])
    xv = -b / (2.0 * a)
    if xv < x[0] or xv > x[-1]:
        return float(freqs[index]), float(db[index])
    return float(np.exp(xv)), float(a * xv * xv + b * xv + c)


def _crossing_log_frequency(freqs: np.ndarray, db: np.ndarray, i0: int,
                            i1: int, target_db: float) -> float | None:
    y0, y1 = float(db[i0]), float(db[i1])
    if not (np.isfinite(y0) and np.isfinite(y1)) or y0 == y1:
        return None
    fraction = (target_db - y0) / (y1 - y0)
    if fraction < 0.0 or fraction > 1.0:
        return None
    x0, x1 = np.log(float(freqs[i0])), np.log(float(freqs[i1]))
    return float(np.exp(x0 + fraction * (x1 - x0)))


def response_peak_labels(freqs_hz, admittance, *, top_k: int = DEFAULT_TOP_K,
                         prominence_db: float = DEFAULT_PROMINENCE_DB) -> PeakBatch:
    """Measure hollow response peaks and half-power Q on a dense search grid."""
    freqs = np.asarray(freqs_hz, float)
    Y = np.asarray(admittance)
    if Y.ndim == 1:
        Y = Y[None, :]
    if (freqs.ndim != 1 or freqs.size < 3 or Y.ndim != 2
            or Y.shape[1] != freqs.size or not np.all(np.isfinite(freqs))
            or np.any(freqs <= 0.0) or np.any(np.diff(freqs) <= 0.0)
            or not np.all(np.isfinite(Y)) or not np.isfinite(prominence_db)
            or prominence_db < 0.0):
        raise ValueError("invalid hollow response for peak extraction")
    db_all = magnitude_db(Y)
    candidates: list[list[tuple[float, float, float]]] = []
    for bridge, db in enumerate(db_all):
        peaks, _ = find_peaks(db, prominence=float(prominence_db))
        found: list[tuple[float, float, float]] = []
        for peak in peaks:
            f_peak, a_peak = _parabolic_peak_log_frequency(freqs, db, int(peak))
            target = a_peak - HALF_POWER_DB
            left = int(peak)
            while left > 0 and db[left] > target:
                left -= 1
            right = int(peak)
            while right < db.size - 1 and db[right] > target:
                right += 1
            if left == peak or right == peak or db[left] > target or db[right] > target:
                continue
            f_left = _crossing_log_frequency(freqs, db, left, left + 1, target)
            f_right = _crossing_log_frequency(freqs, db, right - 1, right, target)
            if f_left is None or f_right is None or f_right <= f_left:
                continue
            q = f_peak / (f_right - f_left)
            if np.isfinite(q) and q > 0.0:
                found.append((f_peak, a_peak, float(q)))
        candidates.append(found)

    n_bridge = Y.shape[0]
    max_candidates = max((len(row) for row in candidates), default=0)
    if max_candidates == 0:
        return PeakBatch(np.zeros((n_bridge, int(top_k))),
                         np.zeros((n_bridge, int(top_k))),
                         np.zeros((n_bridge, int(top_k))),
                         np.zeros((n_bridge, int(top_k)), bool),
                         np.zeros(n_bridge, np.int64),
                         np.zeros(n_bridge, bool))
    out_f = np.zeros((n_bridge, int(top_k)), float)
    out_a = np.zeros_like(out_f)
    out_q = np.zeros_like(out_f)
    out_m = np.zeros_like(out_f, bool)
    counts = np.asarray([len(row) for row in candidates], np.int64)
    for bridge, row in enumerate(candidates):
        if not row:
            continue
        vals = np.asarray(row, float)
        idx = np.arange(vals.shape[0])
        if idx.size > int(top_k):
            idx = idx[np.argsort(vals[:, 1], kind="stable")[-int(top_k):]]
        idx = idx[np.argsort(vals[idx, 0], kind="stable")]
        n = idx.size
        out_f[bridge, :n] = vals[idx, 0]
        out_a[bridge, :n] = vals[idx, 1]
        out_q[bridge, :n] = vals[idx, 2]
        out_m[bridge, :n] = True
    return PeakBatch(out_f, out_a, out_q, out_m, counts,
                     counts > np.int64(top_k))


def validate_peak_batch(batch: dict, n_bridges: int, top_k: int,
                        freq_min: float, freq_max: float,
                        min_count: int = 0) -> list[str]:
    """Fail-closed schema validation shared by worker/orchestrator tests."""
    reasons: list[str] = []
    expected = (int(n_bridges), int(top_k))
    arrays: dict[str, np.ndarray] = {}
    for key in ("peak_frequency_hz", "peak_amplitude_db", "peak_q", "peak_mask"):
        try:
            arrays[key] = np.asarray(batch[key])
        except Exception:
            reasons.append(f"missing or invalid {key}")
            continue
        if arrays[key].shape != expected:
            reasons.append(f"{key} shape {arrays[key].shape} != {expected}")
    if reasons:
        return reasons
    mask = arrays["peak_mask"]
    if mask.dtype.kind != "b":
        reasons.append("peak_mask is not boolean")
        mask = mask.astype(bool, copy=False)
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        reasons.append("peak_mask valid entries are not prefix packed")
    for key in ("peak_frequency_hz", "peak_amplitude_db", "peak_q"):
        arr = np.asarray(arrays[key], float)
        if not np.all(np.isfinite(arr)):
            reasons.append(f"non-finite {key}")
        if np.any(arr[~mask] != 0.0):
            reasons.append(f"{key} padding is not zero")
    f = np.asarray(arrays["peak_frequency_hz"], float)
    q = np.asarray(arrays["peak_q"], float)
    if np.any(mask & ((f < freq_min) | (f > freq_max))):
        reasons.append("peak frequency outside analysis band")
    if np.any(mask & (q <= 0.0)):
        reasons.append("peak Q is not positive")
    for row in range(expected[0]):
        vals = f[row, mask[row]]
        if np.any(np.diff(vals) <= 0.0):
            reasons.append(f"bridge {row} peaks are not strictly frequency ordered")
    try:
        count = np.asarray(batch["peak_count_total"])
        truncated = np.asarray(batch["peak_truncated"])
        if count.shape != (expected[0],) or count.dtype.kind not in "iu":
            reasons.append("peak_count_total has invalid shape or dtype")
        elif (np.any(count < np.sum(mask, axis=1))
              or np.any(count < int(min_count))):
            reasons.append(
                "peak_count_total is smaller than stored/minimum peak count")
        if truncated.shape != (expected[0],) or truncated.dtype.kind != "b":
            reasons.append("peak_truncated has invalid shape or dtype")
        elif count.shape == (expected[0],) and not np.array_equal(
                truncated, count > int(top_k)):
            reasons.append("peak_truncated disagrees with peak_count_total")
    except Exception:
        reasons.append("missing peak count metadata")
    return reasons
