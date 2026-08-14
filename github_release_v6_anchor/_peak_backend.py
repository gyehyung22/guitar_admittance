"""NumPy-only replacements for the two SciPy calls the metrics depend on.

The server's ``fenicsx`` environment cannot import ``scipy.signal``: its
``pypocketfft`` extension is linked against a newer ``libstdc++`` than the one
the dynamic loader finds (``CXXABI_1.3.15 not found``).  That breaks
``find_peaks`` and, through the same import chain, ``linear_sum_assignment``.

Every resonance number in this project now flows through one metric module, so
a broken SciPy blocks the entire analysis on the one machine that has the data.
These fallbacks remove that dependency.  They are not approximations: the
prominence, distance and width rules follow SciPy's own algorithms, and
``test_peak_backend.py`` asserts index-for-index agreement with SciPy on
thousands of random curves wherever SciPy does import.

Set ``ELMER_FORCE_NUMPY_PEAKS=1`` to exercise the fallback on a machine where
SciPy works -- that is how the agreement test runs both sides.
"""
from __future__ import annotations

import os

import numpy as np


def _scipy_available() -> bool:
    if os.environ.get("ELMER_FORCE_NUMPY_PEAKS"):
        return False
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
        from scipy.signal import find_peaks  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# peak detection
# ---------------------------------------------------------------------------

def _local_maxima(x: np.ndarray):
    """Midpoints of strictly-rising/falling runs, SciPy's plateau convention.

    A flat top counts once, at its middle (floor of the midpoint), which is what
    ``_local_maxima_1d`` does.  Endpoints are never peaks.
    """
    n = x.size
    peaks, left_edges, right_edges = [], [], []
    i = 1
    while i < n - 1:
        if x[i - 1] < x[i]:
            ahead = i + 1
            while ahead < n - 1 and x[ahead] == x[i]:
                ahead += 1
            if x[ahead] < x[i]:
                left, right = i, ahead - 1
                peaks.append((left + right) // 2)
                left_edges.append(left)
                right_edges.append(right)
                i = ahead
        i += 1
    return (np.asarray(peaks, dtype=np.intp),
            np.asarray(left_edges, dtype=np.intp),
            np.asarray(right_edges, dtype=np.intp))


def _prominences(x: np.ndarray, peaks: np.ndarray):
    """SciPy's ``peak_prominences`` with ``wlen=-1`` (whole signal).

    Walk outward from the peak while the signal stays at or below the peak's
    own height; the lowest point reached on each side is that side's base, and
    the prominence is measured from the higher of the two bases.
    """
    prominences = np.empty(peaks.size)
    left_bases = np.empty(peaks.size, dtype=np.intp)
    right_bases = np.empty(peaks.size, dtype=np.intp)
    for k, peak in enumerate(peaks):
        summit = x[peak]
        # The walk stops at the first sample STRICTLY higher than the peak, so
        # the searched segment runs from the peak up to (not including) it.
        # Slicing to that boundary and taking a minimum is the same computation
        # as the step-by-step walk, minus the Python loop -- and this is 60% of
        # the runtime of the whole analysis when SciPy is unavailable.
        left = x[peak::-1]
        higher = np.flatnonzero(left > summit)
        segment = left[:higher[0]] if higher.size else left
        left_min = segment.min()
        # argmin returns the FIRST occurrence, which walking outward means the
        # minimum nearest the peak -- the one SciPy's strict `<` update keeps.
        left_bases[k] = peak - int(np.argmin(segment))

        right = x[peak:]
        higher = np.flatnonzero(right > summit)
        segment = right[:higher[0]] if higher.size else right
        right_min = segment.min()
        right_bases[k] = peak + int(np.argmin(segment))

        prominences[k] = summit - max(left_min, right_min)
    return prominences, left_bases, right_bases


def _widths(x: np.ndarray, peaks: np.ndarray, prominences: np.ndarray,
            left_bases: np.ndarray, right_bases: np.ndarray,
            rel_height: float = 0.5):
    """SciPy's ``peak_widths``: width where the peak has fallen ``rel_height``.

    Linear interpolation between the two samples that straddle the crossing, so
    the width is not quantised to whole bins.
    """
    widths = np.empty(peaks.size)
    for k, peak in enumerate(peaks):
        height = x[peak] - prominences[k] * rel_height
        i = peak
        i_min = left_bases[k]
        while i_min < i and height < x[i]:
            i -= 1
        left_ip = float(i)
        if x[i] < height:
            left_ip += (height - x[i]) / (x[i + 1] - x[i])
        i = peak
        i_max = right_bases[k]
        while i < i_max and height < x[i]:
            i += 1
        right_ip = float(i)
        if x[i] < height:
            right_ip -= (height - x[i]) / (x[i - 1] - x[i])
        widths[k] = right_ip - left_ip
    return widths


def _select_by_distance(peaks: np.ndarray, priority: np.ndarray,
                        distance: float) -> np.ndarray:
    """SciPy's ``_select_by_peak_distance``: tallest first, suppress neighbours.

    The comparison is strict (``< distance``) and the distance is rounded up,
    both to match SciPy exactly.
    """
    distance = int(np.ceil(distance))
    keep = np.ones(peaks.size, dtype=bool)
    for position in np.argsort(priority)[::-1]:
        if not keep[position]:
            continue
        k = position - 1
        while k >= 0 and peaks[position] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = position + 1
        while k < peaks.size and peaks[k] - peaks[position] < distance:
            keep[k] = False
            k += 1
    return keep


def find_peaks_numpy(x, prominence=None, distance=None, width=None):
    """The subset of ``scipy.signal.find_peaks`` this project uses.

    Filters are applied in SciPy's order -- distance, then prominence, then
    width -- which matters: a peak suppressed as too close to a taller one is
    gone before prominence is ever considered.
    """
    x = np.asarray(x, float)
    peaks, _, _ = _local_maxima(x)
    if peaks.size == 0:
        return peaks, {"prominences": np.zeros(0), "widths": np.zeros(0)}

    if distance is not None:
        peaks = peaks[_select_by_distance(peaks, x[peaks], distance)]
    prominences, left_bases, right_bases = _prominences(x, peaks)
    if prominence is not None:
        keep = prominences >= float(prominence)
        peaks = peaks[keep]
        prominences = prominences[keep]
        left_bases, right_bases = left_bases[keep], right_bases[keep]

    properties = {"prominences": prominences}
    if width is not None:
        widths = _widths(x, peaks, prominences, left_bases, right_bases)
        keep = widths >= float(width)
        peaks = peaks[keep]
        properties = {"prominences": prominences[keep], "widths": widths[keep]}
    return peaks, properties


# ---------------------------------------------------------------------------
# rectangular linear sum assignment
# ---------------------------------------------------------------------------

def linear_sum_assignment_numpy(cost):
    """Jonker-Volgenant shortest augmenting path; the optimal assignment.

    Returns the same ``(row_index, col_index)`` contract as SciPy, sorted by
    row.  The inner search is vectorised over columns, which is what keeps this
    usable at the sizes here (a hollow spectrum can carry twenty target peaks
    against a hundred predicted ones).

    SciPy's tie-breaking among equal-cost assignments is an implementation
    detail and is NOT reproduced -- only the total cost is guaranteed equal.
    The metrics only ever sum over matched pairs, so that is the property that
    matters, and the agreement test asserts equal totals rather than equal
    pairings.
    """
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix must be finite")

    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    # p[j] is the 1-based row currently assigned to column j; 0 means free.
    p = np.zeros(m + 1, dtype=np.intp)
    way = np.zeros(m + 1, dtype=np.intp)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            current = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (current < minv[1:])
            minv[1:][better] = current[better]
            way[1:][better] = j0
            candidates = np.where(free)[0]
            j1 = int(candidates[np.argmin(minv[1:][candidates])]) + 1
            delta = minv[j1]
            used_j = np.where(used)[0]
            u[p[used_j]] += delta
            v[used_j] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    columns = np.where(p[1:] != 0)[0]
    rows = p[1:][columns] - 1
    order = np.argsort(rows)
    rows, columns = rows[order], columns[order]
    if transposed:
        rows, columns = columns, rows
        order = np.argsort(rows)
        rows, columns = rows[order], columns[order]
    return rows.astype(np.intp), columns.astype(np.intp)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

HAVE_SCIPY = _scipy_available()


def find_peaks(x, **kwargs):
    if HAVE_SCIPY:
        from scipy.signal import find_peaks as _scipy_find_peaks
        return _scipy_find_peaks(x, **kwargs)
    return find_peaks_numpy(x, **kwargs)


def linear_sum_assignment(cost):
    if HAVE_SCIPY:
        from scipy.optimize import linear_sum_assignment as _scipy_lsa
        return _scipy_lsa(cost)
    return linear_sum_assignment_numpy(cost)


def backend_name() -> str:
    return "scipy" if HAVE_SCIPY else "numpy-fallback"
