"""One implementation of "what counts as a resonance", shared by every path.

Sixty runs optimised a normalised MSE and arrived at a smooth conditional mean:
the model emits about three peaks where the target has eleven.  Judging that
needs event metrics, and event metrics are only comparable if the detector, the
de-normalisation and the matching are literally the same code everywhere --
validation plots, dB errors, PCA oracles, structured-decoder oracles and
checkpoint selection alike.  That is what this module is for.

Two cautions are built in rather than left to the reader:

* the localisation error of MATCHED peaks is conditional on matching.  With
  recall near 0.08 it describes the handful of easy peaks that were found and
  says nothing about the ones that were not, so ``match_frequency_mae_cents``
  is always reported next to the unmatched-target distances that qualify it.
* frequency-sorted prefix slots are not used anywhere here.  Assignment is
  Hungarian on an unordered set, because slot k is not a consistent quantity
  across the bridges of one case.
"""

from __future__ import annotations

import numpy as np

# Log-frequency grid: one bin is a fixed musical interval, so tolerances are
# quoted in cents and the bin width is reported alongside them.
DEFAULT_TOLERANCES_CENTS = (50, 100, 200)
DEFAULT_PROMINENCE_DB = 1.5
# Costs for the Hungarian assignment.  Frequency dominates: a resonance in the
# wrong place is wrong whatever its height, and height is the noisier label.
MATCH_WEIGHTS = {"frequency": 1.0, "prominence": 0.15, "width": 0.05}


def cents_axis(freqs) -> np.ndarray:
    return np.log2(np.maximum(np.asarray(freqs, float), 1e-9)) * 1200.0


def cents_per_bin(freqs) -> float:
    return float(np.mean(np.diff(cents_axis(freqs))))


def _smooth(curve: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        return curve
    kernel = np.exp(-0.5 * (np.arange(-3 * bins, 3 * bins + 1) / bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(curve, (len(kernel) // 2, len(kernel) // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_events(curve_db, freqs, *, prominence_db: float = DEFAULT_PROMINENCE_DB,
                  distance_cents: float = 0.0, width_cents: float = 0.0,
                  smooth_bins: int = 0, kind: str = "peak") -> dict:
    """Peaks (``kind='peak'``) or anti-resonances (``kind='valley'``).

    Valleys are found by detecting peaks of the negated curve, so a dip is
    described by exactly the same three numbers a peak is and the two are
    directly comparable.  The attached plots show the model flattening sharp
    dips as badly as sharp peaks, and optimising only the peaks would licence a
    curve that is wrong in the other direction.
    """
    from _peak_backend import find_peaks

    curve = np.asarray(curve_db, float)
    signal = _smooth(curve, int(smooth_bins))
    if kind == "valley":
        signal = -signal
    per_bin = cents_per_bin(freqs)
    kwargs = {"prominence": float(prominence_db)}
    if distance_cents > 0:
        kwargs["distance"] = max(int(round(distance_cents / per_bin)), 1)
    if width_cents > 0:
        kwargs["width"] = max(width_cents / per_bin, 1e-6)
    index, properties = find_peaks(signal, **kwargs)
    axis = cents_axis(freqs)
    widths = properties.get("widths")
    return {
        "index": index,
        "cents": axis[index] if index.size else np.zeros(0),
        "hz": np.asarray(freqs, float)[index] if index.size else np.zeros(0),
        "prominence": properties.get("prominences", np.zeros(index.size)),
        "width_cents": (np.asarray(widths) * per_bin if widths is not None
                        else np.full(index.size, np.nan)),
        "amplitude": curve[index] if index.size else np.zeros(0),
    }


def match_events(true_events: dict, pred_events: dict, cutoff_cents: float,
                 weights: dict = None) -> dict:
    """One-to-one Hungarian assignment, with a cutoff that bounds every cost.

    A greedy nearest match can pair a predicted peak with a target peak that a
    different prediction fits far better, which inflates recall at exactly the
    operating point being studied.  Costs are clipped at the cutoff so an
    assignment can never be dragged across the tolerance to save a worse one.
    """
    from _peak_backend import linear_sum_assignment

    weights = weights or MATCH_WEIGHTS
    n_true = len(true_events["cents"])
    n_pred = len(pred_events["cents"])
    empty = {"pairs": [], "frequency_error": np.zeros(0),
             "prominence_error": np.zeros(0), "n_true": n_true,
             "n_pred": n_pred}
    if not n_true or not n_pred:
        return empty
    gap = np.abs(true_events["cents"][:, None] - pred_events["cents"][None, :])
    cost = weights["frequency"] * np.minimum(gap, cutoff_cents)
    cost = cost + weights["prominence"] * np.abs(
        true_events["prominence"][:, None] - pred_events["prominence"][None, :])
    # Width is the noisiest of the three (a half-power width read off a sampled
    # curve), so it only breaks ties.  NaNs appear whenever find_peaks was not
    # asked for widths; they must not poison the assignment.
    if weights.get("width"):
        width_gap = np.abs(np.nan_to_num(true_events["width_cents"])[:, None]
                           - np.nan_to_num(pred_events["width_cents"])[None, :])
        cost = cost + weights["width"] * width_gap
    rows, cols = linear_sum_assignment(cost)
    keep = gap[rows, cols] <= cutoff_cents
    rows, cols = rows[keep], cols[keep]
    return {
        "pairs": list(zip(rows.tolist(), cols.tolist())),
        "frequency_error": gap[rows, cols],
        "prominence_error": np.abs(true_events["prominence"][rows]
                                   - pred_events["prominence"][cols]),
        "n_true": n_true, "n_pred": n_pred,
    }


def event_metrics(predicted_db, target_db, freqs, *, kind: str = "peak",
                  tolerances_cents=DEFAULT_TOLERANCES_CENTS,
                  detector: dict = None, prefix: str = "") -> dict:
    """Set metrics over every row, aggregated over events rather than rows.

    F2 is reported because recall is the failing quantity here: the model finds
    roughly one target peak in eleven, so a score that weights precision and
    recall equally understates how far off it is.
    """
    detector = dict(detector or {})
    detector.pop("kind", None)
    predicted_db = np.asarray(predicted_db, float)
    target_db = np.asarray(target_db, float)
    per_bin = cents_per_bin(freqs)
    tolerances = tuple(sorted(set(list(tolerances_cents) + [round(per_bin, 3)])))

    counts_true, counts_pred = [], []
    matched = {t: {"n": 0, "freq": [], "prom": []} for t in tolerances}
    nearest_to_true, unmatched_prominence = [], []
    band_edges = ((20.0, 200.0), (200.0, 1000.0), (1000.0, 5000.0))
    band_true = {b: 0 for b in band_edges}
    band_hit = {b: 0 for b in band_edges}
    prominence_bins = ((1.5, 3.0), (3.0, 6.0), (6.0, 1e9))
    prom_true = {b: 0 for b in prominence_bins}
    prom_hit = {b: 0 for b in prominence_bins}

    for row in range(target_db.shape[0]):
        truth = detect_events(target_db[row], freqs, kind=kind, **detector)
        guess = detect_events(predicted_db[row], freqs, kind=kind, **detector)
        counts_true.append(len(truth["cents"]))
        counts_pred.append(len(guess["cents"]))
        if len(truth["cents"]) and len(guess["cents"]):
            gap = np.abs(truth["cents"][:, None] - guess["cents"][None, :])
            nearest_to_true.append(gap.min(1))
        elif len(truth["cents"]):
            nearest_to_true.append(np.full(len(truth["cents"]), np.inf))
        for tolerance in tolerances:
            assignment = match_events(truth, guess, float(tolerance))
            matched[tolerance]["n"] += len(assignment["pairs"])
            matched[tolerance]["freq"].append(assignment["frequency_error"])
            matched[tolerance]["prom"].append(assignment["prominence_error"])
            if abs(tolerance - 100.0) < 1e-6:
                hit_rows = {r for r, _c in assignment["pairs"]}
                for i, hz in enumerate(truth["hz"]):
                    for band in band_edges:
                        if band[0] <= hz < band[1]:
                            band_true[band] += 1
                            band_hit[band] += int(i in hit_rows)
                    for lo, hi in prominence_bins:
                        if lo <= truth["prominence"][i] < hi:
                            prom_true[(lo, hi)] += 1
                            prom_hit[(lo, hi)] += int(i in hit_rows)
                for i in range(len(truth["cents"])):
                    if i not in hit_rows:
                        unmatched_prominence.append(truth["prominence"][i])

    total_true = float(np.sum(counts_true))
    total_pred = float(np.sum(counts_pred))
    out = {
        f"{prefix}count_true": float(np.mean(counts_true)) if counts_true else 0.0,
        f"{prefix}count_pred": float(np.mean(counts_pred)) if counts_pred else 0.0,
        f"{prefix}count_ratio": total_pred / max(total_true, 1.0),
        f"{prefix}count_mae": float(np.mean(np.abs(np.array(counts_pred)
                                                   - np.array(counts_true))))
        if counts_true else 0.0,
        f"{prefix}cents_per_bin": per_bin,
    }
    for tolerance in tolerances:
        hits = matched[tolerance]["n"]
        precision = hits / max(total_pred, 1.0)
        recall = hits / max(total_true, 1.0)
        tag = f"{prefix}{int(round(tolerance))}c"
        out[f"{tag}_precision"] = precision
        out[f"{tag}_recall"] = recall
        out[f"{tag}_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
        out[f"{tag}_f2"] = (5 * precision * recall
                            / max(4 * precision + recall, 1e-12))
        errors = np.concatenate(matched[tolerance]["freq"]) \
            if matched[tolerance]["freq"] else np.zeros(0)
        proms = np.concatenate(matched[tolerance]["prom"]) \
            if matched[tolerance]["prom"] else np.zeros(0)
        # Conditional on matching -- see the module docstring.
        out[f"{tag}_match_cents_mae"] = float(errors.mean()) if errors.size else float("nan")
        out[f"{tag}_match_cents_p90"] = float(np.percentile(errors, 90)) if errors.size else float("nan")
        out[f"{tag}_match_prominence_mae"] = float(proms.mean()) if proms.size else float("nan")
    if nearest_to_true:
        gaps = np.concatenate(nearest_to_true)
        finite = gaps[np.isfinite(gaps)]
        # The unconditional number: how far the NEAREST prediction is from each
        # target event, missed ones included.
        out[f"{prefix}nearest_cents_median"] = float(np.median(finite)) if finite.size else float("nan")
        out[f"{prefix}nearest_cents_p90"] = float(np.percentile(finite, 90)) if finite.size else float("nan")
        out[f"{prefix}targets_with_no_prediction"] = float((~np.isfinite(gaps)).mean())
    for (lo, hi), total in band_true.items():
        name = f"{int(lo)}_{int(hi)}"
        out[f"{prefix}recall_100c_band_{name}"] = band_hit[(lo, hi)] / max(total, 1)
    for (lo, hi), total in prom_true.items():
        name = "6plus" if hi > 1e8 else f"{lo:g}_{hi:g}"
        out[f"{prefix}recall_100c_prom_{name}"] = prom_hit[(lo, hi)] / max(total, 1)
    if unmatched_prominence:
        out[f"{prefix}missed_prominence_median"] = float(
            np.median(unmatched_prominence))
    return out


def spectrum_event_report(predicted_db, target_db, freqs, *,
                          tolerances_cents=DEFAULT_TOLERANCES_CENTS,
                          detector: dict = None, prefix: str = "") -> dict:
    """Peaks AND valleys, which is what a usable admittance curve needs."""
    report = event_metrics(predicted_db, target_db, freqs, kind="peak",
                           tolerances_cents=tolerances_cents,
                           detector=detector, prefix=f"{prefix}res_")
    report.update(event_metrics(predicted_db, target_db, freqs, kind="valley",
                                tolerances_cents=tolerances_cents,
                                detector=detector, prefix=f"{prefix}val_"))
    return report


def detail_energy(curve_db, freqs, sigmas_cents=(50, 100, 150, 250)) -> dict:
    """How much of a spectrum is narrow structure rather than envelope.

    ``detail = y - blur(y)`` at a few scales.  The ratio between a prediction's
    detail RMS and its target's is the most direct measure of the failure under
    study: a model that has collapsed onto the conditional mean has a ratio near
    zero however good its MSE looks.
    """
    curve = np.asarray(curve_db, float)
    per_bin = cents_per_bin(freqs)
    out = {}
    for sigma in sigmas_cents:
        bins = max(sigma / per_bin, 1e-6)
        envelope = np.stack([_smooth(row, int(round(bins))) for row in
                             np.atleast_2d(curve)])
        detail = np.atleast_2d(curve) - envelope
        out[f"detail_rms_{int(sigma)}c"] = float(np.sqrt((detail ** 2).mean()))
        out[f"envelope_rms_{int(sigma)}c"] = float(
            np.sqrt(((envelope - envelope.mean(1, keepdims=True)) ** 2).mean()))
    return out
