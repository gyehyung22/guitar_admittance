"""
nn_train.py
-----------
Training script for AdmittanceNet.

Usage:
  python nn_train.py --dataset path/to/dataset
  python nn_train.py --dataset run_v4 run_v5 run_v6      # train on several at once
  python nn_train.py --dataset run_v6 --require-certified   # demand a finished run
  python nn_train.py --dataset d --epochs 200 --batch 16 --lr 1e-3
  python nn_train.py --dataset d --model physics_only   # baseline ablation
  python nn_train.py --dataset d --no-qc-filter         # include all done samples
  python nn_train.py --dataset d --device 1 --num-workers 8   # 2nd GPU, 8 loaders
  python nn_train.py --dataset d --wandb --wandb-project my-proj
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

# torch is imported before numpy/matplotlib on purpose: on Windows+conda a
# numpy that has already loaded Intel's libiomp5md makes torch's DLL load fail
# outright ("Error loading shm.dll").  Import order is the whole fix.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn_dataset import (GLOBAL_GEOMETRY_NAMES, RELATIONAL_SETS,
                        SHAPE_CHANNEL_SETS, SHAPE_KEY_MODES, SHAPE_PAIRINGS,
                        SPLIT_MODES, TARGET_NORM_MODES, build_datasets,
                        shape_key_counts)
from nn_spectrum_decoders import (DifferenceOfGaussians, LogGaussianBlur,
                                  build_decoder, count_loss,
                                  dog_detail_loss,
                                  focal_bce_with_logits)
from nn_model import (AdmittanceNet, AdmittanceNetPhysicsOnly,
                      AdmittanceNetShapeID, AdmittanceNetShapeOnly,
                      BodyExpertNet, CaseBridgeNet, PCAOutput, PCASpectrumNet,
                      PhysicsShapeResidualNet, ScalarOnlyNet, SpatialQueryNet)
from nn_model import BRIDGE_CONDITIONING


# ---------------------------------------------------------------------------
# Spectral diagnostics
# ---------------------------------------------------------------------------

class SpectralBasis:
    """Train-fitted PCA basis used purely to INSTRUMENT a run.

    Reports where a prediction lives in the target's own principal directions.
    The point is to separate three things a single MSE cannot:

      (A) output-subspace capacity   -> how well the rank-r basis reconstructs
                                        held-out spectra with oracle coefficients
      (B) decoder/loss behaviour     -> which PCs the model reproduces
      (C) latent content             -> answered separately, by latent_probe.py

    Effective output rank must NOT be read as an encoder rank: a loss dominated by
    the leading component, early stopping, or weight decay all suppress trailing
    PCs on their own.
    """

    def __init__(self, train_spectra, n_components: int = 64):
        spectra = np.asarray(train_spectra, dtype=np.float64)
        self.mean = spectra.mean(0)
        centered = spectra - self.mean
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        self.singular = singular
        self.components = vt[:int(n_components)]
        self.variance_share = (singular ** 2) / max((singular ** 2).sum(), 1e-30)
        coefficients = centered @ self.components.T
        self.coeff_std = coefficients.std(0) + 1e-8

    def project(self, spectra):
        return (np.asarray(spectra, float) - self.mean) @ self.components.T

    def oracle_mse(self, spectra, rank: int) -> float:
        """MSE of the best rank-``r`` reconstruction — capacity question (A)."""
        centered = np.asarray(spectra, float) - self.mean
        basis = self.components[:int(rank)]
        return float(((centered - (centered @ basis.T) @ basis) ** 2).mean())

    @staticmethod
    def effective_rank(matrix) -> float:
        """exp(entropy of the normalized singular spectrum) (Roy & Vetterli)."""
        matrix = np.asarray(matrix, float)
        if matrix.size == 0:
            return 0.0
        singular = np.linalg.svd(matrix - matrix.mean(0), compute_uv=False)
        power = singular ** 2
        total = power.sum()
        if total <= 0.0:
            return 0.0
        p = power / total
        p = p[p > 0.0]
        return float(np.exp(-(p * np.log(p)).sum()))

    def r2_per_component(self, predicted, target):
        """R^2 of the prediction in each principal direction of the TARGET."""
        cp = self.project(predicted)
        ct = self.project(target)
        residual = ((cp - ct) ** 2).sum(0)
        variance = ((ct - ct.mean(0)) ** 2).sum(0)
        return 1.0 - residual / np.maximum(variance, 1e-30)


def spectral_report(basis: SpectralBasis, predicted, target, prefix: str) -> dict:
    r2 = basis.r2_per_component(predicted, target)
    report = {
        f"{prefix}/rank_pred": basis.effective_rank(predicted),
        f"{prefix}/rank_target": basis.effective_rank(target),
        f"{prefix}/rank_residual": basis.effective_rank(
            np.asarray(predicted, float) - np.asarray(target, float)),
        f"{prefix}/pc_above_r2_0.5": float((r2 >= 0.5).sum()),
    }
    for k in (0, 1, 2, 3, 5, 7, 15, 31, 63):
        if k < r2.size:
            report[f"{prefix}/pc{k:02d}_r2"] = float(r2[k])
    return report


def to_decibels(values, samples, stats) -> np.ndarray:
    """Invert the target normalisation exactly, row by row.

    ``stats.denorm_y`` is per-bin under --target-norm per_freq and per-bin AND
    per-body under per_freq_body, so a single multiplication cannot undo it.
    Every physical-unit metric goes through here; anything that multiplies a
    normalised array by one scalar is only correct in the global mode and
    silently wrong in the others.
    """
    values = np.asarray(values, float)
    if stats.y_mode == "global":
        return values * stats.y_std + stats.y_mean
    return np.stack([stats.denorm_y(row, sample["body_type"])
                     for row, sample in zip(values, samples)])


def band_errors(predicted, target, freqs, y_std: float = 1.0) -> dict:
    """Absolute dB error split into the bands that fail differently."""
    predicted = np.asarray(predicted, float)
    target = np.asarray(target, float)
    error = np.abs(predicted - target) * float(y_std)
    freqs = np.asarray(freqs, float)
    out = {}
    for name, low, high in (("20_200", 20.0, 200.0),
                            ("200_1k", 200.0, 1000.0),
                            ("1k_5k", 1000.0, 5000.0)):
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            out[f"mae_db_{name}"] = float(error[:, mask].mean())
    out["mae_db_all"] = float(error.mean())
    return out


def split_report(predicted, target, samples, freqs, stats,
                 shape_key_mode: str, prefix: str) -> dict:
    """Aggregated error, reported the way the nesting demands.

    Bridges sit inside cases and cases inside base shapes, and the generator
    finishes cheap shapes first, so a flat per-row mean weights a shape by how
    many of its cases happen to exist.  The by-shape number is the one that
    answers "how well does this generalize to a body", and it is reported under
    BOTH shape keys because the two differ by a factor of two in mixed-v6.

    Solid and hollow are also reported apart: they are different physics (a
    hollow body has an air cavity and a Helmholtz resonance a solid one cannot
    have) and a pooled mean hides one failing while the other carries it.
    """
    from nn_dataset import _shape_key

    # MSE stays in the normalised space the loss and model selection use; the
    # dB and resonance metrics below use true decibels.
    per_sample = ((np.asarray(predicted, float)
                   - np.asarray(target, float)) ** 2).mean(1)
    report = {
        f"{prefix}/mse_by_case": grouped_mean(
            per_sample, [s["case_key"] for s in samples]),
        f"{prefix}/mse_by_shape": grouped_mean(
            per_sample, [_shape_key(s, shape_key_mode) for s in samples]),
        f"{prefix}/mse_by_shape_file": grouped_mean(
            per_sample, [_shape_key(s, "path") for s in samples]),
        f"{prefix}/mse_by_contour": grouped_mean(
            per_sample, [_shape_key(s, "contour") for s in samples]),
    }
    # Bridge sensitivity: how well the model reproduces the DEPARTURE of each
    # bridge from its own case mean.  An evaluation metric whether or not the
    # matching loss term is switched on -- a model can be excellent on the
    # absolute spectrum while predicting the same curve for all ten bridges,
    # and only this number says so.
    case_keys = [s["case_key"] for s in samples]
    predicted = np.asarray(predicted, float)
    target = np.asarray(target, float)
    predicted_db = to_decibels(predicted, samples, stats)
    target_db = to_decibels(target, samples, stats)
    order: dict = {}
    for row, key in enumerate(case_keys):
        order.setdefault(key, []).append(row)
    deltas_pred, deltas_true = [], []
    for rows in order.values():
        if len(rows) < 2:
            continue
        # In DECIBELS: a bridge-to-bridge difference is a physical quantity and
        # must not depend on which normalisation the run happened to use.
        block_p = predicted_db[rows]
        block_t = target_db[rows]
        deltas_pred.append(block_p - block_p.mean(0, keepdims=True))
        deltas_true.append(block_t - block_t.mean(0, keepdims=True))
    if deltas_pred:
        dp = np.concatenate(deltas_pred, 0)
        dt = np.concatenate(deltas_true, 0)
        report[f"{prefix}/bridge_delta_mae_db"] = float(np.abs(dp - dt).mean())
        report[f"{prefix}/bridge_delta_target_rms_db"] = float(
            np.sqrt((dt ** 2).mean()))
        report[f"{prefix}/bridge_delta_pred_rms_db"] = float(
            np.sqrt((dp ** 2).mean()))
        # 1.0 means the model reproduces the bridge-to-bridge spread; 0.0 means
        # it predicts one curve per case and lets the bridges collapse onto it.
        report[f"{prefix}/bridge_delta_rms_ratio"] = float(
            np.sqrt((dp ** 2).mean()) / max(np.sqrt((dt ** 2).mean()), 1e-12))

    # Absolute, physical-unit accuracy.  A relative sweep can only rank models;
    # these say whether any of them is actually usable.
    report.update({f"{prefix}/{k}": v for k, v in spectrum_peak_metrics(
        predicted_db, target_db, freqs).items()})
    # Reported here as well as at the call site, so a run under per-frequency
    # normalisation carries dB numbers that mean the same thing as every other
    # run's.
    report.update({f"{prefix}/{k}": v for k, v in
                   band_errors(predicted_db, target_db, freqs).items()})

    body = np.asarray([s["body_type"] for s in samples])
    for name in ("solid", "hollow"):
        rows = body == name
        if not rows.any():
            continue
        report[f"{prefix}/mse_{name}"] = float(per_sample[rows].mean())
        report[f"{prefix}/mse_by_shape_{name}"] = grouped_mean(
            per_sample[rows],
            [_shape_key(s, shape_key_mode)
             for s, keep in zip(samples, rows) if keep])
        report.update({
            f"{prefix}/{key}_{name}": value for key, value in band_errors(
                predicted_db[rows], target_db[rows], freqs).items()})
    return report


def spectrum_peak_metrics(predicted, target, freqs,
                          prominence_db: float = 1.5,
                          tolerances_cents=(50, 100, 200)) -> dict:
    """Resonance accuracy read off the CURVES, not off the auxiliary head.

    A thin wrapper over ``spectral_metrics.spectrum_event_report`` so training,
    the offline analyses and any future decoder are all judged by ONE detector
    and ONE matcher.  Two implementations of "did it find the resonance" drift
    apart the moment either is tuned, and the numbers then look comparable while
    measuring different things.

    Both arguments must already be in decibels -- pass them through
    ``to_decibels`` first.  Under ``--target-norm per_freq`` the normalisation
    scale differs per bin, so scaling a normalised curve by one number reshapes
    it and moves the very peaks being counted.

    The matching is a Hungarian assignment, not the nearest-first greedy pass
    this function used to do: greedy can commit a predicted peak to a target
    that a different prediction would have served better, which understates
    recall on dense hollow spectra.  Old key names are kept as aliases so
    existing summaries keep resolving.
    """
    from spectral_metrics import spectrum_event_report

    out = spectrum_event_report(predicted, target, freqs,
                                tolerances_cents=tolerances_cents,
                                detector={"prominence_db": prominence_db})
    out["peaks_per_row_pred"] = out["res_count_pred"]
    out["peaks_per_row_target"] = out["res_count_true"]
    out["peak_count_mae"] = out["res_count_mae"]
    for tolerance in tolerances_cents:
        tag = f"res_{int(tolerance)}c"
        out[f"res_precision_{int(tolerance)}c"] = out[f"{tag}_precision"]
        out[f"res_recall_{int(tolerance)}c"] = out[f"{tag}_recall"]
        out[f"res_f1_{int(tolerance)}c"] = out[f"{tag}_f1"]
        out[f"res_median_cents_{int(tolerance)}c"] = out[f"{tag}_match_cents_mae"]
    return out


def _peak_selection_score(diagnostics: dict, metric: str, best_mae: float,
                          guardrail: float):
    """Higher is better; ``None`` means this checkpoint is not eligible.

    ``constrained_peak`` is the plan's rule verbatim: among checkpoints whose dB
    MAE is within ``guardrail`` of the best seen, take the best peak F2.  The
    guardrail is what stops the rule degenerating -- peak recall alone is
    maximised by emitting events everywhere, and on this data a curve with a
    realistic number of events in arbitrary places already matches roughly half
    the hollow targets, so an unconstrained peak objective can look excellent
    while carrying no localisation at all.
    """
    if metric == "peak_f1_50":
        return diagnostics.get("val/res_50c_f1")
    if metric == "peak_f2_100":
        return diagnostics.get("val/res_100c_f2")
    if metric == "constrained_peak":
        score = diagnostics.get("val/res_100c_f2")
        mae = diagnostics.get("val/mae_db_all")
        if score is None or mae is None:
            return None
        if np.isfinite(best_mae) and mae > guardrail * best_mae:
            return None
        return score
    return None


def grouped_mean(values, keys) -> float:
    """Mean over groups, not over rows.

    Bridge samples are 10-per-case and cases are many-per-shape, so a flat mean
    silently weights a shape by how many of its cases happen to be finished — and
    the current dataset is cheap-shapes-first, i.e. exactly that kind of skew.
    """
    buckets: dict = {}
    for value, key in zip(np.asarray(values, float), keys):
        buckets.setdefault(key, []).append(value)
    return float(np.mean([np.mean(v) for v in buckets.values()])) if buckets else 0.0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# Relative weights inside the peak term.  Frequency is worth the most: a peak
# predicted at the wrong place is wrong no matter how good its height, and the
# whole reason for this head is to pin resonance LOCATION.  Q is the noisiest
# label (a half-power width measured off a sampled curve) and is weighted least.
PEAK_SUB_WEIGHTS = {"presence": 1.0, "freq_norm": 2.0,
                    "amp_norm": 1.0, "logq_norm": 0.5}

# Detection tolerances for the evaluation-only peak metrics.  100 cents is a
# semitone; 200 is a whole tone and roughly the width over which a body
# resonance still reads as "the same" resonance to a player.
PEAK_TOLERANCES_CENTS = (100, 200)

# The bands that fail differently, and the ones every metric is split over.
LOSS_BANDS = (("20_200", 20.0, 200.0), ("200_1k", 200.0, 1000.0),
              ("1k_5k", 1000.0, 5000.0))


class LogFrequencySmoother:
    """Gaussian blur along the log-frequency bin axis, in CENTS.

    A multiscale loss asks the model to get the coarse shape of the response
    right even where it cannot place an individual resonance.  Smoothing has to
    happen in cents rather than in bins so the kernel means the same musical
    interval at 40 Hz and at 4 kHz -- which it does automatically here only
    because the analysis grid is logarithmic; the per-bin spacing is measured
    from the actual grid rather than assumed.
    """

    def __init__(self, freqs_hz, cents: float, device=None, truncate: float = 3.0):
        freqs = np.asarray(freqs_hz, dtype=np.float64)
        steps = np.diff(np.log2(np.clip(freqs, 1e-9, None))) * 1200.0
        self.cents_per_bin = float(np.mean(steps))
        sigma = float(cents) / max(self.cents_per_bin, 1e-9)
        radius = max(int(round(truncate * sigma)), 1)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / max(sigma, 1e-9)) ** 2)
        kernel /= kernel.sum()
        self.sigma_bins = sigma
        self.radius = radius
        self.kernel = torch.as_tensor(
            kernel, dtype=torch.float32, device=device).view(1, 1, -1)

    def __call__(self, curves):
        """``(B, n_freq) -> (B, n_freq)``, edges handled by replication.

        Zero padding would pull the first and last bins toward zero and invent
        an error the model cannot fix; replication keeps the endpoints honest.
        """
        padded = F.pad(curves[:, None, :], (self.radius, self.radius),
                       mode="replicate")
        return F.conv1d(padded, self.kernel)[:, 0, :]


class SurrogateLoss:
    """Weighted magnitude MSE plus the auxiliary peak term.

    Two independent mechanisms, both aimed at the same failure: a plain MSE over
    500 log-magnitude bins is minimised by a SMOOTH curve, because peaks are few
    bins wide and move between samples, so averaging beats guessing.

    * ``peak_weight`` supervises the peak head, forcing the shared embedding to
      encode resonance frequency/amplitude/Q explicitly.
    * ``band_weight`` up-weights the magnitude bins that sit near a labelled peak,
      so the curve itself pays for flattening them.

    The reported validation number is deliberately the UNWEIGHTED magnitude MSE
    (``spectrum_mse`` below), so runs with different weightings stay comparable
    and model selection never chases its own loss shaping.
    """

    def __init__(self, freqs_hz, *, peak_weight: float, band_weight: float,
                 peak_bodies: str = "both",
                 band_sigma_decades: float = 0.02, device=None,
                 spectrum_loss: str = "mse", huber_delta: float = 1.0,
                 band_balanced: bool = False,
                 multiscale: "list[tuple[float, float]] | None" = None):
        self.peak_weight = float(peak_weight)
        self.peak_bodies = str(peak_bodies)
        self.band_weight = float(band_weight)
        self.band_sigma = float(band_sigma_decades)
        self.log_freqs = torch.log10(
            torch.as_tensor(np.asarray(freqs_hz, dtype=np.float32),
                            device=device).clamp_min(1e-6))
        self.spectrum_loss = str(spectrum_loss)
        self.huber_delta = float(huber_delta)
        self.band_balanced = bool(band_balanced)
        freqs = np.asarray(freqs_hz, float)
        # Boolean masks per band, precomputed once.  A band that the grid does
        # not reach is dropped rather than contributing an empty mean.
        self.band_masks = []
        for name, low, high in LOSS_BANDS:
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                self.band_masks.append((name, torch.as_tensor(
                    mask, dtype=torch.bool, device=device)))
        # (weight, smoother) pairs; weight 1.0 with no smoother is the raw term.
        self.scales = [(1.0, None)]
        for weight, cents in (multiscale or []):
            if weight > 0.0 and cents > 0.0:
                self.scales.append(
                    (float(weight),
                     LogFrequencySmoother(freqs_hz, cents, device=device)))

    def _elementwise(self, predicted, target):
        if self.spectrum_loss == "huber":
            return F.huber_loss(predicted, target, reduction="none",
                                delta=self.huber_delta)
        return (predicted - target) ** 2

    def _reduce(self, elementwise, weights=None):
        """Mean, or the equally-weighted mean of the three band means.

        Band balancing is NOT the old 5x boost on 200-1k: each band is averaged
        within itself and the three are then given equal say, so a band with
        fewer bins or a smaller natural error can no longer be drowned out --
        and no band is assigned a multiplier by hand.
        """
        if weights is not None:
            elementwise = elementwise * weights
            if not self.band_balanced:
                return elementwise.sum() / weights.sum()
        if not self.band_balanced:
            return elementwise.mean()
        terms = [elementwise[:, mask].mean() for _name, mask in self.band_masks]
        return torch.stack(terms).mean()

    def spectrum_term(self, predicted, target, weights=None):
        """Raw term plus every smoothed scale, each reduced the same way."""
        total = None
        for weight, smoother in self.scales:
            if smoother is None:
                value = self._reduce(self._elementwise(predicted, target),
                                     weights)
            else:
                # Weights are peak-position bumps and would be blurred into
                # meaninglessness at 150 cents, so the smoothed scales are
                # deliberately unweighted.
                value = self._reduce(
                    self._elementwise(smoother(predicted), smoother(target)))
            total = value * weight if total is None else total + value * weight
        return total

    def _band_weights(self, peaks):
        """Per-bin magnitude weights: 1 everywhere, higher near labelled peaks."""
        log_peak = torch.log10(peaks["frequency_hz"].clamp_min(1e-6))
        distance = (self.log_freqs[None, None, :]
                    - log_peak[:, :, None]) / self.band_sigma
        bump = torch.exp(-0.5 * distance * distance)
        bump = bump * peaks["mask"].to(bump.dtype)[:, :, None]
        # amax, not sum: where several resonances crowd together the emphasis
        # should saturate, not stack into a spike that dominates the batch.
        return 1.0 + self.band_weight * bump.amax(dim=1)

    def peak_term(self, pred_peaks, peaks, row_gate):
        """Peak losses, with whole ROWS excluded by ``row_gate``.

        The gate must not be folded into the BCE target: writing 0 there would
        teach "this body has no resonances" instead of "do not supervise this
        body".  Occupancy is supervised on gated rows only, and normalized by the
        number of gated slots.

        Hollow rows are gated off by default.  Across the 10 bridges of ONE hollow
        case the frequency in slot k varies by a median 823 cents (94 % of slots
        move more than 100 cents): a mode invisible at one drive point shifts
        every later slot by one, so the label contradicts itself within a case.
        """
        occupancy = peaks["mask"].to(row_gate.dtype)
        gated_slots = (torch.ones_like(occupancy) * row_gate).sum().clamp_min(1.0)
        valid = occupancy * row_gate
        parts = {"presence": (
            F.binary_cross_entropy_with_logits(
                pred_peaks["presence"], occupancy, reduction="none")
            * row_gate).sum() / gated_slots}
        for name in ("freq_norm", "amp_norm", "logq_norm"):
            squared = (pred_peaks[name] - peaks[name]) ** 2
            parts[name] = (squared * valid).sum() / valid.sum().clamp_min(1.0)
        return parts

    def __call__(self, pred_magnitude, pred_peaks, y, peaks):
        # Reported separately from the training term and always as the plain
        # unweighted MSE, so model selection and every cross-run comparison stay
        # independent of how the loss happens to be shaped.
        spectrum_mse = ((pred_magnitude - y) ** 2).mean()
        weights = self._band_weights(peaks) if self.band_weight > 0.0 else None
        magnitude_term = self.spectrum_term(pred_magnitude, y, weights)

        parts = {"spectrum_mse": spectrum_mse.detach(),
                 "magnitude_term": magnitude_term.detach()}
        total = magnitude_term
        if self.peak_weight > 0.0:
            peak_parts = self.peak_term(
                pred_peaks, peaks, self._body_gate(peaks, pred_magnitude))
            peak_total = sum(PEAK_SUB_WEIGHTS[name] * value
                             for name, value in peak_parts.items())
            total = total + self.peak_weight * peak_total
            parts.update({f"peak_{k}": v.detach()
                          for k, v in peak_parts.items()})
            parts["peak_total"] = peak_total.detach()
        return total, parts

    def _body_gate(self, peaks, reference):
        """Per-ROW 0/1 weight: 1 where the peak labels are trusted, else 0."""
        rows = peaks["is_solid"].reshape(-1, 1).to(reference.dtype)
        if self.peak_bodies == "both":
            return torch.ones_like(rows)
        if self.peak_bodies == "none":
            return torch.zeros_like(rows)
        return rows                          # solid rows only


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def group_mean(values, case_index):
    """Per-case mean, broadcast back to rows.

    ``case_index`` holds a GLOBAL case id per row; the groups are formed
    batch-locally, so a batch that happens to hold three cases produces three
    means regardless of what those ids are.
    """
    groups, inverse = torch.unique(case_index, return_inverse=True)
    totals = torch.zeros(groups.numel(), values.shape[-1], device=values.device,
                         dtype=values.dtype)
    totals.index_add_(0, inverse, values)
    counts = torch.zeros(groups.numel(), 1, device=values.device,
                         dtype=values.dtype)
    counts.index_add_(0, inverse,
                      torch.ones(values.shape[0], 1, device=values.device,
                                 dtype=values.dtype))
    return (totals / counts.clamp_min(1.0))[inverse]


def bridge_delta_terms(predicted, target, case_index, *, delta: float = 1.0):
    """How the response CHANGES from bridge to bridge, within a case.

    The absolute spectrum is dominated by material and size, which are shared by
    every bridge of a case; the part that answers "what does moving the bridge
    do" is what remains once the case mean is removed.  Supervising it directly
    stops that part from being absorbed as noise, and it is also the only
    quantity here that a player would notice.
    """
    predicted_delta = predicted - group_mean(predicted, case_index)
    target_delta = target - group_mean(target, case_index)
    return {
        "loss": F.huber_loss(predicted_delta, target_delta, delta=float(delta)),
        "mae": (predicted_delta - target_delta).abs().mean().detach(),
        "target_rms": target_delta.pow(2).mean().sqrt().detach(),
    }


def _to_device(mapping, device):
    return {name: value.to(device) for name, value in mapping.items()}


def shuffle_shape_input(inputs):
    """Permute the shape raster within the batch, leaving physics and target.

    The ablation this serves asks whether the contour image is USED, not whether
    it is useful: after the permutation each row still carries a real, correctly
    formed shape, just somebody else's, so nothing about the input distribution
    or the network's shape of computation changes -- only the correspondence
    between image and target.  A model that matches its unshuffled twin was
    never reading the image.

    ``shape_index`` is permuted with it, so the shape-ID diagnostic is ablated
    consistently rather than keeping a working back channel to shape identity.
    """
    permutation = torch.randperm(inputs["shape"].shape[0],
                                 device=inputs["shape"].device)
    shuffled = dict(inputs)
    shuffled["shape"] = inputs["shape"][permutation]
    if "shape_index" in inputs:
        shuffled["shape_index"] = inputs["shape_index"][permutation]
    return shuffled


class Objective:
    """Adapter so the loop is identical for a bin decoder and a PCA-coeff decoder.

    Returns ``(loss, parts, reconstructed_magnitude)``; the reported spectrum MSE
    is always computed on the reconstructed 500-bin curve, so the two target
    parameterizations are compared on the same number.
    """

    def __init__(self, surrogate: "SurrogateLoss", *, mode: str = "bins",
                 l1_weight: float = 0.1, coeff_weights=None,
                 bridge_delta_weight: float = 0.0,
                 bridge_delta_huber: float = 1.0,
                 zero_mean_weight: float = 0.0,
                 aux_relational_weight: float = 0.0,
                 envelope_blur=None, envelope_weight: float = 0.0,
                 detail_weight: float = 0.0,
                 dog=None, dog_weight: float = 0.0,
                 peak_loss_weight: float = 0.0,
                 valley_loss_weight: float = 0.0,
                 count_loss_weight: float = 0.0,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0):
        # Peak-aware terms (plan Phase P1).  All default to zero, so a run that
        # sets none of them is bit-identical to the objective that produced the
        # 60-run baseline -- every one of these is opt-in.
        self.envelope_blur = envelope_blur
        self.envelope_weight = float(envelope_weight)
        self.detail_weight = float(detail_weight)
        self.dog = dog
        self.dog_weight = float(dog_weight)
        self.peak_loss_weight = float(peak_loss_weight)
        self.valley_loss_weight = float(valley_loss_weight)
        self.count_loss_weight = float(count_loss_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.surrogate = surrogate
        self.mode = str(mode)
        self.l1_weight = float(l1_weight)
        self.bridge_delta_weight = float(bridge_delta_weight)
        self.bridge_delta_huber = float(bridge_delta_huber)
        self.zero_mean_weight = float(zero_mean_weight)
        # Supervising the queried spatial feature to reproduce the relational
        # scalars asks whether that feature encodes those relations at all.  The
        # answer is worth having even when the term does not help the spectrum,
        # and it is reported separately for that reason.
        self.aux_relational_weight = float(aux_relational_weight)
        # Per-component weights for the PCA objective.  Full whitening
        # (alpha=1) makes the 64th direction as important as the first and
        # wrecks the leading components; alpha=0 is the raw scale, where PC0's
        # 72% of the variance leaves the trailing directions with no gradient
        # worth having.  The sweep is over what lies between.
        self.coeff_weights = coeff_weights

    def _case_terms(self, model, inputs, magnitude, y, parts):
        """Bridge-delta supervision and the residual-centring constraint.

        Both need ``case_index``; a loader that does not keep a case whole in
        one batch would make the case mean a different quantity in every batch,
        so ``--case-batches`` is required rather than merely recommended.
        """
        extra = 0.0
        if self.bridge_delta_weight > 0.0:
            terms = bridge_delta_terms(magnitude, y, inputs["case_index"],
                                       delta=self.bridge_delta_huber)
            extra = extra + self.bridge_delta_weight * terms["loss"]
            parts["bridge_delta"] = terms["loss"].detach()
            parts["bridge_delta_mae"] = terms["mae"]
            parts["bridge_delta_target_rms"] = terms["target_rms"]
        if self.aux_relational_weight > 0.0 and hasattr(model, "auxiliary"):
            predicted = model.auxiliary(inputs)
            if predicted is not None:
                target = inputs["bridge_extra"][:, :predicted.shape[1]]
                aux = F.mse_loss(predicted, target)
                extra = extra + self.aux_relational_weight * aux
                parts["aux_relational"] = aux.detach()
                variance = target.var(0, unbiased=False).mean().clamp_min(1e-12)
                parts["aux_relational_r2"] = (1.0 - aux / variance).detach()
        if self.zero_mean_weight > 0.0 and hasattr(model, "parts"):
            _case, residual, _fused = model.parts(inputs)
            # The case path already cannot vary within a case, so this only
            # stops the BRIDGE path from carrying a constant the case path
            # should own -- which would make both terms unidentifiable.
            offset = group_mean(residual, inputs["case_index"]).abs().mean()
            extra = extra + self.zero_mean_weight * offset
            parts["residual_offset"] = offset.detach()
        return extra

    def _peak_aware_terms(self, decoded, y, peaks, parts):
        """Envelope/detail, DoG and salience supervision (plan Phase P1).

        The decomposition targets are built from ``y`` as the loop sees it, i.e.
        in the normalised output space -- see the note in nn_spectrum_decoders
        about why that is deliberate and what it means for per-frequency runs.
        The salience targets are the opposite: heatmaps precomputed from the dB
        curves, so they mean the same thing in every normalisation mode.
        """
        extra = y.new_zeros(())
        delta = self.surrogate.huber_delta
        if self.envelope_blur is not None and "envelope" in decoded:
            envelope_true = self.envelope_blur(y)
            detail_true = y - envelope_true
            envelope_term = F.huber_loss(decoded["envelope"], envelope_true,
                                         delta=delta)
            detail_term = F.huber_loss(decoded["detail"], detail_true,
                                       delta=delta)
            extra = (extra + self.envelope_weight * envelope_term
                     + self.detail_weight * detail_term)
            parts["envelope_term"] = envelope_term.detach()
            parts["detail_term"] = detail_term.detach()
            # Recorded on every run, per the plan: a dual-head model whose
            # detail branch has collapsed looks identical to a single-head
            # model in the spectrum loss and nowhere else.
            parts["detail_rms"] = decoded["detail"].detach().pow(2).mean().sqrt()
            parts["detail_rms_target"] = detail_true.pow(2).mean().sqrt()
        if self.dog is not None and self.dog_weight > 0.0:
            term = dog_detail_loss(decoded["spectrum"], y, self.dog,
                                   delta=delta)
            extra = extra + self.dog_weight * term
            parts["dog_term"] = term.detach()
        for kind, weight in (("peak", self.peak_loss_weight),
                             ("valley", self.valley_loss_weight)):
            logits = decoded.get(f"{kind}_logits")
            target = peaks.get(f"{kind}_heatmap")
            if logits is None or target is None or weight <= 0.0:
                continue
            term = focal_bce_with_logits(logits, target,
                                         alpha=self.focal_alpha,
                                         gamma=self.focal_gamma)
            extra = extra + weight * term
            parts[f"{kind}_focal"] = term.detach()
            counts = peaks.get(f"{kind}_event_count")
            if self.count_loss_weight > 0.0 and counts is not None:
                # Never on its own: matching the count while scattering the
                # events anywhere is a strong-looking score with no
                # localisation, which is exactly what the chance nulls measure.
                c_term = count_loss(logits, counts)
                extra = extra + self.count_loss_weight * c_term
                parts[f"{kind}_count"] = c_term.detach()
        return extra

    def __call__(self, model, inputs, y, peaks):
        if self.mode == "bins":
            decoded, pred_peaks = model.decode_parts(inputs)
            magnitude = decoded["spectrum"]
            loss, parts = self.surrogate(magnitude, pred_peaks, y, peaks)
            loss = loss + self._peak_aware_terms(decoded, y, peaks, parts)
            loss = loss + self._case_terms(model, inputs, magnitude, y, parts)
            return loss, parts, magnitude

        coefficients, pred_peaks = model.forward_parts(inputs)
        target = model.project(y)
        squared = (coefficients - target) ** 2
        if self.coeff_weights is not None:
            weights = self.coeff_weights.to(squared.device)
            coefficient_mse = (squared * weights).sum() / (
                weights.sum() * squared.shape[0])
        else:
            coefficient_mse = squared.mean()
        magnitude = model.reconstruct(coefficients)
        # The reconstruction term is what keeps a coefficient that is cheap in
        # the weighted metric from being expensive in dB.  It follows the
        # spectrum loss setting, so a Huber run is Huber on both sides.
        reconstruction_l1 = self.surrogate.spectrum_term(magnitude, y)
        total = coefficient_mse + self.l1_weight * reconstruction_l1
        parts = {
            "spectrum_mse": ((magnitude - y) ** 2).mean().detach(),
            "magnitude_term": reconstruction_l1.detach(),
            "coefficient_mse": coefficient_mse.detach(),
        }
        if self.surrogate.peak_weight > 0.0:
            peak_parts = self.surrogate.peak_term(
                pred_peaks, peaks, self.surrogate._body_gate(peaks, magnitude))
            peak_total = sum(PEAK_SUB_WEIGHTS[name] * value
                             for name, value in peak_parts.items())
            total = total + self.surrogate.peak_weight * peak_total
            parts["peak_total"] = peak_total.detach()
        total = total + self._case_terms(model, inputs, magnitude, y, parts)
        return total, parts, magnitude


def train_epoch(model, loader, optimizer, objective, device,
                shuffle_shape: bool = False):
    model.train()
    total_loss = 0.0
    total_spectrum = 0.0
    for inputs, y, peaks in loader:
        inputs = _to_device(inputs, device)
        if shuffle_shape:
            inputs = shuffle_shape_input(inputs)
        peaks = _to_device(peaks, device)
        y = y.to(device)
        optimizer.zero_grad()
        loss, parts, _magnitude = objective(model, inputs, y, peaks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_spectrum += parts["spectrum_mse"].item() * len(y)
    n = len(loader.dataset)
    return total_loss / n, total_spectrum / n


@torch.no_grad()
def eval_epoch(model, loader, objective, device, stats, collect: bool = False,
               shuffle_shape: bool = False):
    """Return ``(total_loss, spectrum_mse, peak_metrics, predictions, targets)``.

    ``spectrum_mse`` is the unweighted magnitude MSE — the model-selection metric
    and the only one comparable across loss configurations.  ``collect`` also
    returns the stacked curves, which the spectral diagnostics need.

    Peak metrics are reported over SOLID rows only.  A hollow slot-k label is not
    a consistent quantity across the bridges of one case, so averaging it in would
    report the model's error against a moving target.
    """
    model.eval()
    total_loss = total_spectrum = 0.0
    cent_error = amp_error = 0.0
    valid_slots = 0.0
    count_error = 0.0
    n_rows = 0.0
    detection = {tol: {"hit": 0.0, "claimed": 0.0, "labelled": 0.0}
                 for tol in PEAK_TOLERANCES_CENTS}
    span = ((stats.f_log_max - stats.f_log_min)
            if stats.has_peak_scalars else 1.0)
    predictions, targets = [], []
    for inputs, y, peaks in loader:
        inputs = _to_device(inputs, device)
        if shuffle_shape:
            inputs = shuffle_shape_input(inputs)
        peaks = _to_device(peaks, device)
        y = y.to(device)
        loss, parts, magnitude = objective(model, inputs, y, peaks)
        _pred_magnitude, pred_peaks = model.forward_all(inputs)
        total_loss += loss.item() * len(y)
        total_spectrum += parts["spectrum_mse"].item() * len(y)
        if collect:
            predictions.append(magnitude.cpu().numpy())
            targets.append(y.cpu().numpy())

        solid = peaks["is_solid"].to(magnitude.dtype)[:, None]
        mask = peaks["mask"].to(magnitude.dtype) * solid
        valid_slots += mask.sum().item()
        n_rows += solid.sum().item()
        predicted_count = (pred_peaks["presence"] > 0.0).to(magnitude.dtype)
        count_error += ((predicted_count * solid).sum(1)
                        - mask.sum(1)).abs().sum().item()
        # Frequency error in cents: musically meaningful and scale free, which
        # a Hz error over a 20 Hz - 5 kHz band is not.
        delta_decades = (pred_peaks["freq_norm"] - peaks["freq_norm"]) * span
        cent_error += (delta_decades.abs() * (1200.0 / math.log10(2.0))
                       * mask).sum().item()
        amp_error += ((pred_peaks["amp_norm"] - peaks["amp_norm"]).abs()
                      * mask).sum().item() * (stats.pk_amp_std or 1.0)
        # Detection quality per slot, at two musical tolerances.  A slot counts
        # as a hit when the model both claims it is occupied and places it
        # within the tolerance; "false" is a claimed slot that is not a hit and
        # "missed" is a labelled slot the model did not claim.  Slot-wise rather
        # than set-matched, which is only defensible because the labels are
        # frequency-ascending and prefix packed -- and only for SOLID rows,
        # where that ordering is stable across the bridges of a case.
        cents = (delta_decades.abs() * (1200.0 / math.log10(2.0)))
        claimed = predicted_count * solid
        for tol in PEAK_TOLERANCES_CENTS:
            hit = (claimed * mask * (cents <= float(tol))).sum().item()
            detection[tol]["hit"] += hit
            detection[tol]["claimed"] += claimed.sum().item()
            detection[tol]["labelled"] += mask.sum().item()
    n = len(loader.dataset)
    valid = max(valid_slots, 1.0)
    metrics = {
        "peak_freq_mae_cents_solid": cent_error / valid,
        "peak_amp_mae_db_solid": amp_error / valid,
        "peak_count_mae_solid": count_error / max(n_rows, 1.0),
    }
    for tol, counts in detection.items():
        precision = counts["hit"] / max(counts["claimed"], 1.0)
        recall = counts["hit"] / max(counts["labelled"], 1.0)
        metrics[f"peak_precision_{tol}c_solid"] = precision
        metrics[f"peak_recall_{tol}c_solid"] = recall
        metrics[f"peak_f1_{tol}c_solid"] = (
            2.0 * precision * recall / max(precision + recall, 1e-12))
        metrics[f"peak_false_{tol}c_solid"] = (
            (counts["claimed"] - counts["hit"]) / max(n_rows, 1.0))
        metrics[f"peak_missed_{tol}c_solid"] = (
            (counts["labelled"] - counts["hit"]) / max(n_rows, 1.0))
    if collect:
        return (total_loss / n, total_spectrum / n, metrics,
                np.concatenate(predictions, 0), np.concatenate(targets, 0))
    return total_loss / n, total_spectrum / n, metrics, None, None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curve(train_losses, val_losses, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train", lw=1.5)
    ax.plot(val_losses,   label="Val",   lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalized dB)")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, val_ds, stats, device, n_plot: int = 4, out_path: Path = None):
    """Plot predicted vs true log|Y| for a few validation samples."""
    model.eval()
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]

    indices = np.random.choice(len(val_ds), min(n_plot, len(val_ds)), replace=False)
    for ax, idx in zip(axes, indices):
        inputs, y_norm, _peak_targets = val_ds[idx]
        inputs = {
            name: value.unsqueeze(0).to(device)
            for name, value in inputs.items()
        }

        pred_norm, pred_peaks = model.forward_all(inputs)
        pred_norm = pred_norm.cpu().numpy().squeeze()
        body = val_ds.samples[idx]["body_type"]
        y_true_db = stats.denorm_y(y_norm.numpy(), body)
        y_pred_db = stats.denorm_y(pred_norm, body)

        s = val_ds.samples[idx]
        freqs = s["freqs"]
        ax.semilogx(freqs, y_true_db, lw=1.2, label="True", color="steelblue")
        ax.semilogx(freqs, y_pred_db, lw=1.2, label="Pred", color="darkorange",
                    ls="--")
        # Where the peak head thinks the resonances are.  Plotted against the
        # true peaks so a glance shows whether the auxiliary task is working even
        # when the magnitude curve is still smooth.
        true_mask = np.asarray(s["peak_mask"], bool)
        for f_true in np.asarray(s["peak_frequency_hz"], float)[true_mask]:
            ax.axvline(f_true, color="steelblue", lw=0.6, alpha=0.35)
        if stats.has_peak_scalars:
            keep = pred_peaks["presence"].cpu().numpy().squeeze() > 0.0
            predicted = stats.denorm_peak_freq(
                pred_peaks["freq_norm"].cpu().numpy().squeeze())
            for f_pred in np.atleast_1d(predicted)[np.atleast_1d(keep)]:
                ax.axvline(f_pred, color="darkorange", lw=0.6, alpha=0.45,
                           ls=":")
        ax.set_ylabel("|Y| dB")
        ax.set_title(f"{s['sample_id']}  {s['body_type']}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150)
        print(f"Prediction plot: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _resolve_device(spec: str) -> "torch.device":
    """Turn --device into a torch.device, failing loudly on an absent GPU.

    Accepts 'auto', 'cpu', 'cuda', 'cuda:N', or a bare index 'N'.  An explicit
    GPU request is never silently downgraded to CPU: a run that quietly trains
    100x slower than intended is worse than one that stops.
    """
    spec = str(spec).strip().lower()
    if spec in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec.isdigit():
        spec = f"cuda:{spec}"
    if not spec.startswith("cuda"):
        raise SystemExit(f"--device: unrecognised value {spec!r}")
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {spec}: CUDA is not available in this build")
    index = int(spec.split(":", 1)[1]) if ":" in spec else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise SystemExit(
            f"--device {spec}: only {count} CUDA device(s) visible "
            f"(indices 0..{count - 1})")
    return torch.device(f"cuda:{index}")


class _WandbRun:
    """Optional W&B logging.  A no-op object when --wandb is not given, so the
    training loop never has to branch and wandb stays an optional dependency."""

    def __init__(self, args=None, config=None):
        self.run = None
        if args is None or not args.wandb:
            return
        try:
            import wandb
        except ImportError as exc:                       # pragma: no cover
            raise SystemExit(
                "--wandb needs the wandb package: pip install wandb") from exc
        self._wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name, mode=args.wandb_mode,
            dir=str(args.out), config=config or {})

    def log(self, values: dict, step: int = None):
        if self.run is not None:
            self._wandb.log(values, step=step)

    def summary(self, values: dict):
        if self.run is not None:
            self.run.summary.update(values)

    def log_images(self, images: dict):
        if self.run is None:
            return
        payload = {name: self._wandb.Image(str(path))
                   for name, path in images.items() if Path(path).is_file()}
        if payload:
            self._wandb.log(payload)

    def log_artifact(self, path: Path, name: str, kind: str = "model"):
        if self.run is None or not Path(path).is_file():
            return
        artifact = self._wandb.Artifact(name, type=kind)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self):
        if self.run is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, nargs="+", type=Path,
                   metavar="DIR",
                   help="One or more dataset directories. Several may be given "
                        "(e.g. mixed-v4/v5/v6 runs) and are trained on together; "
                        "they must share the frequency grid. Unfinished runs are "
                        "trained on as-is unless --require-certified. No default: "
                        "point this at your own data.")
    p.add_argument("--qc-csv", nargs="*", default="qc_results/qc_metrics.csv",
                   type=str, metavar="CSV",
                   help="QC CSV to filter flagged samples: one path applied to "
                        "every dataset, or one per --dataset (relative paths are "
                        "resolved inside each dataset directory).")
    p.add_argument("--no-qc-filter", action="store_true",
                   help="Include flagged samples")
    p.add_argument("--require-certified", action="store_true",
                   help="Refuse a dataset that is not certified complete. By "
                        "default an unfinished run is trained on as-is (its "
                        "finished cases), which is announced at load; this flag "
                        "demands the sealed certificate instead.")
    p.add_argument("--grid-size", default=None, type=int,
                   help="Shape-raster resolution (default: "
                        "nn_dataset.DEFAULT_GRID_SIZE)")
    p.add_argument("--split-mode", default="shape", choices=list(SPLIT_MODES),
                   help="'shape' (default) honours the plan's shape-level split, "
                        "so validation measures generalization to an UNSEEN body "
                        "— the deployment condition, and the only split whose "
                        "number is reportable. 'case' groups the 10 bridge points "
                        "of one (shape, material) case; 'random' splits per "
                        "sample. Both leak geometry into validation and are "
                        "diagnostic only.")
    p.add_argument("--shape-key", default="path", choices=list(SHAPE_KEY_MODES),
                   help="What counts as one shape for subsetting and for the "
                        "by-shape metrics. 'path' = the contour FILE (the "
                        "historical default; mixed-v6 stores solid and hollow of "
                        "one base shape as two byte-identical files, so this "
                        "double counts). 'contour' = the contour CONTENTS, i.e. "
                        "distinct geometries. Both counts are always logged.")
    p.add_argument("--shuffle-shape", default="off",
                   choices=["off", "train", "both"],
                   help="Permute the shape raster within each batch. 'train' is "
                        "the ablation that asks whether the image is used at "
                        "all; 'both' additionally destroys it at evaluation.")
    p.add_argument("--drop-physics-inputs", nargs="*", default=None,
                   metavar="NAME",
                   help="Zero these columns of the physics vector (names from "
                        "nn_dataset.PHYSICS_INPUT_NAMES). Use "
                        "--drop-global-geometry for the standard shortcut "
                        "ablation. DIAGNOSTIC ONLY.")
    p.add_argument("--drop-global-geometry", action="store_true",
                   help=f"Shorthand for --drop-physics-inputs "
                        f"{' '.join(GLOBAL_GEOMETRY_NAMES)}: the scalars that "
                        f"duplicate what the raster carries and that identify a "
                        f"training shape essentially uniquely.")
    p.add_argument("--memorize", action="store_true",
                   help="Turn off every regularizer (dropout, weight decay, "
                        "peak/band loss, LR schedule, BatchNorm running stats) "
                        "without forcing a subset size, so it composes with "
                        "--max-train-shapes for a memorization ladder.")
    p.add_argument("--diag-train", action="store_true",
                   help="Also run the spectral diagnostics on the TRAIN split. "
                        "Needed for the memorization ladder, where the train "
                        "floor is the measurement.")
    p.add_argument("--model",      default="full",
                   choices=["full", "physics_only", "shape_only", "shape_id",
                            "body_experts", "residual", "residual_film",
                            "spatial", "case_bridge", "physics_wide",
                            "relational"],
                   help="full=CNN+MLP; physics_only=scalars only; "
                        "shape_only=contour image only; shape_id=a learned "
                        "embedding per TRAIN shape instead of the CNN "
                        "(TRAIN DIAGNOSTIC ONLY -- its val number is "
                        "meaningless, every unseen shape shares one vector); "
                        "body_experts=shared trunk with one output head per "
                        "body type; residual=explicit physics baseline plus a "
                        "zero-initialised shape residual; residual_film=the "
                        "same with the conv stack FiLM-conditioned on "
                        "case-level physics; spatial=adds a soundhole channel "
                        "and a bridge that samples the conv feature map at its "
                        "own position (use with --shape-channels); "
                        "case_bridge=a case-level response plus a per-bridge "
                        "correction (use with --case-batches); "
                        "physics_wide=scalars only at the FiLM model's "
                        "capacity (the control for whether that gain was "
                        "capacity); relational=physics_wide plus the "
                        "bridge/hole/edge distance scalars (the control a "
                        "spatial CNN branch has to beat).")
    p.add_argument("--shape-pairing", default="correct",
                   choices=list(SHAPE_PAIRINGS),
                   help="'fixed_shuffle' draws the wrong contour once and keeps "
                        "it, 'constant' gives every row the train mean image. "
                        "Together with --shuffle-shape train (redrawn every "
                        "batch) these separate 'the image is unused' from "
                        "'re-drawing it is a useful stochastic regulariser'.")
    p.add_argument("--augment", default="none", choices=["none", "mirror"],
                   help="'mirror' adds the three reflections of every TRAINING "
                        "body (x, y, xy). The material axes are coordinate "
                        "aligned and orthotropic, so a reflected body has the "
                        "identical response -- this is an exact symmetry, not "
                        "an approximation, and it quadruples the training set "
                        "at no FEM cost. Validation is never augmented.")
    p.add_argument("--relational-set", default="basic",
                   choices=sorted(RELATIONAL_SETS),
                   help="Which relational scalars the dataset emits. 'basic' is "
                        "the eight that produced the -10.7 percent result; "
                        "'extended' adds boundary rays cast from the bridge, "
                        "its position in the body's principal frame, "
                        "normalised area moments and the local edge curvature. "
                        "'basic' is a strict prefix of 'extended', so runs "
                        "under the two stay comparable.")
    p.add_argument("--drop-relational", nargs="*", default=None, metavar="NAME",
                   help="Zero these relational columns (leave-one-out). Layer "
                        "widths are unchanged, so capacity is not confounded.")
    p.add_argument("--keep-only-relational", nargs="*", default=None,
                   metavar="NAME",
                   help="Zero every relational column EXCEPT these "
                        "(leave-one-in).")
    p.add_argument("--target-norm", default="global",
                   choices=list(TARGET_NORM_MODES),
                   help="'per_freq' standardizes each frequency bin, so the "
                        "loss is not dominated by whichever part of the band "
                        "happens to vary most in dB. 'per_freq_body' uses "
                        "separate statistics for solid and hollow -- body type "
                        "is a known input at inference, so this is not leakage.")
    p.add_argument("--target-norm-floor", default=0.1, type=float, metavar="F",
                   help="Per-bin std floor, as a fraction of the mean per-bin "
                        "std, so a quiet bin's noise cannot become the loss.")
    p.add_argument("--shape-channels", default="occupancy",
                   choices=sorted(SHAPE_CHANNEL_SETS),
                   help="What the shape image carries. 'occupancy' is the "
                        "historical single binary channel. 'sdf' replaces it "
                        "with a signed distance field. The *_soundhole sets add "
                        "the hole, which the outer contour polygon structurally "
                        "cannot represent -- solid and hollow images of one base "
                        "shape are otherwise pixel-identical.")
    p.add_argument("--bridge-conditioning", default="query",
                   choices=list(BRIDGE_CONDITIONING),
                   help="--model spatial only. 'scalar' is the Diaz-style "
                        "arrangement: the image becomes one global vector and "
                        "the bridge enters the MLP as coordinates, so the shape "
                        "encoding is computed once per case and reused. 'query' "
                        "keeps that and adds a bilinear read of the conv maps "
                        "at the bridge's own coordinate. 'heatmap' draws the "
                        "bridge INTO the image, which forfeits the caching -- "
                        "the CNN then runs once per bridge, ten times per case.")
    p.add_argument("--bridge-map-type", default="gaussian",
                   choices=["gaussian", "distance", "delta"],
                   help="Form of the bridge channel. 'delta' is a sanity check "
                        "only: one pixel survives neither three max-pools nor "
                        "coordinate quantisation. 'distance' duplicates the "
                        "relational scalars and lets the network score well "
                        "without consulting the contour channel.")
    p.add_argument("--bridge-sigma-px", default=3.0, type=float, metavar="PX")
    p.add_argument("--bridge-query-levels", default="24,12", metavar="A,B",
                   help="Feature-map sizes to query. The soundhole is about a "
                        "pixel across at 12x12, so querying 12 alone throws "
                        "away the resolution the hole needs.")
    p.add_argument("--coordconv", action="store_true",
                   help="Append normalised x/y channels to the image.")
    p.add_argument("--freeze-scalar-baseline", action="store_true",
                   help="Freeze the scalar path so the spatial head cannot be "
                        "compensated for by it. Training them jointly from "
                        "scratch is what let the earlier residual experiment's "
                        "scalar branch absorb everything.")
    peaky = p.add_argument_group(
        "peak-aware output (plan Phase P1)",
        "All default to off: a run that sets none of these trains the exact "
        "objective the 60-run baseline used.")
    peaky.add_argument("--spectrum-decoder", default="direct",
                       choices=("direct", "envelope_detail", "coupled"),
                       help="direct: one linear map to 500 bins (current). "
                            "envelope_detail: separate broad and narrow heads, "
                            "summed, detail zero-initialised. coupled: the "
                            "salience maps render the detail, so a learned "
                            "peak position actually moves the spectrum.")
    peaky.add_argument("--envelope-sigma-cents", default=150.0, type=float,
                       metavar="C",
                       help="Width separating envelope from detail.")
    peaky.add_argument("--envelope-loss-weight", default=0.3, type=float,
                       metavar="W")
    peaky.add_argument("--detail-loss-weight", default=1.0, type=float,
                       metavar="W",
                       help="Weight on the detail head's own target. Applies "
                            "to the dual-head decomposition, not to --detail-loss.")
    peaky.add_argument("--detail-loss", default="none",
                       choices=("none", "dog"),
                       help="dog: Huber on difference-of-Gaussians band-pass "
                            "views. A raw second derivative is not offered: on "
                            "a 19-cent grid it is dominated by one-bin jitter.")
    peaky.add_argument("--dog-weight", default=0.5, type=float, metavar="W")
    peaky.add_argument("--dog-scales-cents", default="50,150", metavar="A,B")
    peaky.add_argument("--peak-head", default="none",
                       choices=("none", "heatmap"))
    peaky.add_argument("--valley-head", default="none",
                       choices=("none", "heatmap"),
                       help="Anti-resonances carry the coupling information "
                            "and the model flattens them as badly as peaks.")
    peaky.add_argument("--peak-loss-weight", default=1.0, type=float,
                       metavar="W")
    peaky.add_argument("--valley-loss-weight", default=1.0, type=float,
                       metavar="W")
    peaky.add_argument("--count-loss-weight", default=0.0, type=float,
                       metavar="W",
                       help="Match the NUMBER of events. Only meaningful "
                            "alongside a positional term: a random real hollow "
                            "spectrum already matches ~55%% of target peaks "
                            "within a semitone, so count alone buys a "
                            "strong-looking score with no localisation.")
    peaky.add_argument("--focal-alpha", default=0.75, type=float)
    peaky.add_argument("--focal-gamma", default=2.0, type=float)
    peaky.add_argument("--heatmap-sigma-cents", default=50.0, type=float,
                       metavar="C")
    peaky.add_argument("--peak-prominence-db", default=1.5, type=float,
                       metavar="DB",
                       help="Detector threshold for BOTH the heatmap targets "
                            "and the reported event metrics.")
    peaky.add_argument("--peak-distance-cents", default=0.0, type=float)
    peaky.add_argument("--peak-width-cents", default=0.0, type=float)
    peaky.add_argument("--checkpoint-metric", default="mse",
                       choices=("mse", "peak_f1_50", "peak_f2_100",
                                "constrained_peak"),
                       help="What 'best' means. constrained_peak applies the "
                            "plan's rule: best peak F2 among checkpoints whose "
                            "dB MAE is within --mae-guardrail of the run's own "
                            "best. The MSE-best checkpoint is saved either way.")
    peaky.add_argument("--mae-guardrail", default=1.05, type=float,
                       metavar="R")
    p.add_argument("--aux-relational-weight", default=0.0, type=float,
                   metavar="W",
                   help="Predict the relational scalars FROM the queried "
                        "spatial feature. Answers whether the CNN feature "
                        "encodes those relations at all, independently of "
                        "whether the term helps the spectrum.")
    p.add_argument("--no-bridge-query", dest="bridge_query",
                   action="store_false", default=True,
                   help="--model spatial only: drop the bilinear feature-map "
                        "read, keeping the extra channels. The control that "
                        "separates 'the soundhole channel helped' from 'the "
                        "spatial query helped'.")
    p.add_argument("--body-balanced-sampler", action="store_true",
                   help="Sample solid and hollow rows with equal probability. "
                        "The two bodies are not equally represented once a "
                        "generation run is partway through, and body_experts "
                        "would otherwise train one head on more data than the "
                        "other.")
    p.add_argument("--shape-dim", default=128, type=int, metavar="N",
                   help="Contour embedding width (full model only).")
    p.add_argument("--physics-dim", default=64, type=int, metavar="N",
                   help="Scalar-input embedding width (full model only).")
    p.add_argument("--hidden-dim", default=None, type=int, metavar="N",
                   help="Decoder width (default 256 full / 128 physics_only).")
    p.add_argument("--target", default="bins", choices=["bins", "pca"],
                   help="'bins' predicts the 500 magnitude bins directly. 'pca' "
                        "predicts coefficients in a train-fitted PCA basis and "
                        "reconstructs through a fixed inverse-PCA layer, so the "
                        "loss is not dominated by the leading component.")
    p.add_argument("--pca-components", default=64, type=int, metavar="N")
    p.add_argument("--pca-whiten", dest="pca_whiten", action="store_true",
                   default=True,
                   help="Regress coefficients standardized by their train std "
                        "(default). --no-pca-whiten is the control.")
    p.add_argument("--no-pca-whiten", dest="pca_whiten", action="store_false")
    p.add_argument("--pca-std-floor", default=0.05, type=float, metavar="F",
                   help="Coefficient std floor as a fraction of the leading "
                        "component's std. Stops a near-zero trailing coefficient "
                        "from acquiring an unbounded whitened weight.")
    p.add_argument("--pca-l1-weight", default=0.1, type=float, metavar="W",
                   help="Weight of the L1 term on the reconstructed spectrum.")
    # -- Stage 3: case / bridge decomposition.
    p.add_argument("--case-batches", action="store_true",
                   help="Build batches from WHOLE cases, so every bridge of a "
                        "case is present together. Required by "
                        "--bridge-delta-weight and --zero-mean-weight: a case "
                        "split across batches makes its mean a different "
                        "quantity in each one.")
    p.add_argument("--cases-per-batch", default=4, type=int, metavar="N",
                   help="With --case-batches, cases per batch (N x ~10 rows).")
    p.add_argument("--bridge-delta-weight", default=0.0, type=float,
                   metavar="W",
                   help="Supervise the departure of each bridge from its case "
                        "mean. The absolute spectrum is dominated by material "
                        "and size, which every bridge of a case shares; this is "
                        "the part that answers what MOVING the bridge does.")
    p.add_argument("--bridge-delta-huber", default=0.5, type=float, metavar="D")
    p.add_argument("--zero-mean-weight", default=0.0, type=float, metavar="W",
                   help="Penalise a non-zero per-case mean of the bridge "
                        "residual (case/bridge models only), so the two paths "
                        "cannot trade a constant back and forth.")
    # -- Stage 5: loss shaping.  Vary these with the architecture FIXED; a
    #    variant that wins under one loss and loses under another says nothing
    #    about either factor on its own.
    p.add_argument("--spectrum-loss", default="mse", choices=["mse", "huber"],
                   help="Per-bin loss. Huber bounds the pull of the few bins "
                        "where a missed resonance costs 20 dB, which under MSE "
                        "dominate the gradient and are answered by flattening.")
    p.add_argument("--huber-delta", default=1.0, type=float, metavar="D",
                   help="In NORMALIZED dB units (y_std ~ 7 dB), so 1.0 is "
                        "roughly one standard deviation of the target.")
    p.add_argument("--band-balanced", action="store_true",
                   help="Average the loss within 20-200 / 200-1k / 1k-5k Hz and "
                        "give the three bands equal weight. Not the old 5x "
                        "boost on 200-1k: no band gets a hand-picked multiplier.")
    p.add_argument("--multiscale", default=None, metavar="W:CENTS,...",
                   help="Extra smoothed copies of the spectrum loss, e.g. "
                        "'0.3:50,0.2:150'. Asks for the coarse shape to be "
                        "right even where an individual resonance cannot be "
                        "placed. Gaussian in CENTS on the log-frequency axis.")
    p.add_argument("--pca-weight-alpha", default=0.0, type=float, metavar="A",
                   help="Per-component weight (lambda_j + eps)^-alpha for the "
                        "PCA objective. 0 = raw scale, 1 = full whitening "
                        "(which measurably hurt). Sweep 0.25/0.5/0.75.")
    p.add_argument("--pca-weight-cap", default=8.0, type=float, metavar="C",
                   help="Upper bound on that weight, relative to the leading "
                        "component, so a near-zero eigenvalue cannot take over.")
    p.add_argument("--peak-weight", default=0.3, type=float, metavar="W",
                   help="Weight of the auxiliary peak-head loss. 0 disables the "
                        "term; the head is still built, so checkpoints stay "
                        "layout-compatible either way.")
    p.add_argument("--peak-loss-bodies", default="solid",
                   choices=["both", "solid", "none"],
                   help="Which bodies the peak labels are trusted for. Default "
                        "'solid': across the 10 bridges of one HOLLOW case the "
                        "frequency in slot k moves by a median 823 cents, so "
                        "those slots are not a consistent target. Use 'none' for "
                        "the learning-curve and PCA experiments.")
    p.add_argument("--tiny-set", default="off",
                   choices=["off", "sample", "case", "shape", "shapes5",
                            "shapes20", "full"],
                   help="Memorization ladder rung: shrink train to one sample / "
                        "one case (10 bridges) / one shape / 5 / 20 shapes, or "
                        "'full' to keep every training shape. All rungs turn OFF "
                        "dropout, weight decay, the peak loss, the LR schedule "
                        "and BatchNorm running statistics, so the measurement is "
                        "the TRAIN floor and nothing else. Failure to fit the "
                        "smallest rung is an implementation or optimization "
                        "fault, not a capacity limit.")
    p.add_argument("--max-train-shapes", default=None, type=int, metavar="N",
                   help="Keep only N base shapes in train (size-stratified).")
    p.add_argument("--cases-per-shape", default=None, type=int, metavar="N",
                   help="Keep at most N cases per shape, solid/hollow balanced.")
    p.add_argument("--max-train-cases", default=None, type=int, metavar="N",
                   help="Global cap on training cases — hold the FEM budget "
                        "fixed while --max-train-shapes varies.")
    p.add_argument("--shape-subset-seed", default=0, type=int, metavar="S",
                   help="Seed for shape/case subset selection. Vary this (not "
                        "--seed) for learning-curve repeats.")
    p.add_argument("--diag-every", default=10, type=int, metavar="N",
                   help="Epoch interval for the spectral diagnostics (PCA "
                        "effective rank, per-PC R^2, band MAE).")
    p.add_argument("--peak-band-weight", default=0.0, type=float, metavar="W",
                   help="Extra weight on magnitude bins near a labelled peak. 0 "
                        "(default) keeps the plain per-bin MSE. Try 3-5 to stop "
                        "the decoder from flattening resonances.")
    p.add_argument("--epochs",     default=300, type=int)
    p.add_argument("--batch",      default=16,  type=int)
    p.add_argument("--lr",         default=1e-3, type=float)
    p.add_argument("--wd",         default=1e-4, type=float,
                   help="Weight decay (L2 regularization)")
    p.add_argument("--out",        default="nn_runs/run001", type=Path)
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--device", default="auto", metavar="DEV",
                   help="'auto' (default), 'cpu', 'cuda', 'cuda:1', or a bare GPU "
                        "index like '1'.")
    p.add_argument("--num-workers", default=0, type=int, metavar="N",
                   help="DataLoader worker processes (default 0 = load in the "
                        "training process). Each worker holds its own rasterised "
                        "shape cache, a few MB.")
    p.add_argument("--pin-memory", action="store_true",
                   help="Page-locked host memory for faster host->GPU copies "
                        "(CUDA only; ignored on CPU).")
    p.add_argument("--wandb", action="store_true",
                   help="Log this run to Weights & Biases.")
    p.add_argument("--wandb-project", default="guitar-admittance", metavar="NAME")
    p.add_argument("--wandb-entity", default=None, metavar="NAME",
                   help="W&B team/user (default: your configured entity).")
    p.add_argument("--wandb-name", default=None, metavar="NAME",
                   help="Run name (default: W&B generates one).")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="'offline' logs to disk for a later `wandb sync`.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _resolve_device(args.device)
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})"
             if device.type == "cuda" else ""))
    if args.num_workers < 0:
        p.error("--num-workers must be >= 0")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    qc_csv = None if args.no_qc_filter else args.qc_csv
    grid_kwargs = ({} if args.grid_size is None
                   else {"grid_size": int(args.grid_size)})
    if args.drop_global_geometry:
        args.drop_physics_inputs = sorted(
            set(args.drop_physics_inputs or []) | set(GLOBAL_GEOMETRY_NAMES))
    if args.tiny_set != "off":
        args.memorize = True
        tiny_shapes = {"sample": 1, "case": 1, "shape": 1, "shapes5": 5,
                       "shapes20": 20, "full": None}
        args.max_train_shapes = tiny_shapes[args.tiny_set]
        if args.tiny_set in ("sample", "case"):
            args.cases_per_shape = 1
    if args.memorize:
        # A memorization probe must not be helped OR hindered by regularization,
        # scheduling or a moving target; every one of these would confound "the
        # model cannot fit this" with "the run was not allowed to".
        args.wd = 0.0
        args.peak_weight = 0.0
        args.peak_band_weight = 0.0
        args.diag_train = True
        print("[memorize] dropout=0 wd=0 peak=0 scheduler=off "
              "BN running-stats=off  train diagnostics=on")

    # Heatmap supervision is built from the dB curves at load time, and only if
    # a head will consume it -- the detection pass costs seconds but there is no
    # reason to pay it on a run that has no salience head.
    wants = {("peak" if args.peak_head != "none" else None),
             ("valley" if args.valley_head != "none" else None)} - {None}
    heatmap_kinds = ("both" if len(wants) == 2
                     else (wants.pop() if wants else "none"))
    train_ds, val_ds, stats = build_datasets(
        [str(d) for d in args.dataset], qc_csv=qc_csv, seed=args.seed,
        require_certified=args.require_certified,
        split_mode=args.split_mode,
        max_train_shapes=args.max_train_shapes,
        cases_per_shape=args.cases_per_shape,
        max_train_cases=args.max_train_cases,
        shape_subset_seed=args.shape_subset_seed,
        shape_key_mode=args.shape_key, shape_channels=args.shape_channels,
        shape_pairing=args.shape_pairing, target_norm=args.target_norm,
        target_norm_floor=args.target_norm_floor,
        relational_set=args.relational_set, augment=args.augment,
        event_heatmaps=heatmap_kinds,
        heatmap_sigma_cents=args.heatmap_sigma_cents,
        heatmap_prominence_db=args.peak_prominence_db,
        **grid_kwargs
    )
    if args.tiny_set == "sample":
        train_ds.samples = train_ds.samples[:1]
    print(f"Datasets: {', '.join(str(d) for d in args.dataset)}")
    print(f"Samples: train {len(train_ds)}  val {len(val_ds)}  "
          f"| shape grid {train_ds.grid_size}")
    train_counts = shape_key_counts(train_ds.samples)
    val_counts = shape_key_counts(val_ds.samples)
    # Both counts, always.  mixed-v6 writes solid and hollow of one base shape as
    # two byte-identical contour files, so "shapes" means two different numbers
    # depending on which key you use, and a run that reports only one of them
    # cannot be placed on a shape-axis learning curve afterwards.
    print(f"Shapes: train {train_counts['n_shape_files']} files / "
          f"{train_counts['n_contours']} distinct contours  |  "
          f"val {val_counts['n_shape_files']} files / "
          f"{val_counts['n_contours']} distinct contours  "
          f"(grouping key: {args.shape_key})")
    print(f"Shape channels: {args.shape_channels} -> {train_ds.channel_names}")
    # Created before anything writes into it (W&B included).
    args.out.mkdir(parents=True, exist_ok=True)
    # Inside the RUN directory, not beside it: the statistics depend on which
    # datasets were combined, so a shared file would let run A's checkpoint be
    # de-normalised with run B's stats and silently shift every prediction.
    stats.save(args.out / "norm_stats.npz")

    pin = bool(args.pin_memory) and device.type == "cuda"
    loader_kw = dict(num_workers=args.num_workers, pin_memory=pin)
    if args.num_workers > 0:
        # Workers are respawned every epoch otherwise, which dominates runtime
        # for a dataset this small.
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    print(f"DataLoader: num_workers={args.num_workers}  pin_memory={pin}")
    if args.case_batches:
        # A BatchSampler rather than a sampler: the constraint is on which rows
        # share a batch, not on how often a row is drawn.
        print(f"Case batches: {args.cases_per_batch} cases per batch "
              f"(~{args.cases_per_batch * 10} rows)")
        train_loader = DataLoader(
            train_ds, batch_sampler=train_ds.case_batches(
                args.cases_per_batch, shuffle=True, seed=args.seed),
            **loader_kw)
    elif args.body_balanced_sampler:
        # Inverse-frequency weights, so each draw is solid or hollow with equal
        # probability regardless of how far the generator got through each.
        counts = Counter(s["body_type"] for s in train_ds.samples)
        weights = [1.0 / counts[s["body_type"]] for s in train_ds.samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                        replacement=True)
        print(f"Body-balanced sampler: {dict(counts)} -> equal draw probability")
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  sampler=sampler, drop_last=False, **loader_kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  shuffle=True, drop_last=False, **loader_kw)
    # Validation is always case-whole when a case-level term is in play, so
    # the reported bridge-delta metric means the same thing as the loss.
    if args.case_batches or args.bridge_delta_weight > 0.0:
        val_loader = DataLoader(
            val_ds, batch_sampler=val_ds.case_batches(
                args.cases_per_batch, shuffle=False), **loader_kw)
    else:
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                **loader_kw)
    # Unshuffled pass over train, used only by --diag-train: the loop's running
    # train loss is an average over a model that changed during the epoch and
    # had dropout on, which is not the train FLOOR the ladder is measuring.
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch,
                                   shuffle=False, **loader_kw)
    if (args.bridge_delta_weight > 0.0 or args.zero_mean_weight > 0.0)             and not args.case_batches:
        p.error("--bridge-delta-weight / --zero-mean-weight need "
                "--case-batches, or the per-case mean is computed from a "
                "different subset of bridges in every batch")
    shuffle_train = args.shuffle_shape in ("train", "both")
    shuffle_eval = args.shuffle_shape == "both"

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    n_freq = int(train_ds.samples[0]["log_magnitude_db"].size)
    n_peaks = int(train_ds.samples[0]["peak_frequency_hz"].size)

    # Fitted on the TRAIN split only, and on normalized spectra so the numbers are
    # in the same units as the reported MSE.
    train_spectra = np.asarray(
        [stats.norm_y(s["log_magnitude_db"], s["body_type"])
         for s in train_ds.samples], float)
    basis = SpectralBasis(
        train_spectra, n_components=min(args.pca_components, n_freq,
                                        len(train_ds.samples)))
    val_spectra = np.asarray(
        [stats.norm_y(s["log_magnitude_db"], s["body_type"])
         for s in val_ds.samples], float)
    capacity = {f"oracle/val_mse_rank{r}": basis.oracle_mse(val_spectra, r)
                for r in (8, 16, 32, 64) if r <= basis.components.shape[0]}
    print("Output-subspace capacity (oracle coefficients, unseen val shapes): "
          + "  ".join(f"{k.rsplit('_', 1)[-1]}={v:.5f}"
                      for k, v in capacity.items()))

    STAGE2_MODELS = ("body_experts", "residual", "residual_film", "spatial",
                     "case_bridge", "physics_wide", "relational")
    if args.target == "pca" and args.model not in ("full",) + STAGE2_MODELS:
        p.error(f"--target pca is implemented for full and {STAGE2_MODELS}")

    if args.model in STAGE2_MODELS:
        # One output adapter shared by every Stage 2 variant, so a variant can be
        # compared against bins or against PCA coefficients without a second
        # class.  n_out is the head width; the reported MSE is always on the
        # reconstructed 500-bin curve either way.
        pca_output = None
        n_out = n_freq
        if args.target == "pca":
            floor = float(args.pca_std_floor) * float(basis.coeff_std[0])
            pca_output = PCAOutput(basis.components, basis.mean,
                                   np.maximum(basis.coeff_std, floor),
                                   whiten=bool(args.pca_whiten))
            n_out = int(basis.components.shape[0])
        model_config = {
            "model": args.model, "n_freq": n_freq, "n_peaks": n_peaks,
            "shape_dim": args.shape_dim, "physics_dim": args.physics_dim,
            "hidden_dim": args.hidden_dim if args.hidden_dim else 256,
            "n_out": n_out, "target": args.target,
            "whiten": bool(args.pca_whiten),
        }
        shared = dict(shape_dim=model_config["shape_dim"],
                      physics_dim=model_config["physics_dim"],
                      hidden_dim=model_config["hidden_dim"],
                      n_out=n_out, n_peaks=n_peaks, pca=pca_output,
                      drop_physics_inputs=args.drop_physics_inputs)
        if args.model in ("physics_wide", "relational"):
            names = RELATIONAL_SETS[args.relational_set]
            hidden = args.hidden_dim or 512
            decoder = None
            if (args.spectrum_decoder != "direct" or args.peak_head != "none"
                    or args.valley_head != "none"):
                if args.target != "bins":
                    p.error("--spectrum-decoder/--peak-head need --target bins; "
                            "the decompositions are defined on the 500-bin "
                            "curve, not on PCA coefficients")
                decoder = build_decoder(
                    args.spectrum_decoder, hidden, n_out,
                    peak_head=args.peak_head, valley_head=args.valley_head)
                model_config.update(
                    spectrum_decoder=args.spectrum_decoder,
                    peak_head=args.peak_head, valley_head=args.valley_head,
                    decoder_kind=getattr(decoder, "kind", args.spectrum_decoder))
            model = ScalarOnlyNet(
                hidden_dim=hidden, depth=4, n_out=n_out,
                n_peaks=n_peaks, pca=pca_output, decoder=decoder,
                use_relational=(args.model == "relational"),
                relational_dim=len(names), relational_names=names,
                drop_relational=args.drop_relational,
                keep_only_relational=args.keep_only_relational,
                drop_physics_inputs=args.drop_physics_inputs)
            model_config.update(
                hidden_dim=args.hidden_dim or 512, depth=4,
                use_relational=(args.model == "relational"),
                relational_set=args.relational_set,
                relational_dim=len(names),
                drop_relational=list(args.drop_relational or []),
                keep_only_relational=(list(args.keep_only_relational)
                                      if args.keep_only_relational else None))
            if args.model == "relational":
                active = [n for n in names
                          if n not in (args.drop_relational or [])
                          and (args.keep_only_relational is None
                               or n in args.keep_only_relational)]
                print(f"[relational] {len(active)}/{len(names)} columns active: "
                      f"{', '.join(active) if active else '(none)'}")
        if args.model not in ("spatial", "case_bridge", "physics_wide",
                              "relational"):
            shared["in_channels"] = len(train_ds.channel_names)
        model_config["shape_channels"] = args.shape_channels
        model_config["in_channels"] = len(train_ds.channel_names)
        if args.model in ("physics_wide", "relational"):
            pass                                   # built above
        elif args.model == "body_experts":
            model = BodyExpertNet(**shared)
        elif args.model == "case_bridge":
            model = CaseBridgeNet(
                in_channels=len(train_ds.channel_names), film=True, **shared)
        elif args.model == "spatial":
            levels = tuple(int(v) for v in
                           str(args.bridge_query_levels).split(",") if v)
            conditioning = ("scalar" if not args.bridge_query
                            else args.bridge_conditioning)
            relational_names = RELATIONAL_SETS[args.relational_set]
            model_config.update(
                bridge_conditioning=conditioning,
                bridge_map_type=args.bridge_map_type,
                bridge_sigma_px=float(args.bridge_sigma_px),
                bridge_query_levels=list(levels),
                coordconv=bool(args.coordconv),
                relational_set=args.relational_set,
                relational_dim=len(relational_names),
                aux_relational=bool(args.aux_relational_weight > 0.0))
            model = SpatialQueryNet(
                in_channels=len(train_ds.channel_names),
                relational_dim=len(relational_names),
                bridge_conditioning=conditioning,
                query_levels=levels, bridge_map_type=args.bridge_map_type,
                bridge_sigma_px=args.bridge_sigma_px,
                coordconv=bool(args.coordconv), film=True,
                aux_relational=bool(args.aux_relational_weight > 0.0),
                **shared)
            cached = conditioning in ("scalar", "query")
            print(f"[spatial] channels={train_ds.channel_names}  "
                  f"conditioning={conditioning}  query_levels={levels}  "
                  f"coordconv={args.coordconv}")
            # Built outside the f-string: an implicit concatenation inside an
            # f-string expression only parses on Python 3.12+, and the server
            # runs 3.11.
            note = ("per CASE and reusable across its bridges "
                    "(Diaz-style caching kept)" if cached else
                    "per BRIDGE: the heatmap makes the CNN bridge-dependent, "
                    "so training runs it once per bridge")
            print(f"[spatial] shape encoding is {note}")
            if args.freeze_scalar_baseline:
                frozen = model.freeze_scalar()
                print(f"[spatial] scalar baseline frozen ({frozen:,} params); "
                      f"only the zero-initialised spatial head can move")
        else:
            model = PhysicsShapeResidualNet(
                film=(args.model == "residual_film"), **shared)
    elif args.target == "pca":
        floor = float(args.pca_std_floor) * float(basis.coeff_std[0])
        coeff_std = np.maximum(basis.coeff_std, floor)
        model_config = {
            "model": "pca", "n_freq": n_freq, "n_peaks": n_peaks,
            "shape_dim": args.shape_dim, "physics_dim": args.physics_dim,
            "hidden_dim": args.hidden_dim if args.hidden_dim else 256,
            "n_components": int(basis.components.shape[0]),
            "whiten": bool(args.pca_whiten),
            "std_floor_fraction": float(args.pca_std_floor),
        }
        model = PCASpectrumNet(
            basis.components, basis.mean, coeff_std,
            shape_dim=model_config["shape_dim"],
            physics_dim=model_config["physics_dim"],
            hidden_dim=model_config["hidden_dim"],
            n_peaks=n_peaks, whiten=bool(args.pca_whiten),
            in_channels=len(train_ds.channel_names))
        print(f"PCA target: {model_config['n_components']} components  "
              f"whiten={args.pca_whiten}  std floor {floor:.4g} "
              f"(clipped {int((basis.coeff_std < floor).sum())} coefficients)")
    elif args.model == "full":
        model_config = {
            "model": "full", "n_freq": n_freq, "n_peaks": n_peaks,
            "shape_dim": args.shape_dim, "physics_dim": args.physics_dim,
            "hidden_dim": args.hidden_dim if args.hidden_dim else 256,
        }
        model = AdmittanceNet(
            shape_dim=model_config["shape_dim"],
            physics_dim=model_config["physics_dim"],
            hidden_dim=model_config["hidden_dim"],
            n_freq=n_freq, n_peaks=n_peaks,
            drop_physics_inputs=args.drop_physics_inputs,
            in_channels=len(train_ds.channel_names))
    elif args.model == "shape_only":
        model_config = {
            "model": "shape_only", "n_freq": n_freq, "n_peaks": n_peaks,
            "shape_dim": args.shape_dim,
            "hidden_dim": args.hidden_dim if args.hidden_dim else 256,
        }
        model = AdmittanceNetShapeOnly(
            shape_dim=model_config["shape_dim"],
            hidden_dim=model_config["hidden_dim"],
            n_freq=n_freq, n_peaks=n_peaks,
            in_channels=len(train_ds.channel_names))
    elif args.model == "shape_id":
        model_config = {
            "model": "shape_id", "n_freq": n_freq, "n_peaks": n_peaks,
            "shape_dim": args.shape_dim, "physics_dim": args.physics_dim,
            "hidden_dim": args.hidden_dim if args.hidden_dim else 256,
            "n_shapes": len(train_ds.shape_vocab),
            "shape_key": args.shape_key,
        }
        model = AdmittanceNetShapeID(
            n_shapes=model_config["n_shapes"],
            shape_dim=model_config["shape_dim"],
            physics_dim=model_config["physics_dim"],
            hidden_dim=model_config["hidden_dim"],
            n_freq=n_freq, n_peaks=n_peaks,
            drop_physics_inputs=args.drop_physics_inputs)
        print(f"[shape_id] TRAIN DIAGNOSTIC: {model_config['n_shapes']} learned "
              f"shape vectors; every val shape shares the reserved unknown slot, "
              f"so val/* is NOT a generalization number for this model.")
    else:
        model_config = {
            "model": "physics_only", "n_freq": n_freq, "n_peaks": n_peaks,
            "hidden_dim": args.hidden_dim if args.hidden_dim else 128,
        }
        model = AdmittanceNetPhysicsOnly(
            hidden_dim=model_config["hidden_dim"],
            n_freq=n_freq, n_peaks=n_peaks,
            drop_physics_inputs=args.drop_physics_inputs)
    model_config["drop_physics_inputs"] = list(args.drop_physics_inputs or [])
    model_config["shuffle_shape"] = args.shuffle_shape
    if args.drop_physics_inputs:
        print(f"[ablation] physics inputs zeroed: "
              f"{', '.join(args.drop_physics_inputs)}")
    if args.shuffle_shape != "off":
        print(f"[ablation] shape raster permuted within batch: "
              f"{args.shuffle_shape}")
    if args.memorize:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # Batch statistics only: with a handful of samples the running
                # estimates never converge, and the train/eval gap they create
                # would look like a failure to fit.
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    model.to(device)

    n_params = model.count_parameters()
    model_config["n_parameters"] = n_params
    # Written beside the checkpoint so a loader (the inference server) can rebuild
    # the exact architecture.  Without it, any run with non-default widths
    # produces a checkpoint that cannot be loaded with strict=True.
    (args.out / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8")
    print(f"Model: {args.model}  |  parameters: {n_params:,}")

    multiscale = []
    if args.multiscale:
        for item in str(args.multiscale).split(","):
            if not item.strip():
                continue
            weight, _, cents = item.partition(":")
            if not cents:
                p.error("--multiscale entries look like WEIGHT:CENTS")
            multiscale.append((float(weight), float(cents)))
    criterion = SurrogateLoss(
        train_ds.samples[0]["freqs"], peak_weight=args.peak_weight,
        band_weight=args.peak_band_weight,
        peak_bodies=args.peak_loss_bodies, device=device,
        spectrum_loss=args.spectrum_loss, huber_delta=args.huber_delta,
        band_balanced=args.band_balanced, multiscale=multiscale)
    coeff_weights = None
    if args.target == "pca" and args.pca_weight_alpha > 0.0:
        eigenvalue = np.asarray(basis.coeff_std, float) ** 2
        raw = (eigenvalue + 1e-12) ** (-float(args.pca_weight_alpha))
        # Normalised to the LEADING component so the cap means "at most C times
        # the weight PC0 gets", which is the quantity worth bounding.
        raw = raw / raw[0]
        coeff_weights = torch.as_tensor(
            np.minimum(raw, float(args.pca_weight_cap)), dtype=torch.float32)
        print(f"PCA component weights: alpha={args.pca_weight_alpha} "
              f"cap={args.pca_weight_cap} -> range "
              f"[{coeff_weights.min():.3g}, {coeff_weights.max():.3g}], "
              f"{int((coeff_weights >= args.pca_weight_cap).sum())} capped")
    # One bin is a fixed musical interval on this grid, so a width in cents is
    # one kernel for the whole band.
    # Read from the dataset rather than from `freqs`, which is bound further
    # down: the decoders have to exist before the objective that uses them.
    grid_hz = np.asarray(train_ds.samples[0]["freqs"], float)
    per_bin = float(np.mean(np.diff(np.log2(np.maximum(grid_hz, 1e-9)) * 1200.0)))
    envelope_blur = None
    if args.spectrum_decoder in ("envelope_detail", "coupled"):
        envelope_blur = LogGaussianBlur(args.envelope_sigma_cents,
                                        per_bin).to(device)
    dog = None
    if args.detail_loss == "dog":
        scales = [float(v) for v in str(args.dog_scales_cents).split(",")
                  if v.strip()]
        dog = DifferenceOfGaussians(per_bin, scales_cents=scales).to(device)
    objective = Objective(criterion, mode=args.target,
                          l1_weight=args.pca_l1_weight,
                          coeff_weights=coeff_weights,
                          bridge_delta_weight=args.bridge_delta_weight,
                          bridge_delta_huber=args.bridge_delta_huber,
                          zero_mean_weight=args.zero_mean_weight,
                          aux_relational_weight=args.aux_relational_weight,
                          envelope_blur=envelope_blur,
                          envelope_weight=args.envelope_loss_weight,
                          detail_weight=args.detail_loss_weight,
                          dog=dog, dog_weight=args.dog_weight,
                          peak_loss_weight=(args.peak_loss_weight
                                            if args.peak_head != "none" else 0.0),
                          valley_loss_weight=(args.valley_loss_weight
                                              if args.valley_head != "none"
                                              else 0.0),
                          count_loss_weight=args.count_loss_weight,
                          focal_alpha=args.focal_alpha,
                          focal_gamma=args.focal_gamma)
    scales = "+".join(["raw"] + [f"{w}x{c}c" for w, c in multiscale])
    print(f"Loss: target={args.target}  spectrum={args.spectrum_loss}"
          f"{f'(delta {args.huber_delta})' if args.spectrum_loss == 'huber' else ''}"
          f"  scales={scales}  band_balanced={args.band_balanced}"
          f"  peak_weight={args.peak_weight} ({args.peak_loss_bodies})"
          f"  band_weight={args.peak_band_weight}")
    freqs = np.asarray(train_ds.samples[0]["freqs"], float)

    # Everything needed to reproduce the run, including what the data actually
    # turned out to be (an unfinished dataset yields fewer samples than planned).
    wandb_run = _WandbRun(args, config={
        **{k: (str(v) if isinstance(v, Path) else v)
           for k, v in vars(args).items()},
        "dataset": [str(d) for d in args.dataset],
        "n_train": len(train_ds), "n_val": len(val_ds),
        "grid_size": train_ds.grid_size,
        "n_freq": int(train_ds.samples[0]["log_magnitude_db"].size),
        "n_parameters": n_params,
        "device": str(device),
        "y_mean_db": float(stats.y_mean), "y_std_db": float(stats.y_scale_db),
        "target_norm": args.target_norm,
    })
    wandb_run.summary({"n_parameters": n_params})

    # -----------------------------------------------------------------------
    # Optimizer + scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val  = float("inf")
    # The quantity checkpoint selection minimizes: validation MSE normally, train
    # MSE under --memorize.  Kept separate from best_val so the reported
    # best_val_mse always means the same thing.
    best_selection = float("inf")
    train_losses, val_losses = [], []
    t0 = time.time()

    best_epoch = 0
    best_peaks: dict = {}
    best_peak_score = -float("inf")
    best_peak_epoch = 0
    best_mae = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            diagnose = (epoch == 1 or epoch % max(args.diag_every, 1) == 0
                        or epoch == args.epochs)
            tr_loss, tr_spectrum = train_epoch(
                model, train_loader, optimizer, objective, device,
                shuffle_shape=shuffle_train)
            va_loss, va_spectrum, va_peaks, predicted, target = eval_epoch(
                model, val_loader, objective, device, stats, collect=diagnose,
                shuffle_shape=shuffle_eval)
            if not args.memorize:
                scheduler.step()

            diagnostics = {}
            if diagnose:
                diagnostics.update(spectral_report(basis, predicted, target, "val"))
                diagnostics.update({
                    f"val/{k}": v for k, v in
                    band_errors(to_decibels(predicted, val_ds.samples, stats),
                                to_decibels(target, val_ds.samples, stats),
                                freqs).items()})
                diagnostics.update(split_report(
                    predicted, target, val_ds.samples, freqs, stats,
                    args.shape_key, "val"))
            if diagnose and args.diag_train:
                # The memorization ladder measures the TRAIN floor, so the same
                # instrumentation has to run there.  Evaluated in eval mode on a
                # non-shuffled loader, so it is the fit of the trained model and
                # not the running average the training loop happens to report.
                _tl, tr_eval_mse, _tp, tr_pred, tr_target = eval_epoch(
                    model, train_eval_loader, objective, device, stats,
                    collect=True, shuffle_shape=shuffle_eval)
                diagnostics["train/mse_eval"] = tr_eval_mse
                diagnostics.update(
                    spectral_report(basis, tr_pred, tr_target, "train"))
                diagnostics.update({
                    f"train/{k}": v for k, v in
                    band_errors(to_decibels(tr_pred, train_ds.samples, stats),
                                to_decibels(tr_target, train_ds.samples, stats),
                                freqs).items()})
                diagnostics.update(split_report(
                    tr_pred, tr_target, train_ds.samples, freqs, stats,
                    args.shape_key, "train"))

            train_losses.append(tr_spectrum)
            val_losses.append(va_spectrum)

            # Selection is on the UNWEIGHTED magnitude MSE, not on the training
            # objective: it is the deliverable, and it stays comparable to runs
            # with a different peak/band weighting.
            #
            # Under --memorize the measurement is the TRAIN floor, and validation
            # is not a quantity the run is trying to optimize -- it drifts upward
            # from the first epochs.  Selecting on it would checkpoint an
            # unconverged model and report ITS train error as the floor, which is
            # exactly backwards.  So the memorization ladder selects on train.
            selection = tr_spectrum if args.memorize else va_spectrum
            if selection < best_selection:
                best_selection = selection
                best_val = va_spectrum
                best_epoch = epoch
                best_peaks = dict(va_peaks)
                torch.save(model.state_dict(), args.out / "best_model.pt")

            # A second, peak-aware selection running alongside the MSE one.  The
            # MSE checkpoint is always saved, so every arm keeps a comparable
            # reference point and the two rules can be compared after the fact
            # rather than one replacing the other.  Only evaluated on diagnostic
            # epochs: the event detector is a CPU pass over the whole val split.
            if args.checkpoint_metric != "mse" and diagnose:
                score = _peak_selection_score(
                    diagnostics, args.checkpoint_metric, best_mae,
                    args.mae_guardrail)
                mae_now = diagnostics.get("val/mae_db_all")
                if mae_now is not None and np.isfinite(mae_now):
                    best_mae = min(best_mae, float(mae_now))
                if score is not None and score > best_peak_score:
                    best_peak_score = score
                    best_peak_epoch = epoch
                    torch.save(model.state_dict(),
                               args.out / "best_peak_model.pt")

            lr_now = optimizer.param_groups[0]["lr"]
            # RMSE in dB is the number that means something physically; the
            # normalized MSE is only meaningful against this run's own stats.
            wandb_run.log({
                "train/loss": tr_loss, "train/mse": tr_spectrum,
                "val/loss": va_loss, "val/mse": va_spectrum,
                "val/rmse_db": (va_spectrum ** 0.5) * stats.y_scale_db,
                "val/best_mse": best_val,
                **{f"val/{k}": v for k, v in va_peaks.items()},
                **diagnostics, **capacity,
                "lr": lr_now, "epoch": epoch,
                "elapsed_s": time.time() - t0,
            }, step=epoch)

            if epoch == 1 or epoch % 20 == 0:
                elapsed = time.time() - t0
                extra = ""
                if "val/rank_pred" in diagnostics:
                    extra = (f"  rank={diagnostics['val/rank_pred']:.1f}"
                             f"/{diagnostics['val/rank_target']:.1f}"
                             f"  pc_r2>0.5={diagnostics['val/pc_above_r2_0.5']:.0f}")
                print(f"Epoch {epoch:3d}/{args.epochs}  "
                      f"train={tr_spectrum:.4f}  val={va_spectrum:.4f}  "
                      f"best={best_val:.4f}{extra}  "
                      f"lr={lr_now:.2e}  t={elapsed:.0f}s")

        # The last weights, saved before anything reloads the best ones.  The
        # plan's train/val diagnosis needs the regularized FINAL checkpoint as
        # well as the regularized best-val one: if they disagree, the gap is
        # early stopping rather than capacity.
        torch.save(model.state_dict(), args.out / "final_model.pt")
        if args.memorize:
            print(f"\nCheckpoint selected on TRAIN MSE: {best_selection:.4f} "
                  f"(epoch {best_epoch}); val at that epoch {best_val:.4f} "
                  f"-- validation is not meaningful for a memorization run.")
        print(f"\nBest val MSE: {best_val:.4f}  (epoch {best_epoch})")
        print(f"Best val RMSE (norm dB): {best_val**0.5:.4f}")
        print(f"Best val RMSE (dB): {(best_val**0.5) * stats.y_scale_db:.2f} dB")
        if best_peaks and args.peak_loss_bodies != "none":
            print("At best epoch (solid rows) — peak freq MAE "
                  f"{best_peaks['peak_freq_mae_cents_solid']:.0f} cents, "
                  f"amp MAE {best_peaks['peak_amp_mae_db_solid']:.2f} dB, "
                  f"count MAE {best_peaks['peak_count_mae_solid']:.2f}")
        # Diagnostics at the SELECTED checkpoint, not at whichever epoch the
        # schedule happened to instrument.  Everything a sweep collector reads
        # has to describe the same weights the checkpoint holds, otherwise a
        # table mixes the best epoch's MSE with the last epoch's rank.
        model.load_state_dict(
            torch.load(args.out / "best_model.pt", map_location=device,
                       weights_only=True))
        _bl, _bm, _bp, best_pred, best_target = eval_epoch(
            model, val_loader, objective, device, stats, collect=True,
            shuffle_shape=shuffle_eval)
        final_diagnostics = dict(
            spectral_report(basis, best_pred, best_target, "val"))
        final_diagnostics.update({
            f"val/{k}": v for k, v in
            band_errors(to_decibels(best_pred, val_ds.samples, stats),
                        to_decibels(best_target, val_ds.samples, stats),
                        freqs).items()})
        final_diagnostics.update(split_report(
            best_pred, best_target, val_ds.samples, freqs, stats,
            args.shape_key, "val"))
        if args.diag_train:
            _tl, tr_eval_mse, _tp, tr_pred, tr_target = eval_epoch(
                model, train_eval_loader, objective, device, stats,
                collect=True, shuffle_shape=shuffle_eval)
            final_diagnostics["train/mse_eval"] = tr_eval_mse
            final_diagnostics.update(
                spectral_report(basis, tr_pred, tr_target, "train"))
            final_diagnostics.update({
                f"train/{k}": v for k, v in
                band_errors(to_decibels(tr_pred, train_ds.samples, stats),
                            to_decibels(tr_target, train_ds.samples, stats),
                            freqs).items()})
            final_diagnostics.update(split_report(
                tr_pred, tr_target, train_ds.samples, freqs, stats,
                args.shape_key, "train"))
        if args.diag_train:
            # Same instrumentation on the FINAL weights, under its own prefix,
            # so one run answers "best-val vs final" without a second pass.
            model.load_state_dict(
                torch.load(args.out / "final_model.pt", map_location=device,
                           weights_only=True))
            _fl, _fm, _fp, fin_pred, fin_target = eval_epoch(
                model, val_loader, objective, device, stats, collect=True,
                shuffle_shape=shuffle_eval)
            final_diagnostics.update(split_report(
                fin_pred, fin_target, val_ds.samples, freqs, stats,
                args.shape_key, "valfinal"))
            _fl2, _fm2, _fp2, fin_tr, fin_tr_t = eval_epoch(
                model, train_eval_loader, objective, device, stats,
                collect=True, shuffle_shape=shuffle_eval)
            final_diagnostics.update(split_report(
                fin_tr, fin_tr_t, train_ds.samples, freqs, stats,
                args.shape_key, "trainfinal"))
            model.load_state_dict(
                torch.load(args.out / "best_model.pt", map_location=device,
                           weights_only=True))
            print(f"At best epoch — train MSE {tr_eval_mse:.4f}, "
                  f"train dB MAE {final_diagnostics['train/mae_db_all']:.2f}, "
                  f"train rank {final_diagnostics['train/rank_pred']:.2f}"
                  f"/{final_diagnostics['train/rank_target']:.2f}")

        n_train_shapes = train_counts["n_shape_files"]
        n_train_cases = len({s["case_key"] for s in train_ds.samples})
        print(f"Train subset: {n_train_shapes} shape files / "
              f"{train_counts['n_contours']} distinct contours / "
              f"{n_train_cases} cases / {len(train_ds)} samples")
        summary = {
            "best_val_mse": best_val,
            "best_val_rmse_norm": best_val ** 0.5,
            "best_val_rmse_db": (best_val ** 0.5) * stats.y_scale_db,
            "best_epoch": best_epoch,
            "epochs_run": len(val_losses),
            "train_time_s": time.time() - t0,
            "n_train_shapes": n_train_shapes,
            "n_train_shape_files": train_counts["n_shape_files"],
            "n_train_contours": train_counts["n_contours"],
            "n_val_contours": val_counts["n_contours"],
            "n_train_cases": n_train_cases,
            "n_train_samples": len(train_ds),
            "shape_key": args.shape_key,
            "shuffle_shape": args.shuffle_shape,
            # Last epoch's running training MSE (dropout active).  Not the train
            # floor -- that is train/mse_eval, and only --diag-train computes it.
            "train/mse": train_losses[-1] if train_losses else float("nan"),
            # The lowest running training MSE the run ever reached, independent
            # of which epoch was checkpointed.  For a memorization ladder this is
            # the answer, and it cannot be silently spoiled by the selection rule.
            "train/mse_min": min(train_losses) if train_losses else float("nan"),
            "selection_metric": "train" if args.memorize else "val",
            "checkpoint_metric": args.checkpoint_metric,
            "best_peak_epoch": best_peak_epoch,
            "best_peak_score": (float(best_peak_score)
                                if np.isfinite(best_peak_score) else None),
            "spectrum_decoder": args.spectrum_decoder,
            "detail_loss": args.detail_loss,
            "peak_head": args.peak_head,
            "valley_head": args.valley_head,
            "drop_physics_inputs": ",".join(args.drop_physics_inputs or []),
            **capacity,
            **final_diagnostics,
            **{f"best_{k}": v for k, v in best_peaks.items()},
        }
        wandb_run.summary(summary)
        # A machine-readable copy beside the checkpoint, so a sweep can be
        # collected without a W&B round trip and an offline run is not a dead end.
        (args.out / "run_summary.json").write_text(
            json.dumps({**summary, "args": {
                k: (str(v) if isinstance(v, Path) else
                    [str(x) for x in v] if isinstance(v, list) else v)
                for k, v in vars(args).items()}}, indent=2), encoding="utf-8")

        # -------------------------------------------------------------------
        # Save artifacts
        # -------------------------------------------------------------------
        np.save(args.out / "train_losses.npy", np.array(train_losses))
        np.save(args.out / "val_losses.npy",   np.array(val_losses))
        plot_training_curve(train_losses, val_losses,
                            args.out / "training_curve.png")

        # Best weights are already loaded, above, for the final diagnostics.
        plot_predictions(model, val_ds, stats, device, n_plot=4,
                         out_path=args.out / "val_predictions.png")

        wandb_run.log_images({
            "training_curve": args.out / "training_curve.png",
            "val_predictions": args.out / "val_predictions.png",
        })
        wandb_run.log_artifact(args.out / "best_model.pt",
                               f"{args.model}-best", kind="model")
        wandb_run.log_artifact(args.out / "norm_stats.npz",
                               "norm-stats", kind="preprocessing")
        wandb_run.log_artifact(args.out / "model_config.json",
                               "model-config", kind="preprocessing")
        print(f"\nRun saved to: {args.out}")
    finally:
        # Always close the run, so a crash or Ctrl-C still flushes what was logged.
        wandb_run.finish()


if __name__ == "__main__":
    main()
