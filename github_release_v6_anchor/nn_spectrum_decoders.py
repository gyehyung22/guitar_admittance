"""Peak-aware output heads and losses (plan Phase P1).

Sixty runs minimised a pointwise error on 500 log-frequency bins and converged
on a smooth conditional mean: about three resonances emitted where the target
carries eleven, and a narrow-scale energy ratio near 0.14.  Nothing in a
pointwise loss asks for narrow structure -- removing it is how such a loss is
minimised under uncertainty -- so this module supplies output parameterisations
and loss terms that do ask for it.

Everything here operates on the LOG-FREQUENCY grid, where one bin is a fixed
musical interval (19.156 cents on the 20-5000 Hz / 500-bin grid).  A Gaussian of
a given width in cents is therefore one fixed convolution kernel for the whole
band, which is what makes the envelope/detail split and the DoG losses cheap.

A note on normalisation that matters for reading results: the envelope/detail
decomposition is defined in the model's OUTPUT space, i.e. on whatever the
target normalisation produced.  Under ``--target-norm global`` that is an affine
image of decibels and the split means exactly what it says.  Under ``per_freq``
the per-bin scaling does not commute with blurring, so "detail" there is detail
of the normalised curve, not of the dB curve.  That is deliberate -- it keeps
A5/A6 (per-frequency normalisation + dual head) composable without carrying
per-row 500-vectors of scale through the loss -- but it means the dB detail
ratio reported at evaluation is the number to compare across arms, not the
training-time detail loss.

The heatmap targets are the opposite case: they are built ONCE at dataset load
from the decibel curves, so peak positions are the same fact regardless of which
normalisation a run uses.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ENVELOPE_SIGMA_CENTS = 150.0
DEFAULT_DOG_SCALES_CENTS = (50.0, 150.0)


def gaussian_kernel_bins(sigma_bins: float, truncate: float = 3.0):
    """A normalised 1-D Gaussian, odd length, as a tensor."""
    sigma_bins = max(float(sigma_bins), 1e-3)
    radius = max(int(math.ceil(truncate * sigma_bins)), 1)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel / kernel.sum()


class LogGaussianBlur(nn.Module):
    """Blur along the log-frequency axis by a fixed width in cents.

    Edge handling is replicate padding: the alternative, zero padding, invents a
    40 dB cliff at 20 Hz and 5 kHz and would manufacture "detail" at both ends of
    the very band the low-frequency recall is measured in.
    """

    def __init__(self, sigma_cents: float, cents_per_bin: float):
        super().__init__()
        self.sigma_cents = float(sigma_cents)
        self.cents_per_bin = float(cents_per_bin)
        kernel = gaussian_kernel_bins(self.sigma_cents / self.cents_per_bin)
        self.register_buffer("kernel", kernel.view(1, 1, -1), persistent=False)
        self.pad = kernel.numel() // 2

    def forward(self, curve):
        shape = curve.shape
        flat = curve.reshape(-1, 1, shape[-1])
        flat = F.pad(flat, (self.pad, self.pad), mode="replicate")
        return F.conv1d(flat, self.kernel).reshape(shape)


class DifferenceOfGaussians(nn.Module):
    """Band-pass views of a spectrum at the scales resonances actually live at.

    ``D_0_50  = y - G_50(y)``      structure narrower than 50 cents
    ``D_50_150 = G_50(y) - G_150(y)``  structure between 50 and 150 cents

    The plan rejects a raw second derivative as the primary detail term and this
    is why: a second difference on a 19-cent grid is dominated by one-bin jitter
    and sampling noise, whereas a difference of Gaussians selects a band and is
    insensitive to a shift far smaller than its own width.
    """

    def __init__(self, cents_per_bin: float,
                 scales_cents=DEFAULT_DOG_SCALES_CENTS):
        super().__init__()
        self.scales_cents = tuple(float(s) for s in scales_cents)
        self.blurs = nn.ModuleList([LogGaussianBlur(s, cents_per_bin)
                                    for s in self.scales_cents])

    def forward(self, curve):
        """Returns ``[y - G_first(y), G_first(y) - G_second(y), ...]``."""
        bands, previous = [], curve
        for blur in self.blurs:
            blurred = blur(previous)
            bands.append(previous - blurred)
            previous = blurred
        return bands


def dog_detail_loss(predicted, target, dog: DifferenceOfGaussians,
                    delta: float = 1.0):
    """Huber between the band-pass views of prediction and target."""
    loss = predicted.new_zeros(())
    for a, b in zip(dog(predicted), dog(target)):
        loss = loss + F.huber_loss(a, b, delta=delta)
    return loss


def focal_bce_with_logits(logits, targets, alpha: float = 0.75,
                          gamma: float = 2.0):
    """Focal binary cross-entropy for a very sparse positive class.

    A 500-bin heatmap carries roughly eleven narrow bumps, so under plain BCE
    the all-zero prediction is already excellent and the gradient that would
    create a peak is swamped.  Focal loss down-weights the easy negatives; alpha
    additionally tilts toward the positives.  The plan forbids plain BCE here
    for exactly this reason.
    """
    targets = targets.to(logits.dtype)
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    weight = (1.0 - p_t).clamp_min(1e-6) ** gamma
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * weight * ce).mean()


def count_loss(logits, target_counts, delta: float = 2.0):
    """Match the NUMBER of active events, per the plan's collapse guard.

    Used only alongside a positional term.  On its own it is trivially gamed by
    scattering the right number of events anywhere -- and on this data that is
    not a hypothetical: a random real hollow spectrum already matches ~55% of
    the target peaks within a semitone, so count alone buys a strong-looking
    score with no localisation at all.
    """
    predicted = torch.sigmoid(logits).sum(dim=-1)
    return F.huber_loss(predicted, target_counts.to(predicted.dtype),
                        delta=delta)


# ---------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int, zero_init: bool = False,
         bias_init: float | None = None):
    layers = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                           nn.Linear(hidden, out_dim))
    if zero_init:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    if bias_init is not None:
        nn.init.constant_(layers[-1].bias, float(bias_init))
    return layers


class DirectDecoder(nn.Module):
    """The current model: one linear map from the trunk to 500 bins."""

    kind = "direct"

    def __init__(self, hidden_dim: int, n_out: int, **_ignored):
        super().__init__()
        self.out = nn.Linear(hidden_dim, n_out)

    def forward(self, fused):
        return {"spectrum": self.out(fused)}


class EnvelopeDetailDecoder(nn.Module):
    """Separate heads for the broad envelope and the narrow structure (P1-1).

    The detail head is zero-initialised, so training starts exactly at an
    envelope-only model and any narrow structure has to be earned rather than
    being an artefact of initialisation.  ``detail_rms`` is returned on every
    forward pass because the plan requires collapse of the detail branch to be
    RECORDED, not inferred afterwards from the spectrum.
    """

    kind = "envelope_detail"

    def __init__(self, hidden_dim: int, n_out: int,
                 detail_hidden: int | None = None, **_ignored):
        super().__init__()
        detail_hidden = detail_hidden or hidden_dim
        self.envelope = nn.Linear(hidden_dim, n_out)
        self.detail = _mlp(hidden_dim, detail_hidden, n_out, zero_init=True)

    def forward(self, fused):
        envelope = self.envelope(fused)
        detail = self.detail(fused)
        return {"spectrum": envelope + detail,
                "envelope": envelope,
                "detail": detail,
                "detail_rms": detail.pow(2).mean().sqrt()}


class SalienceHeads(nn.Module):
    """Per-bin peak and valley logits (P1-3).

    Deliberately NOT the old auxiliary peak head, which hung off the latent and
    could be learned or ignored without the spectrum changing at all.  These
    logits share the trunk with the spectrum, and in the coupled decoder below
    they multiply it.
    """

    def __init__(self, hidden_dim: int, n_out: int, peak: bool, valley: bool):
        super().__init__()
        self.peak = _mlp(hidden_dim, hidden_dim, n_out) if peak else None
        self.valley = _mlp(hidden_dim, hidden_dim, n_out) if valley else None

    def forward(self, fused):
        out = {}
        if self.peak is not None:
            out["peak_logits"] = self.peak(fused)
        if self.valley is not None:
            out["valley_logits"] = self.valley(fused)
        return out


class CoupledSalienceDecoder(nn.Module):
    """Salience maps RENDER the detail rather than merely accompanying it (P1-4).

        detail = p_peak * softplus(a_peak) - p_valley * softplus(a_valley)
                 + broad_residual

    The softplus keeps each contribution signed by construction -- a peak head
    can only push the curve up and a valley head only down -- so a detector that
    has learned where a resonance is cannot be cancelled out by the residual
    learning the opposite.

    All three paths start at (near) zero.  The residual is zero-initialised
    outright; the two amplitude heads get a large negative output bias instead,
    because a zero-initialised head would still emit ``softplus(0) = 0.69`` dB
    everywhere the salience map is active.  Without this the coupled arm starts
    somewhere the baseline and dual-head arms do not, and a screening comparison
    between them would be reading initialisation as much as architecture.
    ``softplus(-6) = 0.0025``, so the branch is off at the start and still has a
    live gradient.
    """

    AMPLITUDE_BIAS_INIT = -6.0

    kind = "coupled"

    def __init__(self, hidden_dim: int, n_out: int,
                 detail_hidden: int | None = None,
                 peak: bool = True, valley: bool = True, **_ignored):
        super().__init__()
        detail_hidden = detail_hidden or hidden_dim
        self.envelope = nn.Linear(hidden_dim, n_out)
        self.salience = SalienceHeads(hidden_dim, n_out, peak, valley)
        bias = self.AMPLITUDE_BIAS_INIT
        self.peak_amp = (_mlp(hidden_dim, detail_hidden, n_out, bias_init=bias)
                         if peak else None)
        self.valley_amp = (_mlp(hidden_dim, detail_hidden, n_out,
                                bias_init=bias) if valley else None)
        self.residual = _mlp(hidden_dim, detail_hidden, n_out, zero_init=True)

    def forward(self, fused):
        envelope = self.envelope(fused)
        logits = self.salience(fused)
        detail = self.residual(fused)
        if self.peak_amp is not None:
            detail = detail + (torch.sigmoid(logits["peak_logits"])
                               * F.softplus(self.peak_amp(fused)))
        if self.valley_amp is not None:
            detail = detail - (torch.sigmoid(logits["valley_logits"])
                               * F.softplus(self.valley_amp(fused)))
        return {"spectrum": envelope + detail, "envelope": envelope,
                "detail": detail, "detail_rms": detail.pow(2).mean().sqrt(),
                **logits}


SPECTRUM_DECODERS = {
    "direct": DirectDecoder,
    "envelope_detail": EnvelopeDetailDecoder,
    "coupled": CoupledSalienceDecoder,
}


def build_decoder(kind: str, hidden_dim: int, n_out: int, *,
                  peak_head: str = "none", valley_head: str = "none",
                  detail_hidden: int | None = None):
    """Construct a decoder, attaching stand-alone salience heads if asked.

    ``coupled`` builds its own salience heads because it consumes them; for the
    other decoders a requested head is auxiliary and is wrapped alongside.
    """
    if kind not in SPECTRUM_DECODERS:
        raise ValueError(f"spectrum decoder must be one of "
                         f"{sorted(SPECTRUM_DECODERS)}")
    wants_peak = peak_head != "none"
    wants_valley = valley_head != "none"
    if kind == "coupled":
        if not (wants_peak or wants_valley):
            raise ValueError("--spectrum-decoder coupled needs at least one of "
                             "--peak-head/--valley-head; with neither it is an "
                             "envelope_detail model with extra steps")
        return CoupledSalienceDecoder(hidden_dim, n_out,
                                      detail_hidden=detail_hidden,
                                      peak=wants_peak, valley=wants_valley)
    decoder = SPECTRUM_DECODERS[kind](hidden_dim, n_out,
                                      detail_hidden=detail_hidden)
    if wants_peak or wants_valley:
        return AuxiliarySalienceWrapper(decoder, hidden_dim, n_out,
                                        wants_peak, wants_valley)
    return decoder


class AuxiliarySalienceWrapper(nn.Module):
    """A decoder plus salience heads that do not feed the spectrum.

    This is the plan's explicit first step for P1-3: diagnose with an auxiliary
    head, then couple it.  Keeping the two configurations as separate objects
    makes the comparison an ablation of the COUPLING, with the heads and their
    loss identical on both sides.
    """

    def __init__(self, decoder, hidden_dim: int, n_out: int,
                 peak: bool, valley: bool):
        super().__init__()
        self.decoder = decoder
        self.kind = f"{decoder.kind}+aux_salience"
        self.salience = SalienceHeads(hidden_dim, n_out, peak, valley)

    def forward(self, fused):
        return {**self.decoder(fused), **self.salience(fused)}
