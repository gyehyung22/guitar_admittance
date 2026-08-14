"""Neural networks for the mixed solid/hollow admittance dataset.

The shape branch embeds the 2-D body contour. The physics branch receives all
non-image inputs in the production contract: orthotropic material, bridge
position, geometry scalars, and a solid/hollow one-hot code. The current decoder
predicts the normalized logarithmic-magnitude curve; peak labels remain exposed
by ``nn_dataset`` for a future auxiliary prediction head.
"""

import numpy as np
import torch
import torch.nn as nn

from nn_dataset import (BODY_TYPE_DIM, BRIDGE_DIM, BRIDGE_NAMES, GEOMETRY_DIM,
                        GLOBAL_GEOMETRY_NAMES, MATERIAL_COLS, PEAK_SLOTS,
                        PHYSICS_INPUT_DIM, RELATIONAL_DIM,
                        physics_input_index)


class ShapeEncoder(nn.Module):
    """Encode a binary occupancy grid to a compact embedding.

    Resolution-agnostic: the stack ends in ``AdaptiveAvgPool2d(1)``, so changing
    ``nn_dataset.DEFAULT_GRID_SIZE`` needs no change here.
    """

    def __init__(self, out_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, value):
        return self.net(value)


class PhysicsEncoder(nn.Module):
    """Encode material(10), bridge(6), geometry(12), and body type(2).

    ``bridge`` carries mm coordinates plus the same point in the shape grid's
    normalised frame; ``geometry`` carries the manifest columns plus the
    contour's absolute width/height/area, which the scale-normalised shape
    channel deliberately drops.  The width is taken from
    ``nn_dataset.PHYSICS_INPUT_DIM`` so the two cannot drift apart.
    """

    def __init__(self, in_dim: int = PHYSICS_INPUT_DIM, out_dim: int = 64,
                 drop_inputs=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Suppressed columns are ZEROED, not removed: inputs arrive standardized,
        # so zero is the training mean and carries no information, while the layer
        # widths — and therefore the state_dict layout — stay identical to the
        # unablated model.  ``drop_inputs`` holds the NAMES for the run config.
        self.drop_inputs = list(drop_inputs or [])
        keep = torch.ones(in_dim)
        if self.drop_inputs:
            keep[physics_input_index(self.drop_inputs)] = 0.0
        # Non-persistent: it is derived from model_config, not learned, and
        # keeping it out of the state_dict leaves every existing checkpoint
        # loadable with strict=True.  ``.to(device)`` still moves it.
        self.register_buffer("input_keep", keep, persistent=False)

    def forward(self, material, bridge, geometry, body_type):
        value = torch.cat([material, bridge, geometry, body_type], dim=-1)
        return self.net(value * self.input_keep)


class PeakHead(nn.Module):
    """Auxiliary head predicting the labelled resonances slot by slot.

    ``nn_dataset`` stores ``PEAK_SLOTS`` peaks per bridge, prefix packed and in
    ascending frequency order, so slot k is unambiguously "the k-th lowest
    labelled resonance" and plain slot-wise regression is well posed — no set
    matching, no Hungarian assignment.

    Four numbers per slot: an occupancy logit (is slot k used at all?) plus
    normalized log-frequency, amplitude, and log-Q.  Supervising these forces the
    shared embedding to carry where the resonances ARE, which a plain MSE over the
    magnitude curve does not: peaks occupy few of the 500 bins and their positions
    move between samples, so the MSE-optimal answer under that uncertainty is a
    smooth average with no peaks in it.
    """

    def __init__(self, in_dim: int, n_peaks: int = PEAK_SLOTS,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_peaks = int(n_peaks)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4 * self.n_peaks),
        )

    def forward(self, fused):
        raw = self.net(fused).view(-1, 4, self.n_peaks)
        return {
            "presence": raw[:, 0],
            "freq_norm": raw[:, 1],
            "amp_norm": raw[:, 2],
            "logq_norm": raw[:, 3],
        }


class AdmittanceNet(nn.Module):
    """Predict normalized log-magnitude from the complete surrogate input.

    ``forward`` returns the magnitude curve alone, which is what inference needs
    and what every existing caller expects.  Training calls ``forward_all`` to get
    the auxiliary peak predictions from the same shared embedding.
    """

    def __init__(self, shape_dim: int = 128, physics_dim: int = 64,
                 hidden_dim: int = 256, n_freq: int = 500,
                 n_peaks: int = PEAK_SLOTS, drop_physics_inputs=None,
                 in_channels: int = 1):
        super().__init__()
        self.shape_enc = ShapeEncoder(out_dim=shape_dim,
                                      in_channels=in_channels)
        self.physics_enc = PhysicsEncoder(in_dim=PHYSICS_INPUT_DIM,
                                          out_dim=physics_dim,
                                          drop_inputs=drop_physics_inputs)
        self.decoder = nn.Sequential(
            nn.Linear(shape_dim + physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_freq),
        )
        # Always constructed, even when the peak loss weight is 0: the head's
        # presence keeps the state_dict layout constant, so a checkpoint stays
        # loadable with strict=True whatever the loss weighting was.
        self.peak_head = PeakHead(shape_dim + physics_dim, n_peaks, hidden_dim)

    def latents(self, inputs):
        """Return ``(z_shape, z_physics, z_fused)`` for latent probing.

        Exposed because "the decoder cannot use the information" and "the latent
        never had the information" produce the same spectrum but demand opposite
        fixes; only a probe on these three tensors separates them.
        """
        z_shape = self.shape_enc(inputs["shape"])
        z_physics = self.physics_enc(
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"])
        return z_shape, z_physics, torch.cat([z_shape, z_physics], dim=-1)

    def _fuse(self, inputs):
        return self.latents(inputs)[2]

    def forward(self, inputs):
        """Accept the input mapping returned by ``AdmittanceDataset``."""
        return self.decoder(self._fuse(inputs))

    def forward_all(self, inputs):
        """Return ``(magnitude, peak_predictions)`` from one shared embedding."""
        fused = self._fuse(inputs)
        return self.decoder(fused), self.peak_head(fused)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


class PCASpectrumNet(nn.Module):
    """Same encoders as ``AdmittanceNet``; predicts PCA coefficients instead of bins.

    Motivation: a plain 500-bin MSE is dominated by the leading principal
    direction (PC0 alone carries ~72 % of the target variance), so a model can sit
    at a good MSE while ignoring every trailing component — which is exactly the
    smooth-envelope failure mode.  Regressing STANDARDIZED coefficients gives each
    retained direction comparable weight in the loss.

    The basis, target mean and coefficient scales are non-trainable buffers, so
    they travel inside the checkpoint and reconstruction can never silently use a
    different basis than training did.

    ``whiten=False`` keeps the raw coefficient scale (the leading direction still
    dominates) and exists as the control for that very claim.
    """

    def __init__(self, basis, target_mean, coeff_std, *, shape_dim: int = 128,
                 physics_dim: int = 64, hidden_dim: int = 256,
                 n_peaks: int = PEAK_SLOTS, whiten: bool = True,
                 drop_physics_inputs=None):
        super().__init__()
        basis = torch.as_tensor(basis, dtype=torch.float32)          # (n_coeff, n_freq)
        self.n_coeff, self.n_freq = int(basis.shape[0]), int(basis.shape[1])
        self.whiten = bool(whiten)
        self.shape_enc = ShapeEncoder(out_dim=shape_dim)
        self.physics_enc = PhysicsEncoder(in_dim=PHYSICS_INPUT_DIM,
                                          out_dim=physics_dim,
                                          drop_inputs=drop_physics_inputs)
        self.head = nn.Sequential(
            nn.Linear(shape_dim + physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.n_coeff),
        )
        self.peak_head = PeakHead(shape_dim + physics_dim, n_peaks, hidden_dim)
        self.register_buffer("basis", basis)
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer(
            "coeff_std", torch.as_tensor(coeff_std, dtype=torch.float32))

    def latents(self, inputs):
        z_shape = self.shape_enc(inputs["shape"])
        z_physics = self.physics_enc(
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"])
        return z_shape, z_physics, torch.cat([z_shape, z_physics], dim=-1)

    def coefficients(self, inputs):
        """Predicted coefficients in the space the loss is computed in."""
        return self.head(self.latents(inputs)[2])

    def reconstruct(self, coefficients):
        raw = coefficients * self.coeff_std if self.whiten else coefficients
        return raw @ self.basis + self.target_mean

    def forward(self, inputs):
        return self.reconstruct(self.coefficients(inputs))

    def forward_all(self, inputs):
        coefficients, peaks = self.forward_parts(inputs)
        return self.reconstruct(coefficients), peaks

    def forward_parts(self, inputs):
        """``(coefficients, peak_predictions)`` — the loss needs coefficients."""
        fused = self.latents(inputs)[2]
        return self.head(fused), self.peak_head(fused)

    def project(self, spectra):
        """Target spectra -> the coefficient space the loss compares in."""
        raw = (spectra - self.target_mean) @ self.basis.t()
        return raw / self.coeff_std if self.whiten else raw

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


class AdmittanceNetShapeOnly(nn.Module):
    """Contour image only: no material, no bridge, no geometry scalars.

    The complement of ``AdmittanceNetPhysicsOnly``.  Its train error answers a
    question neither the full model nor the physics baseline can: whether the
    96x96 binary raster plus this encoder carry enough about a body to explain
    ANY of the response, before generalization is even at issue.  It is expected
    to be poor in absolute terms — material alone moves the spectrum far more
    than shape does — so it is read against the constant-prediction baseline,
    not against ``full``.
    """

    def __init__(self, shape_dim: int = 128, hidden_dim: int = 256,
                 n_freq: int = 500, n_peaks: int = PEAK_SLOTS,
                 in_channels: int = 1):
        super().__init__()
        self.shape_enc = ShapeEncoder(out_dim=shape_dim,
                                      in_channels=in_channels)
        self.decoder = nn.Sequential(
            nn.Linear(shape_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_freq),
        )
        self.peak_head = PeakHead(shape_dim, n_peaks, hidden_dim)

    def latents(self, inputs):
        z_shape = self.shape_enc(inputs["shape"])
        return z_shape, torch.zeros_like(z_shape[:, :0]), z_shape

    def _fuse(self, inputs):
        return self.shape_enc(inputs["shape"])

    def forward(self, inputs):
        return self.decoder(self._fuse(inputs))

    def forward_all(self, inputs):
        fused = self._fuse(inputs)
        return self.decoder(fused), self.peak_head(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdmittanceNetShapeID(nn.Module):
    """TRAIN DIAGNOSTIC ONLY — replaces the CNN with a lookup on shape identity.

    This model cannot be deployed: an unseen contour has no index, and every val
    row therefore lands on one shared "unknown" embedding.  Its val number is
    meaningless by construction and must never be quoted.

    What it measures is the TRAIN floor available when shape representation is
    made free.  If a learned per-shape vector drives train error far below what
    the CNN reaches on the same data, the shape ENCODER is the binding
    constraint on train fit.  If the ID model plateaus at the same train error,
    the limit lies elsewhere — missing inputs, the physics/shape interaction, the
    loss, or total capacity — and no amount of shape-encoder work will move it.
    """

    def __init__(self, n_shapes: int, shape_dim: int = 128,
                 physics_dim: int = 64, hidden_dim: int = 256,
                 n_freq: int = 500, n_peaks: int = PEAK_SLOTS,
                 drop_physics_inputs=None):
        super().__init__()
        # +1 for the reserved unknown slot every non-train shape maps to.
        self.embedding = nn.Embedding(int(n_shapes) + 1, shape_dim)
        self.n_shapes = int(n_shapes)
        self.physics_enc = PhysicsEncoder(in_dim=PHYSICS_INPUT_DIM,
                                          out_dim=physics_dim,
                                          drop_inputs=drop_physics_inputs)
        self.decoder = nn.Sequential(
            nn.Linear(shape_dim + physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_freq),
        )
        self.peak_head = PeakHead(shape_dim + physics_dim, n_peaks, hidden_dim)

    def latents(self, inputs):
        z_shape = self.embedding(inputs["shape_index"])
        z_physics = self.physics_enc(
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"])
        return z_shape, z_physics, torch.cat([z_shape, z_physics], dim=-1)

    def _fuse(self, inputs):
        return self.latents(inputs)[2]

    def forward(self, inputs):
        return self.decoder(self._fuse(inputs))

    def forward_all(self, inputs):
        fused = self._fuse(inputs)
        return self.decoder(fused), self.peak_head(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Stage 2 variants
#
# These share one output adapter so a variant can be trained against raw bins or
# against PCA coefficients without a second copy of the class.  The two older
# classes above are deliberately left untouched: their state_dicts are on disk.
# ---------------------------------------------------------------------------

class PCAOutput(nn.Module):
    """Fixed inverse-PCA layer, carried inside the checkpoint.

    Registered as buffers rather than rebuilt at load time so a reconstruction
    can never silently use a different basis than training did.
    """

    def __init__(self, basis, target_mean, coeff_std, whiten: bool = True):
        super().__init__()
        basis = torch.as_tensor(basis, dtype=torch.float32)   # (n_coeff, n_freq)
        self.n_out, self.n_freq = int(basis.shape[0]), int(basis.shape[1])
        self.whiten = bool(whiten)
        self.register_buffer("basis", basis)
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer(
            "coeff_std", torch.as_tensor(coeff_std, dtype=torch.float32))

    def reconstruct(self, coefficients):
        raw = coefficients * self.coeff_std if self.whiten else coefficients
        return raw @ self.basis + self.target_mean

    def project(self, spectra):
        raw = (spectra - self.target_mean) @ self.basis.t()
        return raw / self.coeff_std if self.whiten else raw


class SurrogateBase(nn.Module):
    """Common plumbing: subclasses implement ``_head(inputs) -> (raw, fused)``.

    ``raw`` is in the model's OUTPUT space -- 500 bins, or PCA coefficients when
    a ``PCAOutput`` is attached.  Everything the training loop calls is derived
    from that one method, so a new variant is one function.
    """

    # NOTE: no class-level ``pca = None`` here.  Assigning a Module to
    # ``self.pca`` registers it in ``_modules``, which is only reached through
    # ``__getattr__`` -- and ``__getattr__`` never runs while a class attribute
    # of the same name exists.  A default here would therefore shadow the
    # submodule and silently turn every PCA-target run into a bins run.
    def _pca(self):
        return self._modules.get("pca")

    def _head(self, inputs):
        raise NotImplementedError

    def reconstruct(self, raw):
        pca = self._pca()
        return pca.reconstruct(raw) if pca is not None else raw

    def project(self, spectra):
        pca = self._pca()
        return pca.project(spectra) if pca is not None else spectra

    def forward(self, inputs):
        return self.reconstruct(self._head(inputs)[0])

    def forward_parts(self, inputs):
        """``(raw_output, peaks)`` -- the PCA objective compares in raw space."""
        raw, fused = self._head(inputs)
        return raw, self.peak_head(fused)

    def decode_parts(self, inputs):
        """Everything the spectrum decoder produced, plus the legacy peak head.

        ``forward`` stays a single tensor so no existing caller changes; the
        peak-aware losses need the envelope, detail and salience maps as well,
        and this is the one place that exposes them.  Models without a spectrum
        decoder return just the spectrum, so the loss code has one shape to
        handle either way.
        """
        decoder = self._modules.get("decoder")
        if decoder is None:
            raw, fused = self._head(inputs)
            return {"spectrum": raw, "raw": raw}, self.peak_head(fused)
        fused = self._fuse(inputs)
        parts = decoder(fused)
        parts["raw"] = parts["spectrum"]
        return parts, self.peak_head(fused)

    def forward_all(self, inputs):
        raw, peaks = self.forward_parts(inputs)
        return self.reconstruct(raw), peaks

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _decoder(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class BodyExpertNet(SurrogateBase):
    """Shared encoders and trunk; one output head per body type (Variant A).

    Solid and hollow currently differ by 2.4x in validation MSE, and they are
    not the same physics: a hollow body has an air cavity and a Helmholtz
    resonance a solid slab cannot have.  A single output layer has to place both
    manifolds in one map, and whichever is easier dominates the gradient.

    Only the final projection is duplicated, so the parameter increase is one
    extra ``hidden_dim x n_out`` matrix and the shared trunk still sees every
    sample.  ``body_type`` stays in the physics input: the split is a decoder
    change, not an information change, and confounding the two would make the
    result unreadable.
    """

    def __init__(self, *, shape_dim: int = 128, physics_dim: int = 64,
                 hidden_dim: int = 256, n_out: int = 500,
                 n_peaks: int = PEAK_SLOTS, pca: PCAOutput | None = None,
                 drop_physics_inputs=None, in_channels: int = 1):
        super().__init__()
        self.shape_enc = ShapeEncoder(out_dim=shape_dim,
                                      in_channels=in_channels)
        self.physics_enc = PhysicsEncoder(in_dim=PHYSICS_INPUT_DIM,
                                          out_dim=physics_dim,
                                          drop_inputs=drop_physics_inputs)
        fused_dim = shape_dim + physics_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.head_solid = nn.Linear(hidden_dim, n_out)
        self.head_hollow = nn.Linear(hidden_dim, n_out)
        self.peak_head = PeakHead(fused_dim, n_peaks, hidden_dim)
        self.pca = pca

    def latents(self, inputs):
        z_shape = self.shape_enc(inputs["shape"])
        z_physics = self.physics_enc(
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"])
        return z_shape, z_physics, torch.cat([z_shape, z_physics], dim=-1)

    def _head(self, inputs):
        fused = self.latents(inputs)[2]
        features = self.trunk(fused)
        # body_type is a one-hot (solid, hollow); the gate is differentiable and
        # exact for one-hot input, and both heads stay reachable for either row
        # if a future dataset ever carries a soft label.
        is_solid = inputs["body_type"][:, :1]
        raw = (is_solid * self.head_solid(features)
               + (1.0 - is_solid) * self.head_hollow(features))
        return raw, fused


class FiLM(nn.Module):
    """Feature-wise linear modulation of a conv feature map."""

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(condition_dim, 2 * channels)
        # Start as the identity: gamma = 1, beta = 0, so an untrained FiLM does
        # not perturb the shape features it is supposed to modulate.
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.channels = int(channels)

    def forward(self, features, condition):
        gamma, beta = self.to_gamma_beta(condition).chunk(2, dim=-1)
        gamma = (1.0 + gamma)[:, :, None, None]
        return gamma * features + beta[:, :, None, None]


class ConditionedShapeEncoder(nn.Module):
    """The same conv stack as ``ShapeEncoder``, optionally FiLM-conditioned.

    Conditioning is on CASE-level physics only -- material, geometry scalars and
    body type.  Bridge position is deliberately excluded: it varies within a
    case while the shape features do not, so conditioning on it would make the
    encoder recompute a case-level quantity ten times per case and invite it to
    absorb the bridge effect that the bridge path is there to model.
    """

    def __init__(self, out_dim: int = 128, condition_dim: int | None = None,
                 in_channels: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        in_channels = int(in_channels)
        for channels in (32, 64, 128):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)))
            self.films.append(FiLM(condition_dim, channels)
                              if condition_dim else None)
            in_channels = channels
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.project = nn.Sequential(nn.Linear(128, out_dim), nn.ReLU(),
                                     nn.Dropout(0.3))

    def forward(self, image, condition=None):
        features = image
        for block, film in zip(self.blocks, self.films):
            features = block(features)
            if film is not None and condition is not None:
                features = film(features, condition)
        return self.project(self.pool(features))


class PhysicsShapeResidualNet(SurrogateBase):
    """An explicit physics baseline plus a shape-conditioned residual.

    physics_only reaches 0.1587 against the full model's 0.1515, so nearly all
    of the answer is already in the scalars.  In a single late-concat decoder
    that makes the shape branch's gradient a small correction on top of a large
    term the network can get from elsewhere, and the easiest solution is to
    ignore the image.  Here the baseline is a separate path that owns the
    dominant part, and the shape path is only ever asked for what the baseline
    missed.

    The residual head's last layer starts at ZERO, so the model begins exactly
    at the physics baseline and any residual it grows had to earn its way in.

    ``film=True`` additionally modulates the conv stack with case-level physics,
    which lets the shape features themselves depend on material and size rather
    than being a fixed function of the outline.
    """

    def __init__(self, *, shape_dim: int = 128, physics_dim: int = 64,
                 hidden_dim: int = 256, n_out: int = 500,
                 n_peaks: int = PEAK_SLOTS, pca: PCAOutput | None = None,
                 film: bool = False, drop_physics_inputs=None,
                 in_channels: int = 1):
        super().__init__()
        self.physics_enc = PhysicsEncoder(in_dim=PHYSICS_INPUT_DIM,
                                          out_dim=physics_dim,
                                          drop_inputs=drop_physics_inputs)
        self.baseline = _decoder(physics_dim, hidden_dim, n_out)
        # Case-level conditioning vector: material + geometry + body type.
        self.condition_dim = (len(MATERIAL_COLS) + GEOMETRY_DIM
                              + BODY_TYPE_DIM) if film else None
        self.shape_enc = ConditionedShapeEncoder(
            out_dim=shape_dim, condition_dim=self.condition_dim,
            in_channels=in_channels)
        self.residual = _decoder(shape_dim + physics_dim, hidden_dim, n_out)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.peak_head = PeakHead(shape_dim + physics_dim, n_peaks, hidden_dim)
        self.pca = pca

    def _condition(self, inputs):
        if self.condition_dim is None:
            return None
        return torch.cat([inputs["material"], inputs["geometry"],
                          inputs["body_type"]], dim=-1)

    def latents(self, inputs):
        z_shape = self.shape_enc(inputs["shape"], self._condition(inputs))
        z_physics = self.physics_enc(
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"])
        return z_shape, z_physics, torch.cat([z_shape, z_physics], dim=-1)

    def parts(self, inputs):
        """``(baseline, residual, fused)`` -- the ablation reads the residual."""
        z_shape, z_physics, fused = self.latents(inputs)
        return self.baseline(z_physics), self.residual(fused), fused

    def _head(self, inputs):
        baseline, residual, fused = self.parts(inputs)
        return baseline + residual, fused


class GeometryEncoder(nn.Module):
    """Case-level geometry encoder returning a global vector AND feature maps.

    Diaz et al. ("Rigid-Body Sound Synthesis with Differentiable Modal
    Resonators") encode a 64x64 occupancy grid with EfficientNet-B0 into a
    single 1000-d embedding, then concatenate material and excitation
    coordinates in an MLP.  The excitation point never enters the image, and the
    embedding is computed once per shape and reused for every material and every
    excitation coordinate -- that caching is a stated advantage of the design.

    That is a valid way to predict a position-dependent response: an MLP can in
    principle combine a global shape vector with a coordinate.  The concern here
    is NOT that it is mathematically impossible.  It is that the global vector
    must pre-compress, for every query position at once, whatever geometry those
    queries will need -- and in this problem the quantities that matter are
    local and relative (bridge to boundary, bridge to soundhole, soundhole to
    boundary).  Pre-compressing all of them into one fixed vector is plausibly
    data-inefficient at 53 independent training contours, where Diaz had 500
    convex shapes and ~1e8 samples.

    So this encoder keeps the global path and ADDS the maps, letting a bridge
    read the geometry at its own location instead of only through the summary.
    """

    def __init__(self, in_channels: int, out_dim: int = 128,
                 condition_dim: int | None = None, coordconv: bool = False,
                 widths=(32, 64, 128)):
        super().__init__()
        self.coordconv = bool(coordconv)
        channels_in = int(in_channels) + (2 if self.coordconv else 0)
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        for channels in widths:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels_in, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)))
            self.films.append(FiLM(condition_dim, channels)
                              if condition_dim else None)
            channels_in = channels
        self.level_channels = {48: widths[0], 24: widths[1], 12: widths[2]}
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels_in, out_dim), nn.ReLU(), nn.Dropout(0.3))

    def _coords(self, image):
        batch, _c, height, width = image.shape
        y = torch.linspace(-1.0, 1.0, height, device=image.device)
        x = torch.linspace(-1.0, 1.0, width, device=image.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        stacked = torch.stack([grid_x, grid_y])[None].expand(batch, -1, -1, -1)
        return stacked

    def forward(self, image, condition=None):
        features = (torch.cat([image, self._coords(image)], dim=1)
                    if self.coordconv else image)
        maps = {}
        for block, film in zip(self.blocks, self.films):
            features = block(features)
            if film is not None and condition is not None:
                features = film(features, condition)
            maps[features.shape[-1]] = features
        return self.global_path(features), maps


def sample_at_bridge(feature_map, bridge_xy_norm):
    """Bilinear read of ``feature_map`` at each row's normalised bridge point.

    ``bridge_x_norm``/``bridge_y_norm`` already live in [-1, 1] over the SAME
    isotropic frame the raster was built in, so they index the map directly.
    ``grid_sample`` wants (x, y) last and ``align_corners=True`` matches the
    ``linspace(-1, 1, n)`` convention the rasteriser uses.
    """
    grid = bridge_xy_norm.view(-1, 1, 1, 2)
    sampled = nn.functional.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    return sampled.view(feature_map.shape[0], -1)


def bridge_map(bridge_xy_norm, size: int, kind: str = "gaussian",
               sigma_px: float = 3.0):
    """A per-row image channel marking where the body is driven.

    ``gaussian``  a blob of peak 1 at the bridge.  The default: a single pixel
                  survives neither three max-pools nor coordinate quantisation,
                  and its gradient support is one pixel wide.
    ``distance``  the normalised distance to the bridge at every pixel.  Dense
                  and easy to exploit -- which is the risk: it duplicates the
                  relational scalars and lets the network score well without
                  ever consulting the contour channel.
    ``delta``     one pixel.  Sanity check only.

    Building it here rather than in the dataset keeps it free: it is a function
    of a coordinate the batch already carries.
    """
    device = bridge_xy_norm.device
    axis = torch.linspace(-1.0, 1.0, int(size), device=device)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    dx = grid_x[None] - bridge_xy_norm[:, 0, None, None]
    dy = grid_y[None] - bridge_xy_norm[:, 1, None, None]
    if kind == "delta":
        step = 2.0 / max(int(size) - 1, 1)
        out = ((dx.abs() <= step / 2) & (dy.abs() <= step / 2)).float()
    elif kind == "distance":
        out = torch.clamp(torch.sqrt(dx * dx + dy * dy) / np.sqrt(2.0), 0.0, 1.0)
    else:
        sigma = 2.0 * float(sigma_px) / max(int(size) - 1, 1)   # px -> normalised
        out = torch.exp(-0.5 * (dx * dx + dy * dy) / max(sigma ** 2, 1e-12))
    return out[:, None]


BRIDGE_CONDITIONING = ("scalar", "heatmap", "query", "query_heatmap")


class SpatialQueryNet(SurrogateBase):
    """Scalar-first prediction plus a spatial correction, with the bridge
    conditioned in one of four ways.

    ``scalar``         Diaz-style: the image is encoded to a global vector and
                       the bridge enters only as coordinates in the MLP.  The
                       shape encoding is computed once per case and reused.
    ``query``          the same global vector PLUS a bilinear read of the conv
                       feature maps at the bridge's own coordinate.  Keeps the
                       caching (the maps are per case, the read is per bridge)
                       and gives the bridge local context.
    ``heatmap``        the bridge is drawn INTO the image, so it interacts with
                       the boundary from the first convolution.  The CNN becomes
                       bridge-dependent, which costs ~10x the conv work per
                       TRAINING epoch (a case carries ten bridges).  It costs
                       nothing at deployment: the surrogate is called for one
                       body and one bridge at a time, so every mode runs the CNN
                       exactly once per request.
    ``query_heatmap``  both, and therefore the most expensive; only worth trying
                       if the two separately both look promising.

    The scalar path predicts first and the spatial head is zero-initialised, so
    training starts exactly at the current best model and any spatial
    contribution had to be earned.  Training them jointly from scratch is what
    let the earlier residual experiment's scalar branch absorb everything.
    """

    def __init__(self, *, shape_dim: int = 128, physics_dim: int = 64,
                 hidden_dim: int = 256, n_out: int = 500,
                 n_peaks: int = PEAK_SLOTS, pca: PCAOutput | None = None,
                 in_channels: int = 2, relational_dim: int = RELATIONAL_DIM,
                 bridge_conditioning: str = "query",
                 query_levels=(24, 12), bridge_map_type: str = "gaussian",
                 bridge_sigma_px: float = 3.0, coordconv: bool = False,
                 film: bool = True, scalar_hidden: int = 352,
                 scalar_depth: int = 4, aux_relational: bool = False,
                 drop_physics_inputs=None):
        super().__init__()
        if bridge_conditioning not in BRIDGE_CONDITIONING:
            raise ValueError(f"bridge_conditioning must be one of "
                             f"{BRIDGE_CONDITIONING}")
        self.bridge_conditioning = str(bridge_conditioning)
        self.bridge_map_type = str(bridge_map_type)
        self.bridge_sigma_px = float(bridge_sigma_px)
        self.query_levels = tuple(int(v) for v in query_levels)
        self.relational_dim = int(relational_dim)

        # -- scalar path: the current best model, kept whole -----------------
        scalar_in = PHYSICS_INPUT_DIM + self.relational_dim
        keep = torch.ones(scalar_in)
        if drop_physics_inputs:
            keep[physics_input_index(list(drop_physics_inputs))] = 0.0
        self.register_buffer("input_keep", keep, persistent=False)
        layers, width = [], scalar_in
        for _ in range(int(scalar_depth)):
            layers += [nn.Linear(width, scalar_hidden),
                       nn.LayerNorm(scalar_hidden), nn.ReLU(), nn.Dropout(0.2)]
            width = scalar_hidden
        self.scalar_trunk = nn.Sequential(*layers)
        self.scalar_out = nn.Linear(scalar_hidden, n_out)

        # -- spatial path ----------------------------------------------------
        uses_heatmap = self.bridge_conditioning in ("heatmap", "query_heatmap")
        self.condition_dim = (len(MATERIAL_COLS) + GEOMETRY_DIM
                              + BODY_TYPE_DIM) if film else None
        self.encoder = GeometryEncoder(
            in_channels=int(in_channels) + (1 if uses_heatmap else 0),
            out_dim=shape_dim, condition_dim=self.condition_dim,
            coordconv=coordconv)
        query_dim = 0
        if self.bridge_conditioning in ("query", "query_heatmap"):
            query_dim = sum(self.encoder.level_channels[level]
                            for level in self.query_levels)
        self.query_dim = query_dim
        self.physics_enc = PhysicsEncoder(in_dim=PHYSICS_INPUT_DIM,
                                          out_dim=physics_dim,
                                          drop_inputs=drop_physics_inputs)
        spatial_in = (shape_dim + query_dim + physics_dim + BRIDGE_DIM
                      + self.relational_dim)
        self.spatial_head = _decoder(spatial_in, hidden_dim, n_out)
        nn.init.zeros_(self.spatial_head[-1].weight)
        nn.init.zeros_(self.spatial_head[-1].bias)
        # Does the queried feature actually encode the relations the explicit
        # scalars carry?  Predicting them from it answers that directly, and
        # the answer stands whether or not the auxiliary term helps the spectrum.
        self.aux_head = (nn.Linear(query_dim + shape_dim, self.relational_dim)
                         if aux_relational and query_dim else None)
        self.peak_head = PeakHead(spatial_in, n_peaks, hidden_dim)
        self.pca = pca

    # -- pieces --------------------------------------------------------------
    def _scalars(self, inputs):
        value = torch.cat([inputs["material"], inputs["bridge"],
                           inputs["geometry"], inputs["body_type"],
                           inputs["bridge_extra"]], dim=-1)
        return self.scalar_trunk(value * self.input_keep)

    def _condition(self, inputs):
        if self.condition_dim is None:
            return None
        return torch.cat([inputs["material"], inputs["geometry"],
                          inputs["body_type"]], dim=-1)

    def encode_geometry(self, inputs):
        """Global vector and feature maps.  Independent of the bridge unless a
        heatmap is in use -- which is exactly what makes caching possible."""
        image = inputs["shape"]
        if self.bridge_conditioning in ("heatmap", "query_heatmap"):
            xy = inputs.get("query_xy")
            if xy is None:
                xy = inputs["bridge"][:, 3:5]
            image = torch.cat([image, bridge_map(
                xy, image.shape[-1],
                self.bridge_map_type, self.bridge_sigma_px)], dim=1)
        return self.encoder(image, self._condition(inputs))

    def _spatial_latent(self, inputs, encoded=None):
        z_global, maps = encoded if encoded is not None \
            else self.encode_geometry(inputs)
        pieces = [z_global]
        if self.query_dim:
            # ``query_xy`` lets an intervention move ONLY the point the feature
            # maps are read at, leaving the bridge scalars correct.  Without it
            # "the spatial query matters" cannot be separated from "the bridge
            # coordinates matter", because one substitution would change both.
            xy = inputs.get("query_xy")
            if xy is None:
                xy = inputs["bridge"][:, 3:5]
            pieces += [sample_at_bridge(maps[level], xy)
                       for level in self.query_levels]
        queried = torch.cat(pieces, dim=-1)
        fused = torch.cat([
            queried,
            self.physics_enc(inputs["material"], inputs["bridge"],
                             inputs["geometry"], inputs["body_type"]),
            inputs["bridge"], inputs["bridge_extra"]], dim=-1)
        return fused, queried

    def latents(self, inputs):
        fused, queried = self._spatial_latent(inputs)
        return queried, self._scalars(inputs), fused

    def parts(self, inputs):
        """``(scalar_prediction, spatial_correction, fused)``."""
        fused, _queried = self._spatial_latent(inputs)
        return self.scalar_out(self._scalars(inputs)), \
            self.spatial_head(fused), fused

    def auxiliary(self, inputs):
        """Relational scalars predicted FROM the queried spatial feature."""
        if self.aux_head is None:
            return None
        _fused, queried = self._spatial_latent(inputs)
        return self.aux_head(queried)

    def _head(self, inputs):
        scalar, spatial, fused = self.parts(inputs)
        return scalar + spatial, fused

    def freeze_scalar(self) -> int:
        """Freeze the scalar path so the spatial head cannot be compensated for."""
        frozen = 0
        for module in (self.scalar_trunk, self.scalar_out):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        return frozen


class ScalarOnlyNet(SurrogateBase):
    """Scalars only, at a chosen capacity, optionally with relational features.

    Two controls in one class:

      * ``physics_wide`` -- the same inputs as ``physics_only`` but sized to
        match the FiLM model.  If a wide scalar model reaches what the
        FiLM-with-scrambled-raster model reached, that variant's advantage was
        capacity, not conditioning, and no image was involved either way.
      * ``relational`` -- adds the bridge/hole/edge distance scalars.  A spatial
        CNN branch has to beat these before its cost is justified: if a handful
        of distances captures the relationship, the relationship was what was
        missing, not the picture.
    """

    def __init__(self, *, hidden_dim: int = 512, depth: int = 4,
                 n_out: int = 500, n_peaks: int = PEAK_SLOTS,
                 pca: PCAOutput | None = None, use_relational: bool = False,
                 relational_dim: int = RELATIONAL_DIM,
                 relational_names=None, drop_relational=None,
                 keep_only_relational=None, drop_physics_inputs=None,
                 decoder=None):
        super().__init__()
        self.use_relational = bool(use_relational)
        self.relational_dim = int(relational_dim) if use_relational else 0
        in_dim = PHYSICS_INPUT_DIM + self.relational_dim
        self.drop_inputs = list(drop_physics_inputs or [])
        keep = torch.ones(in_dim)
        if self.drop_inputs:
            keep[physics_input_index(self.drop_inputs)] = 0.0
        # Relational columns are suppressed by NAME, and only by zeroing, so
        # every ablation of this block keeps identical layer widths and
        # identical capacity.  A leave-one-out that also shrank the first layer
        # would confound "this feature matters" with "this model is smaller".
        names = list(relational_names or [])[:self.relational_dim]
        self.relational_names = names
        self.drop_relational = list(drop_relational or [])
        self.keep_only_relational = (list(keep_only_relational)
                                     if keep_only_relational else None)
        if self.use_relational and names:
            for index, name in enumerate(names):
                dropped = name in self.drop_relational
                if self.keep_only_relational is not None:
                    dropped = name not in self.keep_only_relational
                if dropped:
                    keep[PHYSICS_INPUT_DIM + index] = 0.0
        elif self.drop_relational or self.keep_only_relational:
            raise ValueError("relational ablation needs relational_names and "
                             "use_relational=True")
        self.register_buffer("input_keep", keep, persistent=False)
        layers, width = [], in_dim
        for _ in range(int(depth)):
            layers += [nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU(), nn.Dropout(0.2)]
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        # ``out`` stays for checkpoint compatibility with every run trained
        # before the decoders existed; when a decoder is attached it is what
        # produces the spectrum and ``out`` is unused.
        self.out = nn.Linear(hidden_dim, n_out)
        if decoder is not None:
            self.decoder = decoder
        self.peak_head = PeakHead(hidden_dim, n_peaks, hidden_dim)
        self.pca = pca

    def _fuse(self, inputs):
        pieces = [inputs["material"], inputs["bridge"], inputs["geometry"],
                  inputs["body_type"]]
        if self.use_relational:
            pieces.append(inputs["bridge_extra"])
        return self.trunk(torch.cat(pieces, dim=-1) * self.input_keep)

    def latents(self, inputs):
        z = self._fuse(inputs)
        return z[:, :0], z, z

    def _head(self, inputs):
        fused = self._fuse(inputs)
        decoder = self._modules.get("decoder")
        if decoder is not None:
            return decoder(fused)["spectrum"], fused
        return self.out(fused), fused


class CaseBridgeNet(SurrogateBase):
    """A case-level response plus a per-bridge correction.

    Ten bridges share one body, one material and one mesh; what actually changes
    between them is where the plate is driven.  A flat per-row model has to
    rediscover the shared part ten times and has no structural reason to keep it
    shared, so bridge-to-bridge differences -- the thing a luthier moves the
    bridge to change -- are free to be absorbed into noise.

    Here the case path sees only case-level inputs, so its output is CONSTANT
    across the bridges of a case by construction, and the bridge path predicts
    the departure from it.  The residual is centred inside each case during the
    loss (``zero_mean`` in nn_train), which is what stops the two paths from
    trading a constant back and forth.

    Rows stay flat: grouping happens through ``case_index`` in the loss, so
    every existing metric and eval path is untouched.
    """

    def __init__(self, *, shape_dim: int = 128, physics_dim: int = 64,
                 hidden_dim: int = 256, n_out: int = 500,
                 n_peaks: int = PEAK_SLOTS, pca: PCAOutput | None = None,
                 in_channels: int = 1,
                 bridge_extra_dim: int = RELATIONAL_DIM,
                 film: bool = True, drop_physics_inputs=None):
        super().__init__()
        self.condition_dim = (len(MATERIAL_COLS) + GEOMETRY_DIM
                              + BODY_TYPE_DIM) if film else None
        self.shape_enc = ConditionedShapeEncoder(
            out_dim=shape_dim, condition_dim=self.condition_dim,
            in_channels=in_channels)
        # Case-level physics: material, geometry, body type -- NO bridge.  The
        # bridge columns are zeroed rather than sliced out so the encoder keeps
        # the standard PHYSICS_INPUT_DIM width and the same code path.
        self.case_physics = PhysicsEncoder(
            in_dim=PHYSICS_INPUT_DIM, out_dim=physics_dim,
            drop_inputs=sorted(set(BRIDGE_NAMES)
                               | set(drop_physics_inputs or [])))
        self.case_head = _decoder(shape_dim + physics_dim, hidden_dim, n_out)
        self.bridge_path = nn.Sequential(
            nn.Linear(BRIDGE_DIM + int(bridge_extra_dim), 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.bridge_head = _decoder(shape_dim + physics_dim + 64, hidden_dim,
                                    n_out)
        # Starts at zero: the model begins as a pure case-level predictor that
        # gives every bridge of a case the same answer, and any bridge-to-bridge
        # structure it grows had to be learned.
        nn.init.zeros_(self.bridge_head[-1].weight)
        nn.init.zeros_(self.bridge_head[-1].bias)
        self.peak_head = PeakHead(shape_dim + physics_dim + 64, n_peaks,
                                  hidden_dim)
        self.pca = pca

    def _condition(self, inputs):
        if self.condition_dim is None:
            return None
        return torch.cat([inputs["material"], inputs["geometry"],
                          inputs["body_type"]], dim=-1)

    def latents(self, inputs):
        z_shape = self.shape_enc(inputs["shape"], self._condition(inputs))
        z_physics = self.case_physics(
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"])
        z_bridge = self.bridge_path(
            torch.cat([inputs["bridge"], inputs["bridge_extra"]], dim=-1))
        return z_shape, z_physics, torch.cat([z_shape, z_physics, z_bridge],
                                             dim=-1)

    def parts(self, inputs):
        """``(case_mean, bridge_residual, fused)``.

        ``case_mean`` is constant within a case because nothing that varies
        within a case reaches it -- that is the whole point, and it is enforced
        by construction rather than by a penalty.
        """
        z_shape, z_physics, fused = self.latents(inputs)
        case = self.case_head(torch.cat([z_shape, z_physics], dim=-1))
        return case, self.bridge_head(fused), fused

    def _head(self, inputs):
        case, residual, fused = self.parts(inputs)
        return case + residual, fused


class AdmittanceNetPhysicsOnly(nn.Module):
    """Baseline using all scalar physics inputs but no contour image.

    Carries the same peak head as ``AdmittanceNet`` so the ablation isolates the
    contour image rather than confounding it with a different training objective.
    """

    def __init__(self, hidden_dim: int = 128, n_freq: int = 500,
                 n_peaks: int = PEAK_SLOTS, drop_physics_inputs=None):
        super().__init__()
        self.drop_inputs = list(drop_physics_inputs or [])
        keep = torch.ones(PHYSICS_INPUT_DIM)
        if self.drop_inputs:
            keep[physics_input_index(self.drop_inputs)] = 0.0
        self.register_buffer("input_keep", keep, persistent=False)
        self.trunk = nn.Sequential(
            nn.Linear(PHYSICS_INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, n_freq)
        self.peak_head = PeakHead(hidden_dim, n_peaks, hidden_dim)

    def _fuse(self, inputs):
        value = torch.cat([
            inputs["material"], inputs["bridge"], inputs["geometry"],
            inputs["body_type"],
        ], dim=-1)
        return self.trunk(value * self.input_keep)

    def latents(self, inputs):
        """``(z_shape, z_physics, z_fused)`` with an EMPTY shape latent."""
        z = self._fuse(inputs)
        return z[:, :0], z, z

    def forward(self, inputs):
        return self.out(self._fuse(inputs))

    def forward_all(self, inputs):
        fused = self._fuse(inputs)
        return self.out(fused), self.peak_head(fused)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)
