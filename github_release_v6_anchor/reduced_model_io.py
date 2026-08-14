"""Load and resample the compact raw models saved by the dataset solvers.

The fixed 500-point response is a convenient preview/initial NN target, not a
peak-resolved audio representation.  These helpers reconstruct the same reduced
transfer functions on any strictly increasing frequency grid without another
FEM eigensolve or static solve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _validated_freqs(freqs) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    if (f.ndim != 1 or f.size < 2 or not np.all(np.isfinite(f))
            or f[0] <= 0.0 or np.any(np.diff(f) <= 0.0)):
        raise ValueError("freqs must be a finite, positive, strictly increasing 1-D array")
    return f


def resample_structural_modal(model_path, freqs) -> dict:
    """Reconstruct every solid driving-point mobility from ``modal_model.npz``."""
    freqs = _validated_freqs(freqs)
    with np.load(Path(model_path), allow_pickle=False) as z:
        if str(np.asarray(z["schema_version"]).item()) != "structural-modal-model-v1":
            raise ValueError("unsupported structural modal-model schema")
        wn = np.asarray(z["omega_n_rad_s"], float)
        residues = np.asarray(z["bridge_residue_inv_kg"], float)
        zeta = np.asarray(z["zeta"], float)
        residual = (np.asarray(z["residual_compliance_m_per_n"], float)
                    if "residual_compliance_m_per_n" in z.files
                    else np.zeros(residues.shape[0], float))
    if (wn.ndim != 1 or wn.size == 0 or residues.ndim != 2
            or residues.shape[1] != wn.size or zeta.shape != wn.shape
            or not np.all(np.isfinite(wn)) or not np.all(np.isfinite(residues))
            or not np.all(np.isfinite(zeta)) or np.any(wn < 0.0)
            or np.any(residues < 0.0) or np.any(zeta < 0.0)):
        raise ValueError("invalid structural modal model")
    if residual.shape != (residues.shape[0],) or not np.all(np.isfinite(residual)):
        raise ValueError("invalid structural residual compliance")
    w = 2.0 * np.pi * freqs
    denom = (wn[None, :] ** 2 - w[:, None] ** 2
             + 1j * 2.0 * zeta[None, :] * wn[None, :] * w[:, None])
    Y = 1j * w[None, :] * np.sum(
        residues[:, None, :] / denom[None, :, :], axis=2)
    Y += 1j * w[None, :] * residual[:, None]
    if not np.all(np.isfinite(Y)):
        raise RuntimeError("structural modal resampling produced non-finite values")
    return {"freqs": freqs, "Y": Y}


def resample_air_coupled(model_path, freqs) -> dict:
    """Re-solve every hollow reduced response from ``reduced_model.npz``."""
    freqs = _validated_freqs(freqs)
    with np.load(Path(model_path), allow_pickle=False) as z:
        if str(np.asarray(z["schema_version"]).item()) != "air-coupled-reduced-model-v1":
            raise ValueError("unsupported air-coupled reduced-model schema")
        values = {k: np.asarray(z[k]).copy() for k in (
            "K_r", "M_r", "Ka_r", "Ma_r", "G_r", "beta_m", "q_b")}
        scalars = {k: float(np.asarray(z[k]).item()) for k in (
            "soundhole_area_m2", "soundhole_inertance_kg_m4",
            "air_speed_m_s", "air_density_kg_m3", "rayleigh_alpha",
            "rayleigh_beta", "force_n")}
    if (not all(a.size and np.all(np.isfinite(a)) for a in values.values())
            or not all(np.isfinite(v) for v in scalars.values())
            or scalars["soundhole_area_m2"] <= 0.0
            or scalars["soundhole_inertance_kg_m4"] <= 0.0
            or scalars["air_speed_m_s"] <= 0.0
            or scalars["air_density_kg_m3"] <= 0.0
            or scalars["force_n"] <= 0.0):
        raise ValueError("invalid air-coupled reduced model")
    from modal_coupled_admittance import reduced_coupled_sweep_general
    out = reduced_coupled_sweep_general(
        freqs, values["K_r"], values["M_r"], values["Ka_r"], values["Ma_r"],
        values["G_r"], values["beta_m"], values["q_b"],
        scalars["soundhole_area_m2"], scalars["soundhole_inertance_kg_m4"],
        scalars["rayleigh_alpha"], scalars["rayleigh_beta"],
        c=scalars["air_speed_m_s"], rho0=scalars["air_density_kg_m3"],
        force_n=scalars["force_n"])
    for key in ("Y", "p_bar", "U_h"):
        if not np.all(np.isfinite(out[key])):
            raise RuntimeError(f"air-coupled resampling produced non-finite {key}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Resample a compact FEM reduced model")
    p.add_argument("model", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--freq-min", type=float, default=20.0)
    p.add_argument("--freq-max", type=float, default=5000.0)
    p.add_argument("--freq-points", type=int, default=5000)
    p.add_argument("--grid", choices=("linear", "log"), default="log")
    args = p.parse_args()
    if args.freq_points < 2 or args.freq_min <= 0 or args.freq_max <= args.freq_min:
        p.error("invalid frequency grid")
    maker = np.linspace if args.grid == "linear" else np.geomspace
    freqs = maker(args.freq_min, args.freq_max, args.freq_points)
    with np.load(args.model, allow_pickle=False) as z:
        schema = str(np.asarray(z["schema_version"]).item())
    if schema == "structural-modal-model-v1":
        out = resample_structural_modal(args.model, freqs)
        arrays = {"frequencies": out["freqs"], "admittance": out["Y"]}
    elif schema == "air-coupled-reduced-model-v1":
        out = resample_air_coupled(args.model, freqs)
        arrays = {"frequencies": out["freqs"], "admittance": out["Y"],
                  "p_bar": out["p_bar"], "U_h": out["U_h"]}
    else:
        raise ValueError(f"unsupported model schema {schema!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"saved {args.output} ({args.freq_points} {args.grid} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
