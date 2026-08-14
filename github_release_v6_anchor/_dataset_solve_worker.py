"""
_dataset_solve_worker.py
------------------------
Per-case FEM solve worker for dataset_gen_mixed.  Run as a SEPARATE process
(one subprocess per shape x body x material case) so that when it exits the OS
reclaims ALL PETSc/SLEPc/MUMPS memory — the orchestrator never accumulates
solver state across the ~2000 cases.

    python _dataset_solve_worker.py <spec.json>

`spec.json` (written by dataset_gen_mixed._real_solve_case) fully describes ONE
case: body type, the (already-built, persistent) mesh, the material, the exact
common frequency grid, the driven bridge points, and the solver knobs.  The
worker writes its artifacts into spec["out_dir"] (a per-attempt STAGING dir that
the orchestrator commits only after QC) and finally writes worker_result.json:

    { "run_status": "complete"|"failed", "solver_revision": ...,
      "multi_npz": <filename>, "peak_npz": <filename>, "bridges": [...],
      "timing": {...}, "A0": {...}, "error": <str|null> }

The worker is SERIAL by contract (the streaming realifier fetches whole
eigenvectors on rank 0); the underlying solvers fail loud if launched multi-rank.
D full-coupled and the D/E comparison are never invoked here.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

SOLID_SOLVER_REVISION = "structural-full-harmonic-peaks-v1"
HOLLOW_SOLVER_REVISION = "modal-coupled-real-basis-v2"


def _finite_value(mapping: dict, key: str, *, nonnegative: bool = False) -> float:
    if not isinstance(mapping, dict) or key not in mapping:
        raise RuntimeError(f"solver metadata missing {key}")
    try:
        value = float(mapping[key])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"solver metadata {key} is not numeric") from exc
    if not np.isfinite(value) or (nonnegative and value < 0.0):
        raise RuntimeError(f"solver metadata {key} is non-finite or invalid")
    return value


def _integer_value(mapping: dict, key: str, *, nonnegative: bool = False) -> int:
    if not isinstance(mapping, dict) or key not in mapping:
        raise RuntimeError(f"solver metadata missing {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"solver metadata {key} is not an integer")
    if nonnegative and value < 0:
        raise RuntimeError(f"solver metadata {key} is negative")
    return value


def _boolean_value(mapping: dict, key: str) -> bool:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), bool):
        raise RuntimeError(f"solver metadata {key} is not a boolean")
    return mapping[key]


def _text_value(mapping: dict, key: str) -> str:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), str):
        raise RuntimeError(f"solver metadata {key} is not text")
    return mapping[key]


def _same_scalar(a, b) -> bool:
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return a == b
    return bool(np.isfinite(af) and np.isfinite(bf)
                and np.isclose(af, bf, rtol=1e-12, atol=0.0))


def _solve_solid(spec: dict, out_dir: Path) -> dict:
    from fenics_admittance import compute_admittance_full_with_eigen_peaks
    freqs = np.asarray(spec["freqs"], float)
    s = spec["solid"]
    contract = spec["production_contract"]
    res = compute_admittance_full_with_eigen_peaks(
        spec["msh_path"], spec["material"],
        bridge_coords=spec["bridge_coords"][0],
        bridge_points=spec["bridge_coords"],
        output_dir=str(out_dir),
        eigen_fmax=float(s["modal_fmax"]),
        n_modes=int(s["n_modes"]),
        rayleigh_alpha=float(contract["rayleigh_alpha"]),
        rayleigh_beta=float(contract["rayleigh_beta"]),
        rigid_freq_hz=float(contract["rigid_frequency_hz"]),
        force_n=float(contract["force_n"]),
        freqs=freqs,
        top_k=int(contract["top_k_peaks"]),
        bulk=True,
    )
    meta = json.loads((out_dir / "admittance_meta.json").read_text(encoding="utf-8"))
    coverage = meta.get("coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("solid solver metadata has no coverage object")
    eb = coverage.get("eigenbasis_diagnostics")
    if not isinstance(eb, dict):
        raise RuntimeError("solid solver metadata has no eigenbasis diagnostics")
    revision = _text_value(coverage, "solver_revision")
    if res.get("solver_revision") != revision:
        raise RuntimeError("solid solver revision disagrees with its metadata")
    if res.get("eigenbasis_diagnostics") != eb:
        raise RuntimeError("solid eigenbasis return value disagrees with metadata")
    timing = meta.get("timing")
    if not isinstance(timing, dict) or timing != res.get("timing"):
        raise RuntimeError("solid timing return value disagrees with metadata")
    diagnostics = {
        "coverage_ok": _boolean_value(coverage, "coverage_ok"),
        "eig_freq_max_hz": _finite_value(coverage, "eig_freq_max_hz",
                                           nonnegative=True),
        "n_modes_retained": _integer_value(coverage, "n_modes_retained",
                                             nonnegative=True),
        "n_eig_converged": _integer_value(coverage, "n_eig_converged",
                                            nonnegative=True),
        "modal_fmax_hz": _finite_value(coverage, "eigen_fmax",
                                         nonnegative=True),
        "coverage_boundary_rtol": _finite_value(
            coverage, "coverage_boundary_rtol", nonnegative=True),
        "analysis_freq_min_hz": _finite_value(coverage, "freq_min",
                                                nonnegative=True),
        "analysis_freq_max_hz": _finite_value(coverage, "freq_max",
                                                nonnegative=True),
        "analysis_freq_points": _integer_value(coverage, "freq_points",
                                                 nonnegative=True),
        "damping": _text_value(coverage, "damping"),
        "structural_mass_orthonormality_max_dev": _finite_value(
            eb, "mass_orthonormality_max_dev", nonnegative=True),
        "max_eigen_residual": _finite_value(
            eb, "max_eigen_residual", nonnegative=True),
        "n_grouped_in_band_modes": _integer_value(
            coverage, "n_grouped_in_band_modes", nonnegative=True),
        "n_exact_candidate_frequencies": _integer_value(
            coverage, "n_exact_candidate_frequencies", nonnegative=True),
    }
    bridges = [{"bridge_requested_xyz": [float(v) for v in m["bridge_requested_xyz"]],
                "bridge_snapped_xyz": [float(v) for v in m["bridge_snapped_xyz"]],
                "snap_distance_mm": float(m["snap_distance_mm"])}
               for m in res["bridge_meta"]]
    return {"run_status": "complete",
            "solver_revision": revision,
            "multi_npz": "admittance_harmonic_multi.npz",
            "peak_npz": "peak_labels.npz",
            "bridges": bridges, "diagnostics": diagnostics,
            "timing": timing, "A0": {}}


def _solve_hollow(spec: dict, out_dir: Path) -> dict:
    from modal_coupled_admittance import compute_modal_coupled_admittance
    freqs = np.asarray(spec["freqs"], float)
    h = spec["hollow"]
    contract = spec["production_contract"]
    res = compute_modal_coupled_admittance(
        spec["msh_path"], spec["material"],
        bridge_coords=spec["bridge_coords"][0],
        bridge_points=spec["bridge_coords"],
        output_dir=str(out_dir),
        struct_fmax=float(h["struct_fmax"]),
        acoustic_fmax=float(h["acoustic_fmax"]),
        n_struct_modes=int(h["n_struct_modes"]),
        n_acoustic_modes=int(h["n_acoustic_modes"]),
        n_attach_acoustic=int(h["n_attach_acoustic"]),
        port_end_corrections=int(h["port_end_corrections"]),
        t_hole_mm=h.get("t_hole_mm"),
        basis=str(contract["structural_basis"]),
        acoustic_attachment=str(contract["acoustic_attachment"]),
        static_solver=str(contract["static_solver"]),
        rayleigh_alpha=float(contract["rayleigh_alpha"]),
        rayleigh_beta=float(contract["rayleigh_beta"]),
        rigid_freq_hz=float(contract["rigid_frequency_hz"]),
        c=float(contract["air_speed_m_s"]),
        rho0=float(contract["air_density_kg_m3"]),
        force_n=float(contract["force_n"]),
        freq_min=float(freqs[0]),
        freq_max=float(freqs[-1]),
        freq_points=int(freqs.size),
        freqs=freqs,
        bulk=True,
    )
    # Pull audited diagnostics from persisted sidecars and require the live return
    # value to agree.  Never turn a malformed string/int into a plausible boolean.
    meta = json.loads((out_dir / "modal_coupled_metadata.json").read_text(
        encoding="utf-8"))
    revision = _text_value(meta, "solver_revision")
    if res.get("solver_revision") != revision:
        raise RuntimeError("hollow solver revision disagrees with its metadata")
    for key in ("band_covered", "coverage_ok", "cutoff_reached"):
        if _boolean_value(meta, key) is not res.get(key):
            raise RuntimeError(f"hollow solver {key} disagrees with metadata")
    structural_eb = meta.get("structural_eigenbasis_diagnostics")
    acoustic_eb = meta.get("acoustic_eigenbasis_diagnostics")
    if not isinstance(structural_eb, dict) or not isinstance(acoustic_eb, dict):
        raise RuntimeError("hollow eigenbasis diagnostics are missing")
    port_diag = meta.get("acoustic_port_attachment_diag")
    if not isinstance(port_diag, dict):
        raise RuntimeError("hollow port-attachment diagnostics are missing")
    timing = json.loads((out_dir / "timing.json").read_text(encoding="utf-8"))
    if not isinstance(timing, dict) or timing != res.get("timing"):
        raise RuntimeError("hollow timing return value disagrees with metadata")
    diagnostics = {
        "band_covered": _boolean_value(meta, "band_covered"),
        "coverage_ok": _boolean_value(meta, "coverage_ok"),
        "cutoff_reached": _boolean_value(meta, "cutoff_reached"),
        "structural_mass_orthonormality_max_dev": _finite_value(
            meta, "structural_mass_orthonormality_max_dev", nonnegative=True),
        "acoustic_mass_orthonormality_max_dev": _finite_value(
            meta, "acoustic_mass_orthonormality_max_dev", nonnegative=True),
        "structural_reduced_mass_condition": _finite_value(
            meta, "structural_reduced_mass_condition", nonnegative=True),
        "acoustic_reduced_mass_condition": _finite_value(
            meta, "acoustic_reduced_mass_condition", nonnegative=True),
        "max_eigen_residual": max(
            _finite_value(structural_eb, "max_eigen_residual", nonnegative=True),
            _finite_value(acoustic_eb, "max_eigen_residual", nonnegative=True)),
        "structural_eigen_residual": _finite_value(
            structural_eb, "max_eigen_residual", nonnegative=True),
        "acoustic_eigen_residual": _finite_value(
            acoustic_eb, "max_eigen_residual", nonnegative=True),
        "cavity_volume_m3": _finite_value(meta, "cavity_volume_m3",
                                            nonnegative=True),
        "soundhole_area_m2": _finite_value(meta, "soundhole_area_m2",
                                             nonnegative=True),
        "basis": _text_value(meta, "structural_basis"),
        "acoustic_port_attachment": _boolean_value(
            meta, "acoustic_port_attachment"),
        "acoustic_port_attachment_diag": {
            "residual_rel": _finite_value(port_diag, "residual_rel",
                                           nonnegative=True),
            "m_orthogonality_to_psi0": _finite_value(
                port_diag, "m_orthogonality_to_psi0", nonnegative=True),
        },
        "struct_fmax_hz": _finite_value(meta, "struct_fmax_hz",
                                          nonnegative=True),
        "acoustic_fmax_hz": _finite_value(meta, "acoustic_fmax_hz",
                                            nonnegative=True),
        "eig_freq_max_struct_hz": _finite_value(
            meta, "eig_freq_max_struct_hz", nonnegative=True),
        "eig_freq_max_acoustic_hz": _finite_value(
            meta, "eig_freq_max_acoustic_hz", nonnegative=True),
        "basis_frequency_limit_struct_hz": _finite_value(
            meta, "basis_frequency_limit_struct_hz", nonnegative=True),
        "basis_frequency_limit_acoustic_hz": _finite_value(
            meta, "basis_frequency_limit_acoustic_hz", nonnegative=True),
        "basis_band_target_hz": _finite_value(
            meta, "basis_band_target_hz", nonnegative=True),
        "basis_band_margin": _finite_value(meta, "basis_band_margin",
                                             nonnegative=True),
        "basis_coverage_boundary_rtol": _finite_value(
            meta, "basis_coverage_boundary_rtol", nonnegative=True),
        "beta0_vs_S_over_sqrtV_rel": _finite_value(
            meta, "beta0_vs_S_over_sqrtV_rel", nonnegative=True),
        "zero_mode_stiffness_rel": _finite_value(
            meta, "zero_mode_stiffness_rel", nonnegative=True),
        "n_attach_acoustic_used": _integer_value(
            meta, "n_attach_acoustic_used", nonnegative=True),
        "port_end_corrections": _integer_value(
            meta, "port_end_corrections", nonnegative=True),
        "static_solver": _text_value(meta, "static_solver"),
        "n_attach_acoustic_requested": _integer_value(
            meta, "n_attach_acoustic_requested", nonnegative=True),
        "analysis_freq_min_hz": _finite_value(meta, "analysis_freq_min_hz",
                                                nonnegative=True),
        "analysis_freq_max_hz": _finite_value(meta, "analysis_freq_max_hz",
                                                nonnegative=True),
        "analysis_freq_points": _integer_value(meta, "analysis_freq_points",
                                                 nonnegative=True),
        "model_reconstruction_max_rel_error": _finite_value(
            meta, "model_reconstruction_max_rel_error", nonnegative=True),
    }
    bridges = [{"bridge_requested_xyz": [float(v) for v in br["bridge_requested_xyz"]],
                "bridge_snapped_xyz": [float(v) for v in br["bridge_snapped_xyz"]],
                "snap_distance_mm": float(br["snap_distance_mm"])}
               for br in res["bridges"]]
    a0_meta = json.loads((out_dir / "A0_estimated_vs_observed.json").read_text(
        encoding="utf-8"))
    detected = _boolean_value(a0_meta, "A0_detected")
    estimated = _finite_value(a0_meta, "A0_estimated_hz", nonnegative=True)
    observed = a0_meta.get("A0_observed_hz")
    if detected:
        observed = _finite_value(a0_meta, "A0_observed_hz", nonnegative=True)
    elif observed is not None:
        raise RuntimeError("undetected A0 has a non-null observed frequency")
    if (res.get("A0_detected") is not detected
            or not _same_scalar(res.get("A0_estimated"), estimated)
            or (detected and not _same_scalar(res.get("A0_observed"), observed))
            or (not detected and res.get("A0_observed") is not None)):
        raise RuntimeError("hollow A0 return value disagrees with metadata")

    # The port radiation impedance is frequency-dependent, so the reduced K/M
    # matrices do not define a valid ordinary pole problem.  Measure response
    # peaks and half-power Q on a dense reduced-model grid instead.
    from peak_labels import (hollow_peak_search_grid, response_peak_labels,
                             validate_peak_batch)
    from reduced_model_io import resample_air_coupled
    with np.load(out_dir / "reduced_model.npz", allow_pickle=False) as model:
        structural_hints = np.asarray(model["omega_s_modes_rad_s"], float) / (2.0 * np.pi)
        acoustic_hints = np.asarray(model["omega_a_elastic_rad_s"], float) / (2.0 * np.pi)
    hints = np.concatenate([
        structural_hints, acoustic_hints,
        np.asarray([estimated] + ([observed] if detected else []), float),
    ])
    t_peak = time.time()
    peak_grid = hollow_peak_search_grid(
        float(freqs[0]), float(freqs[-1]),
        n_points=int(contract["hollow_peak_search_points"]), hints_hz=hints)
    peak_response = resample_air_coupled(out_dir / "reduced_model.npz", peak_grid)
    peak_batch = response_peak_labels(
        peak_grid, peak_response["Y"], top_k=int(contract["top_k_peaks"]),
        prominence_db=float(contract["peak_min_prominence_db"]))
    peak_payload = peak_batch.as_dict()
    peak_reasons = validate_peak_batch(
        peak_payload, len(bridges), int(contract["top_k_peaks"]),
        float(freqs[0]), float(freqs[-1]),
        min_count=int(contract["min_peak_count_per_bridge"]))
    if peak_reasons:
        raise RuntimeError("invalid hollow peak labels: " + "; ".join(peak_reasons))
    np.savez_compressed(out_dir / "peak_labels.npz", **peak_payload)
    np.savez_compressed(
        out_dir / "peak_search_response.npz",
        frequencies=peak_grid, admittance=peak_response["Y"])
    timing = dict(timing)
    timing["peak_labels_s"] = float(time.time() - t_peak)
    timing["total_s"] = float(timing.get("total_s", 0.0) + timing["peak_labels_s"])
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2),
                                          encoding="utf-8")
    return {"run_status": "complete",
            "solver_revision": revision,
            "multi_npz": "admittance_air_modal_coupled_multi.npz",
            "peak_npz": "peak_labels.npz",
            "bridges": bridges, "diagnostics": diagnostics,
            "timing": timing,
            "A0": {"A0_detected": detected,
                   "A0_observed_hz": observed,
                   "A0_estimated_hz": estimated}}


def main(spec_path: str) -> int:
    spec_file = Path(spec_path).resolve()
    spec = json.loads(spec_file.read_text())
    for key in ("msh_path", "out_dir"):
        value = Path(spec[key])
        if not value.is_absolute():
            spec[key] = str((spec_file.parent / value).resolve())
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if spec["body_type"] == "hollow":
            result = _solve_hollow(spec, out_dir)
        elif spec["body_type"] == "solid":
            result = _solve_solid(spec, out_dir)
        else:
            raise ValueError(f"unsupported body_type {spec.get('body_type')!r}")
    except Exception:
        result = {"run_status": "failed", "solver_revision": None,
                  "multi_npz": None, "peak_npz": None,
                  "bridges": [], "diagnostics": {},
                  "timing": {}, "A0": {},
                  "error": traceback.format_exc()[-2000:]}
    # Echo the orchestrator's canonical input identity.  The committed solve_spec is
    # independently revalidated on resume, so a wrong-material/wrong-mesh result
    # cannot be accepted merely because its numerical arrays are finite.
    result["case_input_hash"] = spec.get("case_input_hash")
    try:
        payload = json.dumps(result, indent=2, allow_nan=False)
    except (TypeError, ValueError):
        # A solver may return NaN/Inf inside diagnostics or timing.  That is a failed
        # case, but the orchestrator still needs a valid JSON result explaining why.
        result = {
            "run_status": "failed", "solver_revision": None,
            "multi_npz": None, "peak_npz": None,
            "bridges": [], "diagnostics": {},
            "timing": {}, "A0": {},
            "case_input_hash": spec.get("case_input_hash"),
            "error": "worker result contained non-JSON or non-finite metadata",
        }
        payload = json.dumps(result, indent=2, allow_nan=False)
    (out_dir / "worker_result.json").write_text(
        payload, encoding="utf-8")
    return 0 if result["run_status"] == "complete" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python _dataset_solve_worker.py <spec.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
