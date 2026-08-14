"""
dataset_gen_mixed.py
--------------------
Final MIXED guitar bridge-admittance dataset generator.

Design (a base contour is realised as BOTH a solid and a hollow body):
  * solid   -> full harmonic FEM magnitude + phase-safe eigenmode peak labels
  * hollow  -> modal_coupled_admittance ("E", air-coupled, craig-bampton + port)
  * D full-coupled and the D/E validators are NOT used for bulk generation.
  * dataset_gen.py --method modal (structural-only) is NOT used for hollow.

Robustness contract (the reason this is a separate generator):
  * an IMMUTABLE plan (dataset_plan.json + SHA-256 plan_hash) is written BEFORE
    any FEM; a mismatching plan/config on the same output root aborts.
  * stable IDs from (base_shape_id, body_type, material_id, bridge_idx) — never
    from sequential RNG consumption order.
  * a "case" = shape x body x material; its bridges are one batched solve.
  * temp-dir write + QC + ATOMIC commit; manifest.csv is REGENERATED from case
    statuses and atomically replaced (never appended).
  * fail-closed QC: a QC failure never marks a bridge sample done.
  * shard by SHAPE so parallel workers never contend on a mesh.

FEM/geometry are isolated behind an injectable `GenContext` so the whole
orchestration (plan, ids, split, shard, QC, fan-out, resume, manifest) is
unit-testable with FAKE solver artifacts and NO FEniCSx (test_dataset_gen_mixed).

This module does NOT run the bulk dataset itself; see the production command in
the module docstring of the review prompt / README.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# ---------------------------------------------------------------------------
# Constants / contract
# ---------------------------------------------------------------------------
GENERATOR_VERSION = "mixed-v6"
SOLID_SOLVER_REVISION = "structural-full-harmonic-peaks-v1"
HOLLOW_SOLVER_REVISION = "modal-coupled-real-basis-v2"
# --- material sampling contract ---------------------------------------------
# v5 and earlier: one bank of n_materials (material_000 = Engelmann Spruce fixed,
# the rest random SPD orthotropic) was SHARED by every base shape, so the whole
# dataset contained only n_materials distinct material vectors.  v6 raises material
# diversity for the NN surrogate: EACH base shape draws its own n_materials random
# SPD orthotropic materials (no fixed reference material), giving up to
# n_base_shapes * n_materials globally-unique materials.  Ids are global + carry
# base_shape_id so plan-level material lookup stays unambiguous.

# --- top-plate physics contract (v5) ---------------------------------------
# v4 derived the top plate as body_thickness*(1-cavity_ratio) with
# cavity_ratio ~ U(0.5,0.9), giving a 4.2--25.3 mm top plate (86/100 shapes > 8 mm)
# — far too thick for a real hollow guitar.  v5 samples the TOP PLATE THICKNESS
# directly in a physically realistic archtop range and DERIVES the cavity ratio:
#     top_plate_thickness_mm ~ Uniform(3.0, 6.5)
#     cavity_ratio = 1 - top_plate_thickness_mm / body_thickness   (derived, once)
# (archtop reference: center 6.4, sides 4.7, recurve 3.2 mm; FEM+NN reference 6 mm +-2 mm.)
TOP_PLATE_MIN_MM = 3.0
TOP_PLATE_MAX_MM = 6.5

# Bridge points only need to stay OUT of the soundhole; this is the clearance
# between the hole rim and a bridge point (mm).  (Replaces the old fixed 40 mm
# bridge clearance, which over-constrained placement once n_bridge_points grew.)
BRIDGE_SOUNDHOLE_MARGIN_MM = 5.0

PRODUCTION_CONTRACT = {
    "geometry_revision": "direct-contour-conformal-v3-top-plate",
    "top_plate_thickness_min_mm": TOP_PLATE_MIN_MM,
    "top_plate_thickness_max_mm": TOP_PLATE_MAX_MM,
    "top_plate_sampling": "uniform_mm",
    "cavity_ratio_derivation": "1 - top_plate_thickness_mm / body_thickness",
    "material_sampling": "per-shape-random-spd-orthotropic-v1",
    "material_reference_fixed": False,
    # The earlier 8--10 mm solid mesh differed from a 4--6 mm reference by
    # 4.57 dB RMSE on the available sentinel, so it is not a production default.
    "mesh_size_min_mm": 4.0,
    "mesh_size_max_mm": 6.0,
    "plate_min_size_mm": 1.5,
    "air_mesh_size_mm": 10.0,
    "soundhole_mesh_size_mm": 3.0,
    "geometry_crosscheck_rtol": 0.01,
    "wall_thickness_mm": 5.0,
    "back_plate_mm": 2.0,
    "top_type": "flat",
    "bracing": "none",
    "center_block_width_mm": 40.0,
    "soundhole_bc": "impedance",
    "force_n": 1.0,
    "damping_model": "rayleigh",
    "rayleigh_alpha": 0.0,
    "rayleigh_beta": 5e-6,
    "rigid_frequency_hz": 1.0,
    "include_rigid_modes": True,
    "residual_flexibility": False,
    "structural_basis": "craig-bampton",
    "acoustic_attachment": "port",
    "static_solver": "petsc",
    "air_speed_m_s": 343.0,
    "air_density_kg_m3": 1.204,
    "material_axes_fem": {"L": "+Y", "R": "+X", "T": "+Z"},
    "time_convention": "exp(+i omega t)",
    "frequency_grid": "log_hz",
    "output_schema": "magnitude-peaks-v1",
    "top_k_peaks": 32,
    "peak_order": "select_amplitude_desc_store_frequency_asc",
    "peak_amplitude": "db_re_1_m_per_s_per_n",
    "peak_min_prominence_db": 3.0,
    "min_peak_count_per_bridge": 1,
    "hollow_peak_search_points": 4096,
    "magnitude_floor_m_per_s_per_n": 1e-30,
    "sampled_response_role": (
        "canonical NN magnitude target on a shared logarithmic grid; native complex "
        "full/reduced artifacts are retained for provenance"),
    "response_units": {"Y": "m/s/N", "p_bar": "Pa", "U_h": "m^3/s"},
}

DATASET_SCHEMA_VERSION = "mixed-magnitude-peaks-v1"

DEFAULT_FREQ_MIN = 20.0
DEFAULT_FREQ_MAX = 5000.0
DEFAULT_FREQ_POINTS = 500

N_CANONICAL = 8            # first N base contours are the canonical guitar shapes

# Random orthotropic material ranges (SI), shared with dataset_gen for consistency
MATERIAL_RANGES = {
    "E1": (8.0e9, 14.0e9), "E2": (0.8e9, 2.5e9), "E3": (0.4e9, 1.0e9),
    "G12": (1.0e9, 1.8e9), "G13": (1.0e9, 1.8e9), "G23": (0.05e9, 0.3e9),
    "nu12": (0.30, 0.50), "nu13": (0.30, 0.50), "nu23": (0.30, 0.60),
    "density": (300.0, 700.0),
}

QC_ORTHONORM_TOL = 1e-6
QC_COND_TOL = 1.01
QC_EIGEN_RESIDUAL_TOL = 1e-4  # normwise backward error of the realified GHEP basis;
# 1e-4 is still an excellent basis for a ~52k-DOF eigenproblem (near double-precision
# expectations for SLEPc at a reasonable tol).  1e-6 was physically over-tight and
# fail-closed otherwise-good cases (observed 1.09-1.13e-6 on production geometry).
QC_MODEL_RECONSTRUCTION_TOL = 1e-10
QC_BRIDGE_SNAP_TOL_MM = 1e-3
QC_COVERAGE_BOUNDARY_RTOL = 1e-6
QC_BETA0_REL_TOL = 1e-6
QC_ZERO_MODE_STIFFNESS_REL_TOL = 1e-8
QC_PORT_ATTACHMENT_RESIDUAL_TOL = 1e-8
# Consistency between the geometry-declared cavity volume / soundhole area (exact
# CAD, gmsh.occ.getMass) and what the acoustic FEM actually integrates (1^T M_a 1
# over linear tets).  Linear tets chord across curved boundaries and UNDER-estimate
# the CAD volume by a few percent, so this compares CAD-vs-FEM and must tolerate
# normal discretization (observed ~2.5%).  It still catches gross mesh/scale errors.
QC_GEOM_MESH_CONSISTENCY_RTOL = 0.05
DEFAULT_CASE_TIMEOUT_S = 6.0 * 60.0 * 60.0
PLAN_LOCK_TIMEOUT_S = 300.0
DATASET_READY_FILENAME = "dataset_ready.json"
DATASET_INVENTORY_FILENAME = "dataset_artifact_inventory.json"
PLAN_SOURCE_COMPATIBILITY_FILENAME = "plan_source_compatibility.json"
COMPLETION_POLICY = "drop-terminal-failures-v1"


# ===========================================================================
# Deterministic RNG (independent per-object seeds — NOT sequential state)
# ===========================================================================

def _sub_rng(master_seed: int, *tags) -> np.random.Generator:
    """Independent Generator seeded from (master_seed, *tags) so an object's draw
    never depends on how many other objects were drawn before it (resume-safe)."""
    h = hashlib.sha256(("|".join([str(master_seed), *map(str, tags)])).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# ===========================================================================
# Materials
# ===========================================================================

def _compliance_matrix(m: dict) -> np.ndarray:
    """6x6 orthotropic compliance S (Voigt).  Material is valid iff S is SPD."""
    E1, E2, E3 = m["E1"], m["E2"], m["E3"]
    G12, G13, G23 = m["G12"], m["G13"], m["G23"]
    nu12, nu13, nu23 = m["nu12"], m["nu13"], m["nu23"]
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0 / E1, 1.0 / E2, 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1.0 / G23, 1.0 / G13, 1.0 / G12
    return S


def material_is_spd(m: dict) -> bool:
    """True iff the orthotropic compliance (hence stiffness) is positive-definite."""
    try:
        w = np.linalg.eigvalsh(_compliance_matrix(m))
    except Exception:
        return False
    return bool(np.all(w > 0.0)) and all(
        np.isfinite(v) for v in m.values() if isinstance(v, (int, float)))


def build_material_bank(n_materials: int, seed: int,
                        base_shape_ids: list[int]) -> list[dict]:
    """Deterministic PER-SHAPE material bank (v6).

    Each base shape draws its own ``n_materials`` random SPD orthotropic materials
    (no fixed reference material) with deterministic rejection sampling, so the
    same (base_shape_id, slot) -> the same properties, always, regardless of
    resume / order.  Returns a single flat list spanning all shapes; every material
    carries a globally-unique ``material_id`` and its ``base_shape_id`` so
    plan-level lookup by ``material_id`` stays unambiguous.
    """
    bank: list[dict] = []
    for b in sorted(int(x) for x in base_shape_ids):
        for i in range(n_materials):
            mid = f"material_{b:04d}_{i:02d}"           # global-unique, carries shape
            # deterministic rejection sampling: each attempt has its own sub-seed
            for attempt in range(1000):
                rng = _sub_rng(seed, "material", b, i, attempt)
                m = {k: float(rng.uniform(lo, hi))
                     for k, (lo, hi) in MATERIAL_RANGES.items()}
                if material_is_spd(m):
                    m["material_id"] = mid
                    m["material_name"] = "random"
                    m["base_shape_id"] = int(b)
                    m["material_slot"] = int(i)
                    m["spd_attempts"] = attempt + 1
                    bank.append(m)
                    break
            else:
                raise RuntimeError(f"could not sample an SPD material for {mid}")
    return bank


# NN-loader column mapping: internal SI key -> wood engineering axis name.
# L=longitudinal(1), R=radial(2), T=tangential(3).  nn_dataset.py reads these.
MATERIAL_ENG_MAP = [
    ("E_L", "E1"), ("E_R", "E2"), ("E_T", "E3"),
    ("G_LR", "G12"), ("G_LT", "G13"), ("G_RT", "G23"),
    ("nu_LR", "nu12"), ("nu_LT", "nu13"), ("nu_RT", "nu23"),
    ("density", "density"),
]


def material_columns(material: dict) -> dict:
    """The 10 engineering-axis material columns (E_L..density) for the manifest."""
    return {col: float(material[key]) for col, key in MATERIAL_ENG_MAP}


# ===========================================================================
# Base shapes (a contour + shared geometry realised as solid AND hollow)
# ===========================================================================

def build_base_shapes(n_base: int, seed: int, n_bridge_points: int) -> list[dict]:
    """Return `n_base` base-shape dicts.  The first N_CANONICAL use the canonical
    guitar contours; the rest are deterministic random contours.  Geometry that is
    SHARED between the solid and hollow realisation (contour, thickness, bridge
    positions) is fixed here from per-shape independent seeds.  Hollow-only params
    (cavity ratio, soundhole) are also fixed but consumed only by the hollow body.
    """
    from guitar_shapes import get_guitar_contour, GUITAR_SHAPES
    from shape_gen import random_shape
    from placement_utils import random_bridge_points, random_soundhole

    shapes = []
    for b in range(n_base):
        rng = _sub_rng(seed, "shape", b)
        if b < N_CANONICAL:
            name = GUITAR_SHAPES[b % len(GUITAR_SHAPES)]
            contour = get_guitar_contour(name)
            shape_type, guitar_model = "guitar_model", name
        else:
            contour = random_shape(seed=int(rng.integers(0, 2 ** 31)))
            shape_type, guitar_model = "random", ""

        thickness = float(rng.uniform(40.0, 55.0))
        # v5: sample the top-plate thickness directly from a SEPARATE, independent
        # RNG stream so contour / body-thickness / bridge / soundhole draws above and
        # below are byte-for-byte unchanged from v4.  The cavity ratio is DERIVED.
        top_plate_thickness = float(
            _sub_rng(seed, "top_plate", b).uniform(TOP_PLATE_MIN_MM, TOP_PLATE_MAX_MM))
        cavity_ratio = 1.0 - top_plate_thickness / thickness   # derived, once

        # bridge points shared by solid & hollow — deterministic
        bridge_pts = None
        for min_edge in (25.0, 15.0, 10.0):
            try:
                bridge_pts = random_bridge_points(
                    contour, n=n_bridge_points, min_edge_dist=min_edge,
                    rng=_sub_rng(seed, "bridge", b))
                break
            except RuntimeError:
                continue
        if bridge_pts is None:
            raise RuntimeError(f"base shape {b}: could not place {n_bridge_points} bridges")

        # hollow soundhole (round) — a placement FAILURE is recorded, NOT silently
        # turned into a soundhole-less hollow.  The ONLY bridge constraint is that a
        # bridge point must not fall inside the hole: the centre is kept at least
        # (radius + BRIDGE_SOUNDHOLE_MARGIN_MM) from every bridge point (no arbitrary
        # fixed clearance).  This keeps placement feasible with many bridge points
        # (v6: 10) while guaranteeing bridge nodes stay just clear of the hole rim.
        sh_center = None
        sh_diam = 0.0
        soundhole_attempts = 0
        for hole_attempt in range(128):
            hole_rng = _sub_rng(seed, "soundhole", b, hole_attempt)
            candidate_diam = float(hole_rng.uniform(30.0, 80.0))
            eff_min_bridge = candidate_diam / 2.0 + BRIDGE_SOUNDHOLE_MARGIN_MM
            candidate_center = random_soundhole(
                contour, bridge_pts, diameter_mm=candidate_diam,
                min_bridge_dist=eff_min_bridge, min_edge_dist=20.0,
                rng=_sub_rng(seed, "soundhole-place", b, hole_attempt))
            if candidate_center is not None:
                sh_diam = candidate_diam
                sh_center = candidate_center
                soundhole_attempts = hole_attempt + 1
                break
        if sh_center is None:
            raise RuntimeError(
                f"base shape {b}: could not place a valid soundhole clear of the "
                f"{n_bridge_points} bridge points after 128 attempts")
        soundhole_ok = True

        shapes.append({
            "base_shape_id": b,
            "shape_type": shape_type,
            "guitar_model": guitar_model,
            "contour": np.asarray(contour, float),
            "thickness": thickness,
            "top_plate_thickness_mm": top_plate_thickness,      # v5 direct sample
            "cavity_ratio": cavity_ratio,                       # derived from above
            "bridge_pts_body": np.asarray(bridge_pts, float),   # (n_bridge, 2) mm
            "soundhole_ok": soundhole_ok,
            "soundhole_diameter": sh_diam if soundhole_ok else 0.0,
            "soundhole_center": ([float(sh_center[0]), float(sh_center[1])]
                                 if soundhole_ok else None),
            "soundhole_attempts": int(soundhole_attempts),
        })
    return shapes


# ===========================================================================
# Splits (by base_shape_id so solid/hollow of a base never leak across splits)
# ===========================================================================

def assign_splits(base_shape_ids, seed: int,
                  fracs=(0.70, 0.15, 0.15)) -> dict[int, str]:
    ids = sorted(int(b) for b in base_shape_ids)
    rng = _sub_rng(seed, "split")
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    out = {}
    for rank, idx in enumerate(perm):
        b = ids[int(idx)]
        out[b] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    return out


# ===========================================================================
# Plan (immutable) + hash
# ===========================================================================

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def plan_hash(plan_body: dict) -> str:
    return hashlib.sha256(_canonical_json(plan_body).encode()).hexdigest()


def validate_config(config: dict):
    required = {
        "generator_version", "n_base_shapes", "body_types", "n_materials",
        "n_bridge_points", "seed", "freq_min", "freq_max", "freq_points",
        "solid_modal_fmax", "solid_n_modes", "coupled_struct_fmax",
        "coupled_acoustic_fmax", "coupled_n_struct_modes",
        "coupled_n_acoustic_modes", "coupled_n_attach_acoustic",
        "port_end_corrections", "case_timeout_s",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"dataset config missing keys: {missing}")
    if config["generator_version"] != GENERATOR_VERSION:
        raise ValueError(f"generator_version must be {GENERATOR_VERSION}")
    for key in ("n_base_shapes", "n_materials", "n_bridge_points", "freq_points",
                "solid_n_modes", "coupled_n_struct_modes",
                "coupled_n_acoustic_modes", "coupled_n_attach_acoustic"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["freq_points"]) < 2:
        raise ValueError("freq_points must be at least 2")
    numeric = [config[k] for k in ("freq_min", "freq_max", "solid_modal_fmax",
                                    "coupled_struct_fmax", "coupled_acoustic_fmax")]
    if not np.all(np.isfinite(np.asarray(numeric, float))):
        raise ValueError("frequency limits must be finite")
    if float(config["freq_min"]) <= 0 or float(config["freq_max"]) <= float(config["freq_min"]):
        raise ValueError("frequency range must satisfy 0 < freq_min < freq_max")
    target = 1.5 * float(config["freq_max"])
    for key in ("solid_modal_fmax", "coupled_struct_fmax", "coupled_acoustic_fmax"):
        if float(config[key]) < target:
            raise ValueError(f"{key} must be at least 1.5 * freq_max ({target:g} Hz)")
    bodies = list(config["body_types"])
    if not bodies or any(b not in ("solid", "hollow") for b in bodies):
        raise ValueError("body_types must contain solid and/or hollow")
    if len(bodies) != len(set(bodies)):
        raise ValueError("body_types must not contain duplicates")
    if int(config["port_end_corrections"]) not in (0, 1, 2):
        raise ValueError("port_end_corrections must be 0, 1, or 2")
    timeout = float(config["case_timeout_s"])
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("case_timeout_s must be finite and positive")


def validate_run_options(num_shards: int, shard_index: int, max_cases: int,
                         max_mode_retries: int):
    if int(num_shards) < 1:
        raise ValueError("num_shards must be >= 1")
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if int(max_cases) < 0:
        raise ValueError("max_cases must be >= 0")
    if int(max_mode_retries) < 0:
        raise ValueError("max_mode_retries must be >= 0")


def _implementation_fingerprint() -> dict:
    rel_paths = [
        "dataset_gen_mixed.py", "_dataset_solve_worker.py", "dataset_gen.py",
        "modal_eigenbasis.py", "reduced_model_io.py", "fenics_modal_admittance.py",
        "modal_coupled_admittance.py", "fenics_admittance.py",
        "air_acoustics.py", "mesh_gen.py", "materials.py",
        "fenics_admittance_coupled.py", "fsi_coupling.py",
        "acoustic_helmholtz.py", "peak_labels.py", "backend/model_builder.py",
        "backend/holes.py", "placement_utils.py",
    ]
    out = {}
    for rel in rel_paths:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"required pipeline source is missing: {path}")
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _implementation_fingerprints_compatible(stored: dict, current: dict = None) -> bool:
    """Accept an exact source match or an explicitly audited generator migration.

    A running production plan hashes every solver source.  Operational changes to
    this orchestrator (for example publication policy) must not force completed FEM
    cases to be discarded, but silently accepting arbitrary generator changes would
    weaken the immutable-plan guard.  The external compatibility table therefore
    names the exact old -> new dataset_gen_mixed.py hash pair, while every other
    fingerprint must still match byte-for-byte.
    """
    current = current or _implementation_fingerprint()
    if stored == current:
        return True
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return False
    generator_rel = "dataset_gen_mixed.py"
    if set(stored) != set(current) or generator_rel not in stored:
        return False
    if any(stored[key] != current[key] for key in stored if key != generator_rel):
        return False
    compatibility_path = ROOT / PLAN_SOURCE_COMPATIBILITY_FILENAME
    try:
        payload = json.loads(compatibility_path.read_text(encoding="utf-8"))
        transitions = payload["transitions"]
    except Exception:
        return False
    if (payload.get("schema_version") != "mixed-plan-source-compatibility-v1"
            or not isinstance(transitions, list)):
        return False
    old_hash = stored[generator_rel]
    new_hash = current[generator_rel]
    return any(
        isinstance(item, dict)
        and item.get("file") == generator_rel
        and item.get("from_sha256") == old_hash
        and item.get("to_sha256") == new_hash
        for item in transitions
    )


def build_plan(config: dict) -> dict:
    """Build the immutable plan: config + material bank + base shapes + cases +
    splits + plan_hash.  Deterministic in (config), independent of resume state."""
    validate_config(config)
    seed = int(config["seed"])
    base_shapes = build_base_shapes(config["n_base_shapes"], seed,
                                    config["n_bridge_points"])
    # v6: per-shape material draws.  Build after base_shapes so each shape gets its
    # own bank; keep a flat list for the plan plus a per-shape index for cases.
    materials = build_material_bank(config["n_materials"], seed,
                                    [s["base_shape_id"] for s in base_shapes])
    materials_by_shape: dict[int, list[dict]] = {}
    for m in materials:
        materials_by_shape.setdefault(int(m["base_shape_id"]), []).append(m)
    splits = assign_splits([s["base_shape_id"] for s in base_shapes], seed)

    freqs = np.geomspace(config["freq_min"], config["freq_max"], config["freq_points"])

    cases = []
    for s in base_shapes:
        b = s["base_shape_id"]
        for body_type in config["body_types"]:
            shape_id = b * 2 + (0 if body_type == "solid" else 1)   # loader-compat int
            for mat in materials_by_shape[b]:
                mid = mat["material_id"]
                cases.append({
                    "case_id": f"s{b:04d}_{body_type}_{mid}",
                    "base_shape_id": b,
                    "shape_id": int(shape_id),
                    "body_type": body_type,
                    "material_id": mid,
                    "split": splits[b],
                    "n_bridges": int(s["bridge_pts_body"].shape[0]),
                    "case_seed": int.from_bytes(
                        hashlib.sha256(f"{seed}|case|{b}|{body_type}|{mid}".encode())
                        .digest()[:8], "little"),
                })
    case_ids = [c["case_id"] for c in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("plan contains duplicate case IDs")

    # Plan body (hashed) — a compact, deterministic description (NOT the bulky
    # contour arrays, but their content hash so a contour change is detected).
    serial_shapes = []
    for s in base_shapes:
        serial_shapes.append({
            "base_shape_id": int(s["base_shape_id"]),
            "shape_type": s["shape_type"],
            "guitar_model": s["guitar_model"],
            "contour": np.asarray(s["contour"], float).tolist(),
            "thickness": float(s["thickness"]),
            "top_plate_thickness_mm": float(s["top_plate_thickness_mm"]),
            "cavity_ratio": float(s["cavity_ratio"]),
            "bridge_pts_body": np.asarray(s["bridge_pts_body"], float).tolist(),
            "soundhole_ok": bool(s["soundhole_ok"]),
            "soundhole_diameter": float(s["soundhole_diameter"]),
            "soundhole_center": s["soundhole_center"],
            "soundhole_attempts": int(s.get("soundhole_attempts", 1)),
        })
    shape_digests = {
        s["base_shape_id"]: hashlib.sha256(
            _canonical_json({
                "contour": np.round(s["contour"], 6),
                "thickness": round(s["thickness"], 6),
                "top_plate_thickness_mm": round(s["top_plate_thickness_mm"], 6),
                "cavity_ratio": round(s["cavity_ratio"], 6),
                "bridge": np.round(s["bridge_pts_body"], 6),
                "soundhole_ok": s["soundhole_ok"],
                "soundhole_diameter": round(s["soundhole_diameter"], 6),
                "soundhole_center": s["soundhole_center"],
            }).encode()).hexdigest()
        for s in base_shapes
    }
    plan_body = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "config": config,
        "solid_solver_revision": SOLID_SOLVER_REVISION,
        "hollow_solver_revision": HOLLOW_SOLVER_REVISION,
        "production_contract": PRODUCTION_CONTRACT,
        "implementation_fingerprint": _implementation_fingerprint(),
        "frequencies": np.round(freqs, 9).tolist(),
        "materials": materials,
        "base_shapes": serial_shapes,
        "splits": {str(k): v for k, v in splits.items()},
        "shape_digests": {str(k): v for k, v in shape_digests.items()},
        "cases": [{k: c[k] for k in ("case_id", "base_shape_id", "shape_id",
                                     "body_type", "material_id", "split",
                                     "n_bridges", "case_seed")} for c in cases],
    }
    ph = plan_hash(plan_body)
    return {
        "plan_hash": ph,
        "plan_body": plan_body,
        # non-hashed runtime objects (arrays) kept alongside for execution
        "_materials": materials,
        "_base_shapes": base_shapes,
        "_frequencies": freqs,
        "_cases": cases,
    }


def save_plan(plan: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    body = {"plan_hash": plan["plan_hash"], **plan["plan_body"]}
    _atomic_write_bytes(output_dir / "dataset_plan.json",
                        json.dumps(body, indent=2, default=_json_default).encode())


def _stored_plan_hash(existing: dict) -> str:
    """Recompute the hash of a stored plan's BODY (never trust its plan_hash field)."""
    body = {k: v for k, v in existing.items() if k != "plan_hash"}
    return plan_hash(body)


def check_or_write_plan(plan: dict, output_dir: Path):
    """Immutable-plan guard.  If a plan already exists, its stored body is
    RE-HASHED (the plan_hash field is not trusted) and must equal the new plan's
    hash; otherwise ABORT.  First writer persists the plan atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "dataset_plan.json"
    lock_path = output_dir / ".dataset_plan.lockfile"

    def validate_existing():
        existing = json.loads(p.read_text())
        recomputed = _stored_plan_hash(existing)
        if recomputed != existing.get("plan_hash"):
            raise RuntimeError(
                f"stored dataset_plan.json is corrupt: body re-hash {recomputed} "
                f"!= recorded plan_hash {existing.get('plan_hash')} in {output_dir}.")
        if recomputed != plan["plan_hash"]:
            raise RuntimeError(
                f"plan_hash mismatch for existing output root {output_dir}: "
                f"existing {recomputed} != new {plan['plan_hash']}. "
                "Config/plan changed; refusing to mix incompatible runs.")

    lock = _AdvisoryFileLock(lock_path)
    if not lock.acquire():
        raise RuntimeError(f"timed out waiting for plan lock {lock_path}")
    try:
        if p.exists():
            validate_existing()
        else:
            save_plan(plan, output_dir)
    finally:
        lock.release()


def _runtime_plan_from_stored(stored: dict) -> dict:
    """Validate and deserialize dataset_plan.json into runtime arrays."""
    recomputed = _stored_plan_hash(stored)
    if recomputed != stored.get("plan_hash"):
        raise RuntimeError("stored dataset_plan.json is corrupt (body hash mismatch)")
    required = {"config", "materials", "base_shapes", "frequencies", "cases",
                "production_contract", "implementation_fingerprint"}
    missing = sorted(required - set(stored))
    if missing:
        raise RuntimeError(f"stored dataset plan schema is incomplete: {missing}")
    shapes = []
    for raw in stored["base_shapes"]:
        s = dict(raw)
        s["base_shape_id"] = int(s["base_shape_id"])
        s["contour"] = np.asarray(s["contour"], float)
        s["bridge_pts_body"] = np.asarray(s["bridge_pts_body"], float)
        shapes.append(s)
    return {
        "plan_hash": stored["plan_hash"],
        "plan_body": {k: v for k, v in stored.items() if k != "plan_hash"},
        "_materials": [dict(m) for m in stored["materials"]],
        "_base_shapes": shapes,
        "_frequencies": np.asarray(stored["frequencies"], float),
        "_cases": [dict(c) for c in stored["cases"]],
    }


def load_or_create_plan(config: dict, output_dir: Path) -> dict:
    """Load the persisted concrete plan, or atomically create it once."""
    output_dir = Path(output_dir).resolve()
    p = output_dir / "dataset_plan.json"
    if not p.exists():
        candidate = build_plan(config)
        check_or_write_plan(candidate, output_dir)
    stored = json.loads(p.read_text())
    runtime = _runtime_plan_from_stored(stored)
    if _canonical_json(runtime["plan_body"]["config"]) != _canonical_json(config):
        raise RuntimeError(
            f"plan_hash mismatch for existing output root {output_dir}: config differs")
    if runtime["plan_body"].get("generator_version") != GENERATOR_VERSION:
        raise RuntimeError("stored plan generator version is incompatible")
    if runtime["plan_body"].get("production_contract") != PRODUCTION_CONTRACT:
        raise RuntimeError("stored plan production physics/geometry contract is incompatible")
    if not _implementation_fingerprints_compatible(
            runtime["plan_body"].get("implementation_fingerprint")):
        raise RuntimeError("stored plan was created with different pipeline source code")
    return runtime


def load_existing_plan(output_dir: Path, config: dict | None = None) -> dict:
    p = Path(output_dir) / "dataset_plan.json"
    if not p.is_file():
        raise RuntimeError(f"cannot merge without an existing plan: {p}")
    runtime = _runtime_plan_from_stored(json.loads(p.read_text()))
    if config is not None and (_canonical_json(runtime["plan_body"]["config"])
                               != _canonical_json(config)):
        raise RuntimeError("plan_hash mismatch: merge config differs from stored plan")
    if (runtime["plan_body"].get("generator_version") != GENERATOR_VERSION
            or runtime["plan_body"].get("production_contract") != PRODUCTION_CONTRACT):
        raise RuntimeError("stored plan uses an incompatible generator/physics contract")
    if not _implementation_fingerprints_compatible(
            runtime["plan_body"].get("implementation_fingerprint")):
        raise RuntimeError("stored plan was created with different pipeline source code")
    return runtime


# ===========================================================================
# Sharding (by SHAPE so a mesh is never contended)
# ===========================================================================

def shard_cases(cases: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards <= 1:
        return list(cases)
    return [c for c in cases if (c["base_shape_id"] % num_shards) == shard_index]


# ===========================================================================
# Run policy (RUNTIME ONLY — scheduling, never physics)
# ===========================================================================
#
# Solve cost scales roughly as (mesh size)^1.4, and the random shape sampler spans
# a 15x body-volume range, so a handful of very large shapes dominate the wall
# clock.  A run policy lets an operator (a) run the cheap shapes first so that
# stopping at any moment maximises shape coverage, and (b) spend fewer material
# draws on the most expensive shapes so those shapes still ENTER the dataset
# instead of being dropped (which would truncate the body-size distribution).
#
# This is deliberately NOT part of the plan: it selects and orders which planned
# cases THIS INVOCATION attempts and nothing else.  plan_hash, geometry, solver
# settings, QC and every stored artifact are untouched, and cases it defers stay
# `pending` so a later run — with a different policy or none — completes them.
RUN_POLICY_SCHEMA = "mixed-run-policy-v1"


def load_run_policy(path) -> dict:
    """Read + validate a run policy.  Fail-loud: a typo must not silently skip work."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != RUN_POLICY_SCHEMA:
        raise ValueError(f"run policy schema must be {RUN_POLICY_SCHEMA}")
    order = payload.get("shape_order") or []
    skip = payload.get("skip_shapes") or []
    mps = payload.get("materials_per_shape") or {}
    if not isinstance(order, list) or not isinstance(skip, list):
        raise ValueError("shape_order and skip_shapes must be lists")
    if not isinstance(mps, dict):
        raise ValueError("materials_per_shape must be an object")
    order = [int(b) for b in order]
    if len(set(order)) != len(order):
        raise ValueError("shape_order contains duplicate base_shape_ids")
    mps = {int(k): int(v) for k, v in mps.items()}
    if any(v < 0 for v in mps.values()):
        raise ValueError("materials_per_shape values must be >= 0")
    return {"schema_version": RUN_POLICY_SCHEMA, "shape_order": order,
            "skip_shapes": [int(b) for b in skip], "materials_per_shape": mps,
            "note": payload.get("note", "")}


def apply_run_policy(cases: list[dict], policy: dict | None,
                     mat_by_id: dict) -> list[dict]:
    """Filter + reorder the cases this invocation will attempt.

    * `skip_shapes` / `materials_per_shape` drop cases (they stay pending).
    * `materials_per_shape[b] = k` keeps that shape's k LOWEST material slots, so
      the retained subset is deterministic and stable across resumes.
    * `shape_order` lists base_shape_ids in execution order; any shape absent from
      the list runs after the listed ones, in plan order.
    Ordering is stable, so within a shape the plan's case order is preserved.
    """
    if not policy:
        return list(cases)
    skip = set(policy["skip_shapes"])
    mps = policy["materials_per_shape"]
    kept = []
    for c in cases:
        b = int(c["base_shape_id"])
        if b in skip:
            continue
        limit = mps.get(b)
        if limit is not None:
            slot = mat_by_id.get(c["material_id"], {}).get("material_slot")
            if slot is None:
                raise RuntimeError(
                    f"cannot apply materials_per_shape: {c['material_id']} has no "
                    "material_slot in the plan")
            if int(slot) >= limit:
                continue
        kept.append(c)
    rank = {b: i for i, b in enumerate(policy["shape_order"])}
    tail = len(rank)
    return sorted(kept, key=lambda c: rank.get(int(c["base_shape_id"]), tail))


# ===========================================================================
# QC (fail-closed)
# ===========================================================================

def _finite_scalar(d: dict, key: str, default):
    """Return (value, ok_finite).  A NaN/Inf/missing/non-numeric diagnostic reads as
    NON-finite so it can NEVER slip past a `<=`/`>` comparison (fail-closed)."""
    v = d.get(key, default)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return default, False
    return fv, bool(np.isfinite(fv))


def _check_le(reasons, d, key, tol, default):
    fv, fin = _finite_scalar(d, key, default)
    if not fin:
        reasons.append(f"{key}={d.get(key)} non-finite")
    elif fv < 0.0:
        reasons.append(f"{key}={fv:g} is negative")
    elif fv > tol:
        reasons.append(f"{key}={fv:g} > {tol:g}")


def _finite_int(d: dict, key: str) -> tuple[int, bool]:
    """Return an exact JSON integer; reject booleans and lossy float coercion."""
    value = d.get(key)
    if isinstance(value, (bool, np.bool_)):
        return 0, False
    try:
        fv = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0, False
    if not np.isfinite(fv) or not fv.is_integer():
        return 0, False
    return int(fv), True


def _numeric_tree_is_finite(value, *, nonnegative: bool = False) -> bool:
    """Return False if any numeric leaf in a JSON-like tree is NaN/Inf.

    Solver metadata also contains strings, booleans and optional ``None`` values;
    those are not numerical diagnostics and are intentionally ignored.  Arrays are
    handled as a unit so this helper is also safe on a live (pre-JSON) response.
    """
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, dict):
        return all(_numeric_tree_is_finite(v, nonnegative=nonnegative)
                   for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_tree_is_finite(v, nonnegative=nonnegative)
                   for v in value)
    if isinstance(value, (int, float, complex, np.number)):
        try:
            finite = bool(np.isfinite(value))
            if nonnegative and not isinstance(value, complex):
                finite = finite and float(value) >= 0.0
            return finite
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(value, np.ndarray):
        try:
            if not bool(np.all(np.isfinite(value))):
                return False
            return not nonnegative or bool(np.all(value >= 0))
        except (TypeError, ValueError):
            return False
    return True


def _check_response_array(response, name, n_bridges, freqs, reasons):
    try:
        A = np.asarray(response.get(name, np.zeros((0, 0))))
    except (TypeError, ValueError):
        reasons.append(f"{name} is not a numeric array")
        return
    if A.shape != (n_bridges, freqs.size):
        reasons.append(f"{name} shape {A.shape} != {(n_bridges, freqs.size)}")
        return
    try:
        finite = bool(np.all(np.isfinite(A)))
    except TypeError:
        finite = False
    if not finite:
        reasons.append(f"non-finite {name}")
        return
    row_scale = np.max(np.abs(A), axis=1)
    empty_rows = np.flatnonzero(row_scale <= 1e-20)
    if empty_rows.size:
        reasons.append(
            f"{name} is identically zero or numerically empty for bridge rows "
            f"{empty_rows.tolist()}")


def _compact_reconstruction_error(model_path: Path, body: str,
                                  freqs: np.ndarray, Y: np.ndarray,
                                  p_bar: np.ndarray | None = None,
                                  U_h: np.ndarray | None = None) -> float:
    """Independently reconstruct a compact model and compare its canonical arrays.

    Solid reconstruction is vectorized, so the entire stored grid is audited.
    Hollow reconstruction performs dense reduced solves and therefore uses seven
    deterministic points spanning the band.  The solver itself separately audits
    its compact output; this second check catches packaging or file mix-ups.
    """
    from reduced_model_io import resample_air_coupled, resample_structural_modal

    freqs = np.asarray(freqs, float)
    Y = np.asarray(Y, complex)
    if body == "solid":
        reconstructed = resample_structural_modal(model_path, freqs)
        pairs = ((np.asarray(reconstructed["Y"], complex), Y),)
    else:
        if p_bar is None or U_h is None:
            raise ValueError("hollow compact reconstruction requires p_bar and U_h")
        audit_idx = np.unique(np.linspace(
            0, freqs.size - 1, min(7, freqs.size), dtype=int))
        reconstructed = resample_air_coupled(model_path, freqs[audit_idx])
        pairs = (
            (np.asarray(reconstructed["Y"], complex), Y[:, audit_idx]),
            (np.asarray(reconstructed["p_bar"], complex),
             np.asarray(p_bar, complex)[:, audit_idx]),
            (np.asarray(reconstructed["U_h"], complex),
             np.asarray(U_h, complex)[:, audit_idx]),
        )
    error = 0.0
    for actual, expected in pairs:
        if (actual.shape != expected.shape or not np.all(np.isfinite(actual))
                or not np.all(np.isfinite(expected))):
            return float("inf")
        scale = max(float(np.max(np.abs(expected))), 1e-30)
        error = max(error, float(np.max(np.abs(actual - expected))) / scale)
    return error


def _check_compact_reconstruction(reasons: list[str], response: dict,
                                  body: str, freqs: np.ndarray):
    artifact_dir = response.get("artifact_dir")
    model_name = "reduced_model.npz" if body == "hollow" else "solid_full_eigen.npz"
    if not isinstance(artifact_dir, (str, os.PathLike)):
        reasons.append("solver response is missing its compact-model artifact_dir")
        return
    model_path = Path(artifact_dir) / model_name
    if not model_path.is_file():
        reasons.append(f"solver did not produce {model_name}")
        return
    try:
        error = _compact_reconstruction_error(
            model_path, body, freqs, response.get("admittance"),
            response.get("p_bar"), response.get("U_h"))
    except Exception as exc:
        reasons.append(f"compact-model reconstruction failed: {exc}")
        return
    if not np.isfinite(error) or error > QC_MODEL_RECONSTRUCTION_TOL:
        reasons.append(
            f"compact-model reconstruction relative error={error:g} > "
            f"{QC_MODEL_RECONSTRUCTION_TOL:g}")


def qc_case(case: dict, response: dict, freqs: np.ndarray,
            expected_geom: dict | None = None,
            expected_budget: dict | None = None,
            expected_input_hash: str | None = None) -> tuple[bool, list[str]]:
    """Validate a solver `response` for one case (fail-closed).  Returns (ok,reasons).

    All scalar diagnostics are finite-checked BEFORE any threshold comparison, so a
    NaN/Inf can never bypass a `>`/`<=` test and be accepted.
    """
    reasons: list[str] = []
    if not isinstance(response, dict):
        return False, ["solver response is not an object"]
    body = case["body_type"]
    n_bridges = case["n_bridges"]

    # --- common ---
    if response.get("run_status") != "complete":
        reasons.append(f"run_status={response.get('run_status')}")
    if (expected_input_hash is not None
            and response.get("case_input_hash") != expected_input_hash):
        reasons.append("case_input_hash mismatch")
    try:
        f = np.asarray(response.get("frequencies", []), float)
        grid_ok = (f.shape == freqs.shape and np.all(np.isfinite(f))
                   and np.allclose(f, freqs, rtol=0, atol=1e-9))
    except (TypeError, ValueError):
        grid_ok = False
    if not grid_ok:
        reasons.append("frequency grid mismatch")
    _check_response_array(response, "admittance", n_bridges, freqs, reasons)

    # bridge metadata: count + finite snap distances
    bridges = response.get("bridges", [])
    expected_coords = (expected_geom or {}).get("bridge_coords")
    if not isinstance(bridges, (list, tuple)):
        reasons.append("bridges metadata is not a list")
    elif len(bridges) != n_bridges:
        reasons.append(f"bridges count {len(bridges)} != {n_bridges}")
    else:
        for bi, br in enumerate(bridges):
            if not isinstance(br, dict):
                reasons.append(f"bridge {bi} metadata is not an object")
                continue
            sd, fin = _finite_scalar(br, "snap_distance_mm", float("inf"))
            if not fin:
                reasons.append(f"bridge {bi} snap_distance non-finite")
                continue
            try:
                req = np.asarray(br.get("bridge_requested_xyz", []), float)
                snp = np.asarray(br.get("bridge_snapped_xyz", []), float)
            except (TypeError, ValueError):
                reasons.append(f"bridge {bi} requested/snapped XYZ invalid")
                continue
            if (req.shape != (3,) or snp.shape != (3,)
                    or not np.all(np.isfinite(req)) or not np.all(np.isfinite(snp))):
                reasons.append(f"bridge {bi} requested/snapped XYZ invalid")
                continue
            if expected_coords is not None:
                try:
                    exp = np.asarray(expected_coords[bi], float)
                except (IndexError, TypeError, ValueError):
                    exp = np.zeros(0)
                if (exp.shape != (3,) or not np.all(np.isfinite(exp))
                        or not np.allclose(req, exp, rtol=0, atol=1e-6)):
                    reasons.append(f"bridge {bi} requested XYZ does not match geometry")
            measured = float(np.linalg.norm(snp - req))
            if abs(measured - sd) > max(1e-6, 1e-6 * max(measured, sd, 1.0)):
                reasons.append(f"bridge {bi} snap distance inconsistent with XYZ")
            if sd > QC_BRIDGE_SNAP_TOL_MM:
                reasons.append(
                    f"bridge {bi} snap_distance_mm={sd:g} > {QC_BRIDGE_SNAP_TOL_MM:g}")

    d = response.get("diagnostics", {})
    if not isinstance(d, dict):
        reasons.append("diagnostics is not an object")
        d = {}
    if body == "hollow":
        model_error, model_error_finite = _finite_scalar(
            d, "model_reconstruction_max_rel_error", float("inf"))
        if (not model_error_finite or model_error < 0.0
                or model_error > QC_MODEL_RECONSTRUCTION_TOL):
            reasons.append(
                "model_reconstruction_max_rel_error="
                f"{d.get('model_reconstruction_max_rel_error')}")
    if not _numeric_tree_is_finite(d):
        reasons.append("diagnostics contains non-finite numeric value")
    timing = response.get("timing", {})
    if not isinstance(timing, dict):
        reasons.append("timing is not an object")
    elif not _numeric_tree_is_finite(timing, nonnegative=True):
        reasons.append("timing contains non-finite or negative numeric value")
    rev = response.get("solver_revision")
    if expected_input_hash is not None and body == "hollow":
        _check_compact_reconstruction(reasons, response, body, freqs)

    try:
        from peak_labels import magnitude_db, validate_peak_batch
        log_mag = magnitude_db(np.asarray(response.get("admittance")))
        if log_mag.shape != (n_bridges, freqs.size):
            reasons.append(f"log magnitude shape {log_mag.shape} is invalid")
        peak_reasons = validate_peak_batch(
            response.get("peaks", {}), n_bridges,
            int(PRODUCTION_CONTRACT["top_k_peaks"]),
            float(freqs[0]), float(freqs[-1]),
            min_count=int(PRODUCTION_CONTRACT["min_peak_count_per_bridge"]))
        reasons.extend(peak_reasons)
    except Exception as exc:
        reasons.append(f"magnitude/peak schema validation failed: {exc}")

    if body == "hollow":
        if rev != HOLLOW_SOLVER_REVISION:
            reasons.append(f"solver_revision={rev} (want {HOLLOW_SOLVER_REVISION})")
        # pressure + soundhole volume-velocity must be full, finite (n_bridge,F)
        _check_response_array(response, "p_bar", n_bridges, freqs, reasons)
        _check_response_array(response, "U_h", n_bridges, freqs, reasons)
        if d.get("band_covered") is not True:
            reasons.append("band_covered=false")
        if d.get("conformal_interface_verified") is not True:
            reasons.append("conformal_interface_verified=false")
        if str(d.get("basis", "")) != "craig-bampton":
            reasons.append(f"basis={d.get('basis')} (want craig-bampton)")
        if d.get("acoustic_port_attachment") is not True:
            reasons.append("acoustic_port_attachment=false")
        for key in ("structural_mass_orthonormality_max_dev",
                    "acoustic_mass_orthonormality_max_dev"):
            _check_le(reasons, d, key, QC_ORTHONORM_TOL, 1.0)
        for key in ("structural_reduced_mass_condition",
                    "acoustic_reduced_mass_condition"):
            _check_le(reasons, d, key, QC_COND_TOL, 1e9)
            value, finite = _finite_scalar(d, key, 0.0)
            if finite and value < 1.0 - 1e-8:
                reasons.append(f"{key}={value:g} < 1")
        for key in ("structural_eigen_residual", "acoustic_eigen_residual",
                    "max_eigen_residual"):
            _check_le(reasons, d, key, QC_EIGEN_RESIDUAL_TOL, 1.0)
        _check_le(reasons, d, "model_reconstruction_max_rel_error",
                  QC_MODEL_RECONSTRUCTION_TOL, 1.0)
        cav, fin = _finite_scalar(d, "cavity_volume_m3", 0.0)
        if not fin or cav <= 0.0:
            reasons.append(f"cavity_volume={d.get('cavity_volume_m3')}")
        sh, fin = _finite_scalar(d, "soundhole_area_m2", 0.0)
        if not fin or sh <= 0.0:
            reasons.append(f"soundhole_area={d.get('soundhole_area_m2')}")
        want_struct = float((expected_budget or {}).get("struct_fmax", 7500.0))
        want_acoustic = float((expected_budget or {}).get("acoustic_fmax", 7500.0))
        for key, want in (("struct_fmax_hz", want_struct),
                          ("acoustic_fmax_hz", want_acoustic)):
            fv, fin = _finite_scalar(d, key, 0.0)
            if not fin or abs(fv - want) > 1.0:
                reasons.append(f"{key}={d.get(key)} (want {want})")
        na, fin = _finite_int(d, "n_attach_acoustic_used")
        want_na = int((expected_budget or {}).get("n_attach_acoustic", 20))
        if not fin or na != want_na:
            reasons.append(
                f"n_attach_acoustic_used={d.get('n_attach_acoustic_used')} (want {want_na})")
        nar, fin = _finite_int(d, "n_attach_acoustic_requested")
        if not fin or nar != want_na:
            reasons.append(
                f"n_attach_acoustic_requested={d.get('n_attach_acoustic_requested')} "
                f"(want {want_na})")
        pc, fin = _finite_int(d, "port_end_corrections")
        want_pc = int((expected_budget or {}).get("port_end_corrections", 1))
        if not fin or pc != want_pc:
            reasons.append(
                f"port_end_corrections={d.get('port_end_corrections')} (want {want_pc})")
        if d.get("static_solver") != PRODUCTION_CONTRACT["static_solver"]:
            reasons.append(f"static_solver={d.get('static_solver')} (want petsc)")
        for key, want in (("analysis_freq_min_hz", float(freqs[0])),
                          ("analysis_freq_max_hz", float(freqs[-1]))):
            fv, fin = _finite_scalar(d, key, float("nan"))
            if not fin or abs(fv - want) > 1e-9:
                reasons.append(f"{key}={d.get(key)} (want {want})")
        fp, fin = _finite_int(d, "analysis_freq_points")
        if not fin or fp != int(freqs.size):
            reasons.append(f"analysis_freq_points={d.get('analysis_freq_points')}")

        # Recompute coverage from the scalar eigenfrequency evidence instead of
        # trusting the solver's booleans.  This catches stale or sanitized metadata.
        boundary, boundary_ok = _finite_scalar(
            d, "basis_coverage_boundary_rtol", float("nan"))
        margin, margin_ok = _finite_scalar(d, "basis_band_margin", float("nan"))
        target, target_ok = _finite_scalar(d, "basis_band_target_hz", float("nan"))
        eig_s, eig_s_ok = _finite_scalar(d, "eig_freq_max_struct_hz", float("nan"))
        eig_a, eig_a_ok = _finite_scalar(d, "eig_freq_max_acoustic_hz", float("nan"))
        limit_s, limit_s_ok = _finite_scalar(
            d, "basis_frequency_limit_struct_hz", float("nan"))
        limit_a, limit_a_ok = _finite_scalar(
            d, "basis_frequency_limit_acoustic_hz", float("nan"))
        coverage_scalars_ok = all((boundary_ok, margin_ok, target_ok, eig_s_ok,
                                   eig_a_ok, limit_s_ok, limit_a_ok))
        expected_target = 1.5 * float(freqs[-1])
        if (not coverage_scalars_ok or boundary < 0.0
                or not np.isclose(boundary, QC_COVERAGE_BOUNDARY_RTOL,
                                  rtol=0.0, atol=1e-15)
                or not np.isclose(margin, 1.5, rtol=0.0, atol=1e-12)
                or not np.isclose(target, expected_target, rtol=0.0, atol=1e-8)):
            reasons.append("hollow basis coverage scalar metadata is invalid")
        else:
            beyond_target = (
                eig_s > expected_target * (1.0 + boundary)
                and eig_a > expected_target * (1.0 + boundary))
            recomputed_band = bool(
                want_struct >= expected_target and want_acoustic >= expected_target
                and beyond_target)
            recomputed_cutoff = bool(
                eig_s > want_struct * (1.0 + boundary)
                and eig_a > want_acoustic * (1.0 + boundary))
            expected_limit_s = min(want_struct, eig_s)
            expected_limit_a = min(want_acoustic, eig_a)
            if (d.get("band_covered") is not recomputed_band
                    or d.get("coverage_ok") is not recomputed_cutoff
                    or d.get("cutoff_reached") is not recomputed_cutoff
                    or not np.isclose(limit_s, expected_limit_s, rtol=1e-12)
                    or not np.isclose(limit_a, expected_limit_a, rtol=1e-12)):
                reasons.append("hollow basis coverage flags disagree with eigenfrequencies")

        _check_le(reasons, d, "beta0_vs_S_over_sqrtV_rel", QC_BETA0_REL_TOL, 1.0)
        _check_le(reasons, d, "zero_mode_stiffness_rel",
                  QC_ZERO_MODE_STIFFNESS_REL_TOL, 1.0)
        port_diag = d.get("acoustic_port_attachment_diag")
        if not isinstance(port_diag, dict):
            reasons.append("acoustic_port_attachment_diag is not an object")
        else:
            _check_le(reasons, port_diag, "residual_rel",
                      QC_PORT_ATTACHMENT_RESIDUAL_TOL, 1.0)
            _check_le(reasons, port_diag, "m_orthogonality_to_psi0",
                      QC_PORT_ATTACHMENT_RESIDUAL_TOL, 1.0)

        a0 = response.get("A0")
        if not isinstance(a0, dict):
            reasons.append("A0 is not an object")
        else:
            detected = a0.get("A0_detected")
            estimated, estimated_ok = _finite_scalar(
                a0, "A0_estimated_hz", float("nan"))
            if detected not in (True, False) or not isinstance(
                    detected, (bool, np.bool_)):
                reasons.append("A0_detected is not a boolean")
            if not estimated_ok or estimated <= 0.0:
                reasons.append(f"A0_estimated_hz={a0.get('A0_estimated_hz')}")
            observed = a0.get("A0_observed_hz")
            if detected is True:
                try:
                    observed_f = float(observed)
                    observed_ok = (np.isfinite(observed_f) and observed_f > 0.0
                                   and float(freqs[0]) <= observed_f <= float(freqs[-1]))
                except (TypeError, ValueError, OverflowError):
                    observed_ok = False
                if not observed_ok:
                    reasons.append(f"A0_observed_hz={observed}")
            elif detected is False and observed is not None:
                reasons.append("undetected A0 must have A0_observed_hz=null")
        if expected_geom:
            for key in ("cavity_volume_m3", "soundhole_area_m2"):
                if key in expected_geom:
                    got, got_fin = _finite_scalar(d, key, float("nan"))
                    want = float(expected_geom[key])
                    if (not got_fin or not np.isclose(
                            got, want, rtol=QC_GEOM_MESH_CONSISTENCY_RTOL,
                            atol=max(abs(want) * 1e-6, 1e-12))):
                        reasons.append(f"{key}={d.get(key)} differs from mesh {want}")
    else:  # solid
        if rev != SOLID_SOLVER_REVISION:
            reasons.append(f"solver_revision={rev} (want {SOLID_SOLVER_REVISION})")
        if d.get("coverage_ok") is not True:
            reasons.append("modal coverage_ok=false")
        _check_le(reasons, d, "structural_mass_orthonormality_max_dev",
                  QC_ORTHONORM_TOL, 1.0)
        _check_le(reasons, d, "max_eigen_residual", QC_EIGEN_RESIDUAL_TOL, 1.0)
        want_fmax = float((expected_budget or {}).get("modal_fmax", 7500.0))
        mf, fin = _finite_scalar(d, "modal_fmax_hz", 0.0)
        if not fin or abs(mf - want_fmax) > 1.0:
            reasons.append(f"modal_fmax_hz={d.get('modal_fmax_hz')} (want {want_fmax})")
        eig_max, eig_ok = _finite_scalar(d, "eig_freq_max_hz", float("nan"))
        boundary, boundary_ok = _finite_scalar(
            d, "coverage_boundary_rtol", float("nan"))
        n_retained, n_retained_ok = _finite_int(d, "n_modes_retained")
        n_converged, n_converged_ok = _finite_int(d, "n_eig_converged")
        requested_modes = int((expected_budget or {}).get("n_modes", 0))
        recomputed_coverage = bool(
            eig_ok and boundary_ok and boundary >= 0.0
            and eig_max > want_fmax * (1.0 + boundary))
        if (not eig_ok or not boundary_ok
                or not np.isclose(boundary, QC_COVERAGE_BOUNDARY_RTOL,
                                  rtol=0.0, atol=1e-15)
                or d.get("coverage_ok") is not recomputed_coverage):
            reasons.append("solid modal coverage flags disagree with eigenfrequencies")
        # EPS may report a few more converged pairs than the requested ``nev``
        # when a degenerate cluster converges together.  That is valid evidence,
        # not corruption; only the lower/count consistency bounds are physical.
        if (not n_retained_ok or not n_converged_ok or n_retained <= 0
                or n_converged < n_retained or requested_modes <= 0):
            reasons.append("solid modal mode counts are invalid")
        for key, want in (("analysis_freq_min_hz", float(freqs[0])),
                          ("analysis_freq_max_hz", float(freqs[-1]))):
            fv, finite = _finite_scalar(d, key, float("nan"))
            if not finite or abs(fv - want) > 1e-9:
                reasons.append(f"{key}={d.get(key)} (want {want})")
        fp, finite = _finite_int(d, "analysis_freq_points")
        if not finite or fp != int(freqs.size):
            reasons.append(f"analysis_freq_points={d.get('analysis_freq_points')}")

    return (len(reasons) == 0), [r for r in reasons if r]


# ===========================================================================
# Persistent artifacts: atomic write + commit-marker directory commit
# ===========================================================================

_COMMIT_MARKER = ".committed"


class _AdvisoryFileLock:
    """Small stdlib-only cross-platform exclusive file lock.

    The lock file is persistent, but the OS lock is released automatically if the
    process dies.  That property is essential for safely distinguishing an active
    transaction journal from one left by a crashed worker.
    """
    def __init__(self, path: Path, *, shared: bool = False):
        self.path = Path(path)
        self.shared = bool(shared)
        self.file = None

    def acquire(self, timeout_s: float = PLAN_LOCK_TIMEOUT_S) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"\0")
            self.file.flush()
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    mode = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
                    fcntl.flock(self.file.fileno(), mode | fcntl.LOCK_NB)
                return True
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    self.file.close()
                    self.file = None
                    return False
                time.sleep(0.05)

    def release(self):
        if self.file is None:
            return
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()
            self.file = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def _atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # mkstemp forces 0600 (owner-only).  These are dataset artifacts meant to be
        # copied out; root-created 0600 files fail to read as a non-root user (the
        # "json turned into LOCK on transfer" symptom).  Force world-readable 0644
        # (independent of the container umask, which may itself be restrictive).
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_marker(d: Path):
    """Write the commit marker LAST — its presence certifies a complete dir."""
    (d / _COMMIT_MARKER).write_bytes(b"")


def _is_committed(d: Path) -> bool:
    return d.is_dir() and (d / _COMMIT_MARKER).exists()


def _commit_dir(staging: Path, final: Path):
    """Commit a fully-written staging dir (marker already inside) to `final`.

    Rename-into-place: an existing `final` is first renamed aside, then the staging
    dir is renamed onto the (now free) name, then the old copy is deleted.  A crash
    at any point leaves either the old committed dir or NO final (never a partial),
    because the marker is only ever present in a fully-written dir.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    trash = None
    if final.exists():
        trash = final.with_name(f"{final.name}.trash_{os.getpid()}_{int(time.time()*1e6)}")
        os.rename(final, trash)
    try:
        os.rename(staging, final)
    except Exception:
        # Restore the previous committed directory if promotion fails.  This keeps
        # a transient filesystem error from turning a safe replacement into data
        # loss.
        if trash is not None and trash.exists() and not final.exists():
            os.rename(trash, final)
        raise
    if trash is not None and trash.exists():
        shutil.rmtree(trash, ignore_errors=True)


def _commit_dirs_transaction(pairs: list[tuple[Path, Path]], *,
                             certification_path: Path | None = None,
                             transaction_id: str | None = None,
                             transaction_root: Path | None = None):
    """Promote several staged directories with rollback and a crash journal.

    This is used for a case plus all of its bridge samples.  Every staging
    directory is complete before the first old result is moved.  The journal is
    left behind by a hard kill and recovered at the beginning of the next run.
    """
    if not pairs:
        return
    token = uuid.uuid4().hex
    finals = [Path(final).resolve() for _staging, final in pairs]
    root = (Path(transaction_root).resolve() if transaction_root is not None else
            Path(os.path.commonpath([str(p.parent) for p in finals])).resolve())
    journal_root = root / ".transactions"
    journal_root.mkdir(parents=True, exist_ok=True)
    journal_path = journal_root / f"transaction_{token}.json"
    if (certification_path is None) != (transaction_id is None):
        raise ValueError("certification_path and transaction_id must be provided together")
    lock_rel = str((Path(".transaction_locks") /
                    f"{finals[-1].name}.lock").as_posix())
    lock = _AdvisoryFileLock(root / lock_rel)
    if not lock.acquire():
        raise RuntimeError(f"timed out waiting for case transaction lock {root / lock_rel}")
    try:
        # A previous owner of this same case lock may have died after leaving a
        # journal.  Recover it while holding the stable lock, before observing the
        # old finals for this replacement.
        for old_journal in sorted(journal_root.glob("transaction_*.json")):
            try:
                old_payload = json.loads(old_journal.read_text(encoding="utf-8"))
            except FileNotFoundError:
                # Another shard may have recovered the same stale journal after
                # our glob but before this read.  Recovery is intentionally idempotent.
                continue
            _old_root, _old_records, old_lock_path, _old_cert = _journal_paths(
                old_journal, old_payload)
            if old_lock_path == (root / lock_rel).resolve():
                _recover_transaction_journal(
                    old_journal, held_lock=lock, release_held_lock=False)
        records = []
        for staging, final in pairs:
            staging, final = Path(staging).resolve(), Path(final).resolve()
            if not staging.is_relative_to(root) or not final.is_relative_to(root):
                raise RuntimeError("transaction path escapes its dataset root")
            if not _is_committed(staging):
                raise RuntimeError(f"transaction staging is not committed: {staging}")
            final.parent.mkdir(parents=True, exist_ok=True)
            had_old = final.exists()
            backup = final.with_name(f"{final.name}.backup_{token}")
            records.append({
                "staging": staging.relative_to(root).as_posix(),
                "final": final.relative_to(root).as_posix(),
                "backup": backup.relative_to(root).as_posix(),
                "had_old": had_old,
            })
        cert_rel = None
        if certification_path is not None:
            cert = Path(certification_path).resolve()
            if not cert.is_relative_to(root):
                raise RuntimeError("transaction certification path escapes dataset root")
            cert_rel = cert.relative_to(root).as_posix()
        journal = {"version": 2, "phase": "starting", "records": records,
                   "lock_path": lock_rel, "certification_path": cert_rel,
                   "transaction_id": transaction_id}
        _atomic_write_bytes(journal_path, json.dumps(journal, indent=2).encode())
        for rec in records:
            staging, final = root / rec["staging"], root / rec["final"]
            backup = root / rec["backup"]
            if rec["had_old"]:
                os.rename(final, backup)
            os.rename(staging, final)
    except Exception:
        if journal_path.exists():
            _recover_transaction_journal(journal_path, held_lock=lock)
        else:
            lock.release()
        raise

    # Once this phase marker is durable, recovery finalizes the new set instead of
    # rolling it back.  All promoted directories already contain commit markers.
    journal["phase"] = "promoted"
    _atomic_write_bytes(journal_path, json.dumps(journal, indent=2).encode())
    if certification_path is not None:
        return journal_path, lock
    _finalize_transaction_journal(journal_path, held_lock=lock)


def _journal_paths(journal_path: Path, payload: dict):
    journal_path = Path(journal_path).resolve()
    root = journal_path.parent.parent.resolve()
    if payload.get("version") != 2 or not isinstance(payload.get("records"), list):
        raise RuntimeError(f"invalid transaction journal: {journal_path}")

    def inside(rel):
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
            raise RuntimeError(f"unsafe path in transaction journal: {journal_path}")
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"transaction path escapes dataset root: {path}")
        return path

    records = []
    for rec in payload["records"]:
        if not isinstance(rec, dict) or not isinstance(rec.get("had_old"), bool):
            raise RuntimeError(f"invalid transaction record: {journal_path}")
        records.append({**rec, "staging_path": inside(rec.get("staging")),
                        "final_path": inside(rec.get("final")),
                        "backup_path": inside(rec.get("backup"))})
    lock_path = inside(payload.get("lock_path"))
    cert_path = (inside(payload["certification_path"])
                 if payload.get("certification_path") else None)
    if cert_path is not None:
        case_id = lock_path.stem
        if (lock_path.parent != root / ".transaction_locks"
                or cert_path != root / "case_status" / f"{case_id}.json"):
            raise RuntimeError(f"invalid production transaction identity: {journal_path}")
        n_case = 0
        for rec in records:
            final = rec["final_path"]
            staging = rec["staging_path"]
            backup = rec["backup_path"]
            if final.parent == root / "cases" and final.name == case_id:
                n_case += 1
            elif not (final.parent == root / "samples"
                      and final.name.startswith(f"{case_id}_b")):
                raise RuntimeError(f"unexpected transaction final path: {final}")
            if (staging.parent != final.parent
                    or not staging.name.startswith(f"{final.name}.staging_")
                    or backup.parent != final.parent
                    or not backup.name.startswith(f"{final.name}.backup_")):
                raise RuntimeError(f"invalid staging/backup path in {journal_path}")
        if n_case != 1:
            raise RuntimeError(f"transaction must contain exactly one case: {journal_path}")
    return root, records, lock_path, cert_path


def _finalize_transaction_journal(journal_path: Path,
                                  held_lock: _AdvisoryFileLock | None = None):
    payload = json.loads(Path(journal_path).read_text(encoding="utf-8"))
    _root, records, lock_path, _cert_path = _journal_paths(journal_path, payload)
    lock = held_lock or _AdvisoryFileLock(lock_path)
    owns_lock = held_lock is None
    if owns_lock and not lock.acquire(timeout_s=0.0):
        return False
    try:
        for rec in records:
            if rec["backup_path"].exists():
                shutil.rmtree(rec["backup_path"], ignore_errors=True)
        Path(journal_path).unlink(missing_ok=True)
        return True
    finally:
        lock.release()


def _transaction_is_certified(payload: dict, cert_path: Path | None) -> bool:
    txid = payload.get("transaction_id")
    if cert_path is None or not txid:
        return True
    try:
        status = json.loads(cert_path.read_text(encoding="utf-8"))
        return status.get("status") == "done" and status.get("transaction_id") == txid
    except Exception:
        return False


def _recover_transaction_journal(journal_path: Path,
                                 held_lock: _AdvisoryFileLock | None = None,
                                 release_held_lock: bool = True):
    """Finish or roll back one interrupted directory transaction."""
    journal_path = Path(journal_path).resolve()
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    _root, records, lock_path, cert_path = _journal_paths(journal_path, payload)
    lock = held_lock or _AdvisoryFileLock(lock_path)
    owns_lock = held_lock is None
    if owns_lock and not lock.acquire(timeout_s=0.0):
        return False
    try:
        if (payload.get("phase") == "promoted"
                and _transaction_is_certified(payload, cert_path)):
            for rec in records:
                if rec["backup_path"].exists():
                    shutil.rmtree(rec["backup_path"], ignore_errors=True)
            journal_path.unlink(missing_ok=True)
            return True
        for rec in reversed(records):
            staging, final, backup = (rec["staging_path"], rec["final_path"],
                                      rec["backup_path"])
            if rec["had_old"]:
                if backup.exists():
                    if final.exists():
                        shutil.rmtree(final, ignore_errors=True)
                    os.rename(backup, final)
            elif final.exists():
                shutil.rmtree(final, ignore_errors=True)
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        journal_path.unlink(missing_ok=True)
        return True
    finally:
        if owns_lock or release_held_lock:
            lock.release()


def _recover_transactions(output_dir: Path):
    journal_root = Path(output_dir) / ".transactions"
    if not journal_root.exists():
        return
    for journal_path in sorted(journal_root.glob("transaction_*.json")):
        try:
            _recover_transaction_journal(journal_path)
        except FileNotFoundError:
            # Parallel shards can both observe a stale journal; whichever acquires
            # the case lock first removes it.  The second observer has nothing left
            # to recover and must not abort its run.
            continue


def _rmtree_glob(root: Path, pattern: str):
    for p in root.glob(pattern):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except OSError:
                pass


# ===========================================================================
# Manifest schema (NN-loader compatible)
# ===========================================================================

MANIFEST_COLS = [
    "sample_id", "case_id", "base_shape_id", "shape_id", "body_type", "split",
    "material_id", "material_name",
    "E_L", "E_R", "E_T", "G_LR", "G_LT", "G_RT", "nu_LR", "nu_LT", "nu_RT", "density",
    "bridge_idx",
    "rel_case_response", "rel_admittance", "rel_params", "rel_case_dir",
    "rel_contour", "rel_step", "rel_mesh",
    "shape_type", "guitar_model",
    "bridge_req_x", "bridge_req_y", "bridge_req_z",
    "bridge_snap_x", "bridge_snap_y", "bridge_snap_z", "snap_distance_mm",
    "thickness", "top_plate_thickness_mm", "cavity_ratio", "cavity_volume_m3",
    "soundhole_type", "soundhole_diameter", "soundhole_area_m2",
    "soundhole_center_x", "soundhole_center_y",
    "solver_backend", "solver_revision", "plan_hash", "geometry_digest", "mesh_sha256",
    "transaction_id", "case_input_hash",
    "band_covered", "coverage_ok", "eig_max_residual",
    "struct_mass_dev", "acoustic_mass_dev", "struct_cond", "acoustic_cond",
    "solve_time_s", "n_modes_used", "attempt", "status", "qc_reason",
]


def _solver_backend(body_type: str) -> str:
    return "modal-coupled" if body_type == "hollow" else "full-harmonic"


# ===========================================================================
# Geometry (persistent, built ONCE per shape x body, reused across materials)
# ===========================================================================

def _shape_dir_name(shape_id: int) -> str:
    return f"shape_{int(shape_id):04d}"


def geometry_digest(shape_meta: dict, body_type: str) -> str:
    """Content hash of the geometry inputs for one (shape, body).  A mismatch on
    resume means the persisted mesh is stale and must be rebuilt."""
    payload = _canonical_json({
        "body_type": body_type,
        "contour": np.round(shape_meta["contour"], 6),
        "thickness": round(shape_meta["thickness"], 6),
        "top_plate_thickness_mm": round(shape_meta["top_plate_thickness_mm"], 6),
        "cavity_ratio": round(shape_meta["cavity_ratio"], 6),
        "bridge": np.round(shape_meta["bridge_pts_body"], 6),
        "soundhole_ok": shape_meta["soundhole_ok"],
        "soundhole_diameter": round(shape_meta["soundhole_diameter"], 6),
        "soundhole_center": shape_meta["soundhole_center"],
    })
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _geometry_artifact_hashes(shape_dir: Path, geom: dict) -> dict:
    shape_dir = Path(shape_dir).resolve()
    keys = ["rel_contour", "rel_source_contour", "rel_step",
            "rel_model_params", "rel_mesh"]
    if geom.get("body_type") == "hollow":
        keys.extend(["rel_air_step", "rel_air_mesh_meta"])
    out = {}
    for key in keys:
        rel = geom.get(key)
        if not rel:
            raise RuntimeError(f"geometry metadata missing {key}")
        path = (shape_dir / rel).resolve()
        if not path.is_relative_to(shape_dir):
            raise RuntimeError(f"geometry artifact path escapes shape dir: {rel}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"geometry artifact missing/empty: {path}")
        out[key] = _file_sha256(path)
    return out


def _positive_element_stats(value) -> bool:
    """Validate the per-domain 3-D element statistics emitted by Gmsh."""
    if not isinstance(value, dict):
        return False
    try:
        n_elements = value.get("n_elements")
        if (isinstance(n_elements, (bool, np.bool_))
                or not isinstance(n_elements, (int, np.integer))
                or int(n_elements) <= 0):
            return False
        metrics = np.asarray([
            value["min_edge_mm"], value["median_min_edge_mm"],
            value["p95_max_edge_mm"], value["max_edge_mm"],
        ], float)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(np.all(np.isfinite(metrics)) and np.all(metrics > 0.0))


def _geometry_metadata_matches_plan(shape_dir: Path, geom: dict,
                                    shape_meta: dict, body_type: str) -> bool:
    """Bind derived geometry metadata and coordinate frames back to the plan."""
    try:
        shape_dir = Path(shape_dir)
        source = np.load(shape_dir / geom["rel_source_contour"])
        actual = np.load(shape_dir / geom["rel_contour"])
        expected = np.asarray(shape_meta["contour"], float)
        expected_fem = np.column_stack([expected[:, 0], -expected[:, 1]])
        bridge_body = np.asarray(shape_meta["bridge_pts_body"], float)
        expected_bridges = np.column_stack([
            bridge_body[:, 0], -bridge_body[:, 1],
            np.full(bridge_body.shape[0], float(shape_meta["thickness"]))])
        hollow = body_type == "hollow"
        hole_body = shape_meta.get("soundhole_center") if hollow else None
        expected_hole = ([float(hole_body[0]), -float(hole_body[1])]
                         if hole_body is not None else None)
        expected_hole_body = ([float(hole_body[0]), float(hole_body[1])]
                              if hole_body is not None else None)
        if (geom.get("body_type") != body_type
                or geom.get("geometry_input") != "direct_contour"
                or geom.get("shape_type") != shape_meta["shape_type"]
                or geom.get("guitar_model") != shape_meta["guitar_model"]
                or float(geom.get("thickness", -1.0)) != float(shape_meta["thickness"])
                or float(geom.get("cavity_ratio", -1.0)) != float(shape_meta["cavity_ratio"])
                or geom.get("soundhole_type") != ("round" if hollow else "none")
                or float(geom.get("soundhole_diameter", -1.0))
                   != (float(shape_meta["soundhole_diameter"]) if hollow else 0.0)
                or geom.get("soundhole_center") != expected_hole
                or geom.get("soundhole_center_body") != expected_hole_body
                or not _array_matches(source, expected)
                or not _array_matches(actual, expected_fem)
                or not _array_matches(geom.get("bridge_coords", []), expected_bridges)
                or geom.get("conformal_interface_verified") is not True):
            return False

        model_meta = json.loads(
            (shape_dir / geom["rel_model_params"]).read_text(encoding="utf-8"))
        if (model_meta.get("step_file") != geom["rel_step"]
                or model_meta.get("path_semantics") != "relative_to_committed_shape_dir"
                or model_meta.get("geometry_input") != "direct_contour"
                or not _array_matches(model_meta.get("actual_contour_mm", []), expected_fem)):
            return False
        if hollow:
            model_air = model_meta.get("air") or {}
            planned_area_m2 = np.pi * (
                float(shape_meta["soundhole_diameter"]) * 0.5) ** 2 * 1e-6
            cad_volume_m3 = float(
                model_air.get("cavity_volume_mm3", float("nan"))) * 1e-9
            cross_tol = float(PRODUCTION_CONTRACT["geometry_crosscheck_rtol"])
            if (not np.isfinite(float(geom.get("t_hole_mm", float("nan"))))
                    or float(geom["t_hole_mm"]) <= 0.0
                    or not np.isfinite(float(geom.get("cavity_volume_m3", float("nan"))))
                    or float(geom["cavity_volume_m3"]) <= 0.0
                    or not np.isfinite(float(geom.get("soundhole_area_m2", float("nan"))))
                    or float(geom["soundhole_area_m2"]) <= 0.0
                    or model_air.get("air_step_file") != geom["rel_air_step"]
                    or not np.isclose(
                        float(model_air.get("top_plate_thickness_mm", float("nan"))),
                        float(geom["t_hole_mm"]), rtol=1e-12)
                    or not np.isclose(
                        cad_volume_m3, float(geom["cavity_volume_m3"]),
                        rtol=cross_tol)
                    or not np.isclose(
                        planned_area_m2, float(geom["soundhole_area_m2"]),
                        rtol=cross_tol)
                    or not np.isclose(float(geom.get(
                        "cad_cavity_volume_m3", float("nan"))), cad_volume_m3,
                        rtol=1e-12)
                    or not np.isclose(float(geom.get(
                        "planned_soundhole_area_m2", float("nan"))),
                        planned_area_m2, rtol=1e-12)):
                return False
            air_meta = json.loads(
                (shape_dir / geom["rel_air_mesh_meta"]).read_text(encoding="utf-8"))
            expected_radius_mm = 0.5 * float(shape_meta["soundhole_diameter"])
            position_tol_mm = max(0.1, expected_radius_mm * cross_tol)
            centroid = np.asarray(air_meta.get("soundhole_centroid_mm", []), float)
            target_min = float(PRODUCTION_CONTRACT["plate_min_size_mm"])
            target_max = float(PRODUCTION_CONTRACT["mesh_size_max_mm"])
            target_air = float(PRODUCTION_CONTRACT["air_mesh_size_mm"])
            target_hole = float(PRODUCTION_CONTRACT["soundhole_mesh_size_mm"])
            if (air_meta.get("msh_file") != geom["rel_mesh"]
                    or air_meta.get("path_semantics")
                       != "relative_to_committed_shape_dir"
                    or air_meta.get("conformal_interface_verified") is not True
                    or not np.isclose(
                        float(air_meta.get("top_plate_thickness_mm", float("nan"))),
                        float(geom["t_hole_mm"]), rtol=1e-12)
                    or not np.isclose(
                        float(air_meta.get("soundhole_thickness_mm", float("nan"))),
                        float(geom["t_hole_mm"]), rtol=1e-12)
                    or not np.isclose(float(air_meta.get("cavity_volume_m3", -1.0)),
                                      float(geom["cavity_volume_m3"]), rtol=1e-12)
                    or not np.isclose(float(air_meta.get("soundhole_area_m2", -1.0)),
                                      float(geom["soundhole_area_m2"]), rtol=1e-12)
                    or int(air_meta.get("n_fsi_surfaces", 0)) <= 0
                    or int(air_meta.get("n_soundhole_surfaces", 0)) <= 0
                    or int(air_meta.get("n_soundhole_connected_components", 0)) != 1
                    or int(air_meta.get("n_air_boundary_surfaces", -1))
                       != (int(air_meta.get("n_fsi_surfaces", 0))
                           + int(air_meta.get("n_soundhole_surfaces", 0)))
                    or float(air_meta.get(
                        "soundhole_area_rel_error", float("inf"))) > cross_tol
                    or float(air_meta.get(
                         "cavity_volume_rel_error", float("inf"))) > cross_tol):
                return False
            if (not np.isclose(float(air_meta.get(
                        "wood_mesh_size_target_min_mm", float("nan"))),
                        target_min, rtol=0.0, atol=0.0)
                    or not np.isclose(float(air_meta.get(
                        "wood_mesh_size_target_max_mm", float("nan"))),
                        target_max, rtol=0.0, atol=0.0)
                    or not np.isclose(float(air_meta.get(
                        "air_mesh_size_target_mm", float("nan"))),
                        target_air, rtol=0.0, atol=0.0)
                    or not np.isclose(float(air_meta.get(
                        "air_mesh_size_target", float("nan"))),
                        target_air, rtol=0.0, atol=0.0)
                    or not np.isclose(float(air_meta.get(
                        "soundhole_mesh_size_target_mm", float("nan"))),
                        target_hole, rtol=0.0, atol=0.0)
                    or not _positive_element_stats(
                        air_meta.get("wood_element_stats"))
                    or not _positive_element_stats(
                        air_meta.get("air_element_stats"))):
                return False
            if (not _array_matches(
                    air_meta.get("expected_soundhole_center_xy_mm", []), expected_hole)
                    or not np.isclose(float(air_meta.get(
                        "expected_soundhole_radius_mm", float("nan"))),
                        expected_radius_mm, rtol=0.0, atol=1e-12)
                    or centroid.shape != (3,) or not np.all(np.isfinite(centroid))
                    or not np.allclose(centroid[:2], expected_hole, rtol=0.0,
                                       atol=position_tol_mm)
                    or float(air_meta.get(
                        "soundhole_center_error_mm", float("inf"))) > position_tol_mm
                    or float(air_meta.get(
                        "soundhole_bbox_error_mm", float("inf"))) > position_tol_mm
                    or float(air_meta.get(
                        "soundhole_planarity_span_mm", float("inf"))) > 0.1
                    or not np.isclose(float(air_meta.get(
                        "expected_soundhole_opening_z_mm", float("nan"))),
                        float(model_air.get("cavity_floor_z_mm", float("nan"))),
                        rtol=0.0, atol=1e-9)
                    or float(air_meta.get(
                        "soundhole_opening_z_error_mm", float("inf"))) > 0.1):
                return False
            for key in ("n_soundhole_connected_components",
                        "expected_soundhole_center_xy_mm",
                        "expected_soundhole_radius_mm", "soundhole_centroid_mm",
                        "soundhole_center_error_mm", "soundhole_bbox_error_mm",
                        "soundhole_planarity_span_mm",
                        "expected_soundhole_opening_z_mm",
                        "soundhole_opening_z_error_mm",
                        "wood_mesh_size_target_min_mm",
                        "wood_mesh_size_target_max_mm",
                        "air_mesh_size_target_mm", "air_mesh_size_target",
                        "soundhole_mesh_size_target_mm", "wood_element_stats",
                        "air_element_stats"):
                if geom.get(key) != air_meta.get(key):
                    return False
        return True
    except Exception:
        return False


def _real_prepare_geometry(shape_meta: dict, body_type: str, shape_dir: Path,
                           freqs: np.ndarray) -> dict:  # pragma: no cover (CAD/Gmsh)
    """Build the STEP + (conformal) mesh for one (shape, body) into `shape_dir`.

    Server-only (needs CadQuery + Gmsh).  Reuses the proven dataset_gen data-dict /
    coordinate helpers.  Returns geometry metadata with paths RELATIVE to shape_dir
    (so the dir can be renamed on commit) plus the fenics-frame bridge coords and
    the geometry diagnostics QC needs (conformal_interface_verified, cavity volume,
    soundhole area).  Every bridge point is embedded as an exact mesh vertex.
    """
    from dataset_gen import _make_data_dict, _body_to_fenics_coords
    from model_builder import build_model
    from mesh_gen import generate_mesh, generate_conformal_air_mesh

    shape_dir.mkdir(parents=True, exist_ok=True)
    contour = np.asarray(shape_meta["contour"], float)
    thickness = float(shape_meta["thickness"])
    top_plate_thickness = float(shape_meta["top_plate_thickness_mm"])
    cavity_ratio = float(shape_meta["cavity_ratio"])
    bridge_pts = np.asarray(shape_meta["bridge_pts_body"], float)

    np.save(shape_dir / "source_contour_body_frame.npy", contour)

    soundhole_type = "round" if body_type == "hollow" else "none"
    soundhole_diam = float(shape_meta["soundhole_diameter"]) if body_type == "hollow" else 0.0
    soundhole_center = (tuple(shape_meta["soundhole_center"])
                        if (body_type == "hollow" and shape_meta["soundhole_center"]) else None)
    if body_type == "hollow" and soundhole_center is None:
        raise RuntimeError("hollow geometry requires a valid soundhole center")

    data = _make_data_dict(
        contour, body_type,
        (float(bridge_pts[0, 0]), float(bridge_pts[0, 1])),
        soundhole_type, soundhole_diam, soundhole_center, thickness, cavity_ratio,
        top_plate_thickness_mm=top_plate_thickness, include_raster=False)

    model_dir = shape_dir / "model"
    model_params = build_model(data, model_dir, export_air=(body_type == "hollow"))
    step_path = model_dir / "model.step"

    actual_contour = np.asarray(model_params.get(
        "actual_contour_mm", np.column_stack([contour[:, 0], -contour[:, 1]])), float)
    if actual_contour.shape != contour.shape or not np.all(np.isfinite(actual_contour)):
        raise RuntimeError("model_builder returned an invalid actual contour")
    np.save(shape_dir / "contour.npy", actual_contour)

    bridge_coords = [list(_body_to_fenics_coords(float(p[0]), float(p[1]), thickness))
                     for p in bridge_pts]

    mesh_dir = shape_dir / "mesh"
    soundhole_center_step = ([float(soundhole_center[0]), -float(soundhole_center[1])]
                             if soundhole_center else None)
    geom = {
        "body_type": body_type,
        "rel_contour": "contour.npy",
        "rel_source_contour": "source_contour_body_frame.npy",
        "rel_step": str((step_path).relative_to(shape_dir)).replace("\\", "/"),
        "rel_model_params": str((model_dir / "model_params.json").relative_to(
            shape_dir)).replace("\\", "/"),
        "bridge_coords": bridge_coords,
        "thickness": thickness,
        "top_plate_thickness_mm": top_plate_thickness,
        "cavity_ratio": cavity_ratio,
        "shape_type": shape_meta["shape_type"],
        "guitar_model": shape_meta["guitar_model"],
        "soundhole_type": soundhole_type,
        "soundhole_diameter": soundhole_diam,
        "soundhole_center": soundhole_center_step,
        "soundhole_center_body": list(soundhole_center) if soundhole_center else None,
        "geometry_input": model_params.get("geometry_input", "unknown"),
    }

    if body_type == "hollow":
        air = model_params.get("air", {})
        if not air.get("has_internal_air"):
            raise RuntimeError(f"air cavity export failed: {air.get('error')}")
        # Fail-loud: the CAD cavity MUST realise the requested top-plate thickness.
        cad_top = float(air["top_plate_thickness_mm"])
        cad_floor = float(air["cavity_floor_z_mm"])
        _TOP_TOL_MM = 1e-6
        if (abs(cad_top - top_plate_thickness) > _TOP_TOL_MM
                or abs((thickness - cad_floor) - top_plate_thickness) > _TOP_TOL_MM):
            raise RuntimeError(
                "CAD top-plate mismatch: requested "
                f"{top_plate_thickness:.6f} mm, CAD top={cad_top:.6f} mm, "
                f"thickness-cavity_floor_z={thickness - cad_floor:.6f} mm")
        air_step = air["air_step_file"]
        cad_cavity_volume_mm3 = float(
            air.get("cavity_volume_mm3", float("nan")))
        planned_soundhole_area_mm2 = np.pi * (soundhole_diam * 0.5) ** 2
        if (not np.isfinite(cad_cavity_volume_mm3)
                or cad_cavity_volume_mm3 <= 0.0
                or not np.isfinite(planned_soundhole_area_mm2)
                or planned_soundhole_area_mm2 <= 0.0):
            raise RuntimeError(
                "CAD returned invalid cavity volume or planned soundhole area")
        meta = generate_conformal_air_mesh(
            step_path, air_step, mesh_dir,
            bridge_coords=bridge_coords,
            t_hole_mm=float(air["top_plate_thickness_mm"]),
            soundhole_bc=str(PRODUCTION_CONTRACT["soundhole_bc"]),
            top_plate_thickness=float(air["top_plate_thickness_mm"]),
            back_plate_z=float(air["back_plate_z_mm"]),
            cavity_floor_z=float(air["cavity_floor_z_mm"]),
            mesh_size_min=float(PRODUCTION_CONTRACT["plate_min_size_mm"]),
            mesh_size_max=float(PRODUCTION_CONTRACT["mesh_size_max_mm"]),
            air_size=float(PRODUCTION_CONTRACT["air_mesh_size_mm"]),
            soundhole_size=float(PRODUCTION_CONTRACT["soundhole_mesh_size_mm"]),
            expected_soundhole_area_mm2=planned_soundhole_area_mm2,
            expected_cavity_volume_mm3=cad_cavity_volume_mm3,
            expected_soundhole_center_xy_mm=soundhole_center_step,
            expected_soundhole_radius_mm=0.5 * soundhole_diam,
            geometry_crosscheck_rtol=float(
                PRODUCTION_CONTRACT["geometry_crosscheck_rtol"]))
        msh_path = mesh_dir / "mesh_air.msh"
        if not meta.get("conformal_interface_verified", False):
            raise RuntimeError("conformal mesh generator returned an unverified interface")
        geom.update(
            rel_mesh=str(msh_path.relative_to(shape_dir)).replace("\\", "/"),
            rel_air_step=str(Path(air_step).relative_to(shape_dir)).replace("\\", "/"),
            rel_air_mesh_meta=str((mesh_dir / "air_mesh_meta.json").relative_to(
                shape_dir)).replace("\\", "/"),
            conformal_interface_verified=True,
            t_hole_mm=float(air["top_plate_thickness_mm"]),
            cavity_volume_m3=float(meta.get("cavity_volume_m3",
                                            (air.get("cavity_volume_mm3") or 0.0) * 1e-9)),
            soundhole_area_m2=float(meta.get("soundhole_area_m2", 0.0)),
            cad_cavity_volume_m3=cad_cavity_volume_mm3 * 1e-9,
            planned_soundhole_area_m2=planned_soundhole_area_mm2 * 1e-6,
            cavity_volume_rel_error=float(meta.get(
                "cavity_volume_rel_error", float("inf"))),
            soundhole_area_rel_error=float(meta.get(
                "soundhole_area_rel_error", float("inf"))),
            n_soundhole_connected_components=int(meta.get(
                "n_soundhole_connected_components", 0)),
            expected_soundhole_center_xy_mm=meta.get(
                "expected_soundhole_center_xy_mm"),
            expected_soundhole_radius_mm=meta.get("expected_soundhole_radius_mm"),
            soundhole_centroid_mm=meta.get("soundhole_centroid_mm"),
            soundhole_center_error_mm=meta.get("soundhole_center_error_mm"),
            soundhole_bbox_error_mm=meta.get("soundhole_bbox_error_mm"),
            soundhole_planarity_span_mm=meta.get("soundhole_planarity_span_mm"),
            expected_soundhole_opening_z_mm=meta.get(
                "expected_soundhole_opening_z_mm"),
            soundhole_opening_z_error_mm=meta.get("soundhole_opening_z_error_mm"),
            wood_mesh_size_target_min_mm=meta.get(
                "wood_mesh_size_target_min_mm"),
            wood_mesh_size_target_max_mm=meta.get(
                "wood_mesh_size_target_max_mm"),
            top_plate_mesh_size_target_mm=meta.get(
                "top_plate_mesh_size_target_mm"),
            air_mesh_size_target_mm=meta.get("air_mesh_size_target_mm"),
            air_mesh_size_target=meta.get("air_mesh_size_target"),
            soundhole_mesh_size_target_mm=meta.get(
                "soundhole_mesh_size_target_mm"),
            wood_element_stats=meta.get("wood_element_stats"),
            air_element_stats=meta.get("air_element_stats"),
        )
    else:
        top_plate_t = 0.0
        back_plate_t = 0.0
        msh_path = generate_mesh(
            step_file=step_path, bridge_coords=bridge_coords,
            mesh_size_min=float(PRODUCTION_CONTRACT["mesh_size_min_mm"]),
            mesh_size_max=float(PRODUCTION_CONTRACT["mesh_size_max_mm"]),
            output_dir=mesh_dir,
            top_plate_thickness=top_plate_t, back_plate_thickness=back_plate_t,
            plate_min_size=float(PRODUCTION_CONTRACT["plate_min_size_mm"]))
        geom.update(
            rel_mesh=str(Path(msh_path).relative_to(shape_dir)).replace("\\", "/"),
            conformal_interface_verified=True,   # not applicable to solid
        )
    return geom


def prepare_or_load_geometry(ctx, shape_meta: dict, body_type: str, shape_id: int,
                             shapes_root: Path, plan_hash_str: str) -> dict:
    """Serialize creation/replacement of one persisted shape across all shards."""
    name = _shape_dir_name(shape_id)
    lock = _AdvisoryFileLock(Path(shapes_root) / ".shape_locks" / f"{name}.lock")
    if not lock.acquire():
        raise RuntimeError(f"timed out waiting for geometry lock {name}")
    try:
        shape_dir = Path(shapes_root) / name
        trash = sorted(Path(shapes_root).glob(f"{name}.trash_*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not shape_dir.exists():
            recoverable = next((p for p in trash if _is_committed(p)), None)
            if recoverable is not None:
                os.rename(recoverable, shape_dir)
        for stale in trash:
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
        _rmtree_glob(Path(shapes_root), f"{name}.staging_*")
        return _prepare_or_load_geometry_locked(
            ctx, shape_meta, body_type, shape_id, shapes_root, plan_hash_str)
    finally:
        lock.release()


def _prepare_or_load_geometry_locked(ctx, shape_meta: dict, body_type: str,
                                     shape_id: int, shapes_root: Path,
                                     plan_hash_str: str) -> dict:
    """Build geometry ONCE per (shape, body) and persist it; on resume, re-validate
    (digest + plan_hash + files exist) and reuse.  Returns geom with ABSOLUTE
    msh/step paths resolved against the committed shape dir."""
    name = _shape_dir_name(shape_id)
    shape_dir = shapes_root / name
    digest = geometry_digest(shape_meta, body_type)
    gj = shape_dir / "geometry.json"

    if _is_committed(shape_dir) and gj.exists():
        try:
            meta = json.loads(gj.read_text())
            current_hashes = _geometry_artifact_hashes(shape_dir, meta)
            files_ok = current_hashes == meta.get("artifact_sha256", {})
            if (meta.get("geometry_digest") == digest and
                    meta.get("plan_hash") == plan_hash_str and files_ok and
                    meta.get("mesh_sha256") == current_hashes.get("rel_mesh") and
                    _geometry_metadata_matches_plan(
                        shape_dir, meta, shape_meta, body_type)):
                return _resolve_geom_paths(meta, shape_dir)
        except Exception:
            pass  # fall through to rebuild

    # (re)build into a staging dir, then commit
    staging = shapes_root / f"{name}.staging_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    geom = ctx.prepare_geometry(shape_meta, body_type, staging, None)
    if geom.get("body_type") != body_type:
        raise RuntimeError(
            f"geometry backend returned body_type={geom.get('body_type')} for {body_type}")
    _canonicalize_geometry_sidecars(staging, geom)
    if not _geometry_metadata_matches_plan(staging, geom, shape_meta, body_type):
        raise RuntimeError("geometry backend output does not match the immutable plan")
    artifact_hashes = _geometry_artifact_hashes(staging, geom)
    geom = {**geom, "geometry_digest": digest, "plan_hash": plan_hash_str,
            "shape_id": int(shape_id), "body_type": body_type,
            "artifact_sha256": artifact_hashes,
            "mesh_sha256": artifact_hashes["rel_mesh"]}
    _atomic_write_bytes(staging / "geometry.json",
                        json.dumps(geom, indent=2, default=_json_default).encode())
    _atomic_write_bytes(staging / "params.json", json.dumps({
        "shape_id": int(shape_id), "body_type": body_type,
        "shape_type": shape_meta["shape_type"], "guitar_model": shape_meta["guitar_model"],
        "thickness": float(shape_meta["thickness"]),
        "top_plate_thickness_mm": (float(shape_meta["top_plate_thickness_mm"])
                                   if body_type == "hollow" else 0.0),
        "cavity_ratio": (float(shape_meta["cavity_ratio"])
                           if body_type == "hollow" else 0.0),  # derived from top plate
        "bridge_pts_body": np.asarray(shape_meta["bridge_pts_body"], float).tolist(),
        "soundhole_center": (shape_meta["soundhole_center"]
                              if body_type == "hollow" else None),
        "soundhole_diameter": (float(shape_meta["soundhole_diameter"])
                               if body_type == "hollow" else 0.0),
    }, indent=2, default=_json_default).encode())
    _write_marker(staging)
    _commit_dirs_transaction([(staging, shape_dir)],
                             transaction_root=shapes_root.parent)
    return _resolve_geom_paths(geom, shape_dir)


def _canonicalize_geometry_sidecars(shape_dir: Path, geom: dict):
    """Replace transient staging paths in persisted CAD/mesh metadata."""
    params_path = shape_dir / geom["rel_model_params"]
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["step_file"] = geom["rel_step"]
    params["path_semantics"] = "relative_to_committed_shape_dir"
    if geom.get("rel_air_step"):
        air = params.get("air") if isinstance(params.get("air"), dict) else {}
        air["air_step_file"] = geom["rel_air_step"]
        params["air"] = air
    _atomic_write_bytes(params_path, json.dumps(params, indent=2).encode())

    if geom.get("rel_air_mesh_meta"):
        meta_path = shape_dir / geom["rel_air_mesh_meta"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["msh_file"] = geom["rel_mesh"]
        meta["path_semantics"] = "relative_to_committed_shape_dir"
        _atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode())


def _resolve_geom_paths(geom: dict, shape_dir: Path) -> dict:
    shape_dir = Path(shape_dir).resolve()
    out = dict(geom)
    out["shape_dir"] = str(shape_dir)
    out["msh_path"] = str(shape_dir / geom["rel_mesh"])
    if "rel_step" in geom:
        out["step_path"] = str(shape_dir / geom["rel_step"])
    return out


# ===========================================================================
# Solve (subprocess worker per case) + mode budget / retry
# ===========================================================================

def initial_mode_budget(case: dict, config: dict) -> dict:
    if case["body_type"] == "hollow":
        return {"struct_fmax": float(config["coupled_struct_fmax"]),
                "acoustic_fmax": float(config["coupled_acoustic_fmax"]),
                "n_struct_modes": int(config["coupled_n_struct_modes"]),
                "n_acoustic_modes": int(config["coupled_n_acoustic_modes"]),
                "n_attach_acoustic": int(config["coupled_n_attach_acoustic"]),
                "port_end_corrections": int(config["port_end_corrections"])}
    return {"modal_fmax": float(config["solid_modal_fmax"]),
            "n_modes": int(config["solid_n_modes"])}


def escalate_mode_budget(budget: dict, body_type: str, factor: float = 1.5,
                         cap: int = 2000) -> dict:
    """Increase ONLY the number of requested modes (the cutoff frequency is fixed at
    7500 Hz).  Geometry/mesh are never regenerated for a retry."""
    b = dict(budget)
    def increase(value):
        value = int(value)
        if value >= cap:
            return value
        return max(value + 1, min(cap, int(round(value * factor))))
    if body_type == "hollow":
        b["n_struct_modes"] = increase(b["n_struct_modes"])
        b["n_acoustic_modes"] = increase(b["n_acoustic_modes"])
    else:
        b["n_modes"] = increase(b["n_modes"])
    return b


def _budget_n_modes(budget: dict, body_type: str) -> int:
    return int(budget["n_struct_modes"]) if body_type == "hollow" else int(budget["n_modes"])


def _case_input_payload(case: dict, material: dict, geom: dict,
                        freqs: np.ndarray, budget: dict) -> dict:
    f = np.ascontiguousarray(np.asarray(freqs, dtype="<f8"))
    material_payload = {k: v for k, v in material.items()
                        if isinstance(v, (int, float, str, bool, np.number))}
    return {
        "case_id": case["case_id"],
        "base_shape_id": int(case["base_shape_id"]),
        "shape_id": int(case["shape_id"]),
        "body_type": case["body_type"],
        "material_id": case["material_id"],
        "material": material_payload,
        "geometry_digest": geom.get("geometry_digest"),
        "mesh_sha256": geom.get("mesh_sha256"),
        "geometry_physics": {
            "top_plate_thickness_mm": geom.get("top_plate_thickness_mm"),
            "t_hole_mm": geom.get("t_hole_mm"),
            "cavity_volume_m3": geom.get("cavity_volume_m3"),
            "soundhole_area_m2": geom.get("soundhole_area_m2"),
            "conformal_interface_verified": geom.get("conformal_interface_verified"),
        },
        "bridge_coords": np.asarray(geom["bridge_coords"], float).tolist(),
        "frequency_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
        "frequency_count": int(f.size),
        "frequency_min_hz": float(f[0]),
        "frequency_max_hz": float(f[-1]),
        "mode_budget": budget,
        "production_contract": PRODUCTION_CONTRACT,
        "solver_revision": (HOLLOW_SOLVER_REVISION if case["body_type"] == "hollow"
                            else SOLID_SOLVER_REVISION),
    }


def case_input_hash(case: dict, material: dict, geom: dict,
                    freqs: np.ndarray, budget: dict) -> str:
    return hashlib.sha256(
        _canonical_json(_case_input_payload(case, material, geom, freqs, budget)).encode()
    ).hexdigest()


def _real_solve_case(case: dict, shape_meta: dict, material: dict, geom: dict,
                     freqs: np.ndarray, budget: dict, staging: Path) -> dict:  # pragma: no cover
    """Run ONE case in a fresh subprocess (so PETSc/SLEPc memory is reclaimed on
    exit).  Writes the solver artifacts into `staging`, then returns an in-memory
    response (arrays loaded back from the worker's multi-NPZ) for QC + fan-out."""
    staging.mkdir(parents=True, exist_ok=True)
    body = case["body_type"]
    input_hash = case_input_hash(case, material, geom, freqs, budget)
    staging_abs = Path(staging).resolve()
    mesh_rel = os.path.relpath(Path(geom["msh_path"]).resolve(), staging_abs).replace(
        "\\", "/")
    spec = {"body_type": body, "msh_path": mesh_rel,
            "material": {k: v for k, v in material.items()
                         if isinstance(v, (int, float, str))},
            "bridge_coords": [list(map(float, p)) for p in geom["bridge_coords"]],
            "freqs": [float(x) for x in np.asarray(freqs, float)],
            "out_dir": ".", "plan_hash": geom.get("plan_hash"),
            "geometry_digest": geom.get("geometry_digest"),
            "mesh_sha256": geom.get("mesh_sha256"),
            "case_input_hash": input_hash,
            "production_contract": PRODUCTION_CONTRACT}
    if body == "hollow":
        spec["hollow"] = {**budget, "t_hole_mm": geom.get("t_hole_mm")}
    else:
        spec["solid"] = dict(budget)
    spec_path = staging / "solve_spec.json"
    _atomic_write_bytes(spec_path, json.dumps(spec, indent=2, default=_json_default).encode())

    worker = str(ROOT / "_dataset_solve_worker.py")
    log_path = staging / "worker.log"
    timeout_s = float(case.get("_case_timeout_s", DEFAULT_CASE_TIMEOUT_S))
    with open(log_path, "wb") as log:
        try:
            proc = subprocess.run(
                [sys.executable, worker, str(spec_path)], check=False,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {
                "run_status": "failed", "diagnostics": {}, "bridges": [],
                "worker_returncode": None,
                "error": f"worker timed out after {timeout_s:g} seconds",
            }

    rp = staging / "worker_result.json"
    if not rp.exists():
        return {"run_status": "failed", "diagnostics": {},
                "error": f"worker produced no result (returncode={proc.returncode})",
                "bridges": [], "worker_returncode": int(proc.returncode)}
    result = json.loads(rp.read_text())
    if (proc.returncode != 0 or result.get("run_status") != "complete"
            or not result.get("multi_npz") or not result.get("peak_npz")):
        return {"run_status": "failed", "diagnostics": result.get("diagnostics", {}),
                "error": result.get("error") or f"worker returncode={proc.returncode}",
                "bridges": result.get("bridges", []),
                "worker_returncode": int(proc.returncode)}

    multi_path = staging / result["multi_npz"]
    peak_path = staging / result["peak_npz"]
    if not multi_path.is_file() or not peak_path.is_file():
        return {"run_status": "failed", "diagnostics": result.get("diagnostics", {}),
                "error": "worker result references a missing response/peak artifact",
                "bridges": result.get("bridges", []),
                "worker_returncode": int(proc.returncode)}
    with np.load(peak_path, allow_pickle=False) as peak_file:
        peaks = {key: np.asarray(peak_file[key]).copy() for key in (
            "peak_frequency_hz", "peak_amplitude_db", "peak_q", "peak_mask",
            "peak_count_total", "peak_truncated")}
    with np.load(multi_path, allow_pickle=False) as d:
        resp = {
            "run_status": "complete",
            "solver_revision": result.get("solver_revision"),
            "frequencies": np.asarray(d["frequencies"], float).copy(),
            "admittance": np.atleast_2d(np.asarray(d["admittance"], complex)).copy(),
            "bridges": result.get("bridges", []),
            "diagnostics": {**result.get("diagnostics", {}),
                            "conformal_interface_verified": bool(
                                geom.get("conformal_interface_verified", False))},
            "timing": result.get("timing", {}),
            "A0": result.get("A0", {}),
            "case_input_hash": result.get("case_input_hash"),
            "artifact_dir": str(staging),
            "worker_returncode": int(proc.returncode),
            "peaks": peaks,
        }
        if body == "hollow":
            resp["p_bar"] = np.atleast_2d(np.asarray(d["p_bar"], complex)).copy()
            resp["U_h"] = np.atleast_2d(np.asarray(d["U_h"], complex)).copy()
    return resp


# ===========================================================================
# Case artifact (staged, committed after QC) + per-bridge fan-out
# ===========================================================================

def write_case_artifact(staging: Path, case: dict, response: dict, freqs: np.ndarray,
                        plan_hash_str: str, geom: dict, budget: dict, attempt: int,
                        transaction_id: str, material: dict):
    """Write the canonical, self-describing case artifact into `staging` (marker
    LAST).  Solver-native files the worker already wrote alongside are preserved."""
    staging.mkdir(parents=True, exist_ok=True)
    body = case["body_type"]
    Y = np.asarray(response["admittance"], complex)
    arrays = {"frequencies": np.asarray(freqs, float), "admittance": Y}
    if body == "hollow":
        arrays["p_bar"] = np.asarray(response["p_bar"], complex)
        arrays["U_h"] = np.asarray(response["U_h"], complex)
    np.savez(str(staging / "case_admittance.npz"), **arrays)

    from peak_labels import magnitude_db
    peaks = response["peaks"]
    np.savez_compressed(
        str(staging / "case_response.npz"),
        schema_version=np.array(PRODUCTION_CONTRACT["output_schema"]),
        frequencies_hz=np.asarray(freqs, float),
        log_magnitude_db=magnitude_db(Y),
        peak_frequency_hz=np.asarray(peaks["peak_frequency_hz"], float),
        peak_amplitude_db=np.asarray(peaks["peak_amplitude_db"], float),
        peak_q=np.asarray(peaks["peak_q"], float),
        peak_mask=np.asarray(peaks["peak_mask"], bool),
        peak_count_total=np.asarray(peaks["peak_count_total"], np.int64),
        peak_truncated=np.asarray(peaks["peak_truncated"], bool),
    )

    native_model_name = (
        "reduced_model.npz" if body == "hollow" else "solid_full_eigen.npz")
    native_model_path = staging / native_model_name
    if not native_model_path.is_file():
        raise RuntimeError(f"solver did not preserve canonical {native_model_name}")
    native_model_sha256 = _file_sha256(native_model_path)

    _atomic_write_bytes(staging / "case_meta.json", json.dumps({
        "schema_version": DATASET_SCHEMA_VERSION,
        "case_id": case["case_id"], "base_shape_id": case["base_shape_id"],
        "shape_id": case["shape_id"], "body_type": body,
        "material_id": case["material_id"], "split": case["split"],
        "n_bridges": case["n_bridges"], "bridges": response.get("bridges", []),
        "solver_revision": response.get("solver_revision"),
        "run_status": response.get("run_status"),
        "plan_hash": plan_hash_str, "attempt": int(attempt),
        "geometry_digest": geom.get("geometry_digest"),
        "mesh_sha256": geom.get("mesh_sha256"),
        "transaction_id": transaction_id,
        "case_input_hash": response.get("case_input_hash"),
        "material": {k: v for k, v in material.items()},
        "production_contract": PRODUCTION_CONTRACT,
        "output_schema": PRODUCTION_CONTRACT["output_schema"],
        "nn_response": "case_response.npz",
        "units": PRODUCTION_CONTRACT["response_units"],
        "coordinate_units": "mm",
        "native_model": native_model_name,
        "native_model_sha256": native_model_sha256,
        "mode_budget": budget, "n_modes_used": _budget_n_modes(budget, body),
        "diagnostics": response.get("diagnostics", {}),
        "timing": response.get("timing", {}),
    }, indent=2, default=_json_default).encode())
    _atomic_write_bytes(staging / "timing.json",
                        json.dumps(response.get("timing", {}), indent=2,
                                   default=_json_default).encode())
    _atomic_write_bytes(staging / "case_run_status.json",
                        json.dumps({"status": response.get("run_status")}).encode())
    if body == "hollow":
        _atomic_write_bytes(staging / "A0.json",
                            json.dumps(response.get("A0", {}), default=_json_default,
                                       allow_nan=False).encode())
    spec_path = staging / "solve_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        # Replace process/staging-specific absolute paths with portable paths in the
        # committed provenance record.  The worker has already completed here.
        spec["out_dir"] = "."
        spec["msh_path"] = (Path("../..") / "shapes" /
                            _shape_dir_name(case["shape_id"]) /
                            geom["rel_mesh"]).as_posix()
        _atomic_write_bytes(spec_path, json.dumps(spec, indent=2,
                                                  default=_json_default).encode())
    _write_marker(staging)


def _base_row(case, shape_meta, material, response, geom, plan_hash_str,
              solve_time_s, budget, attempt):
    d = response.get("diagnostics", {})
    body = case["body_type"]
    name = _shape_dir_name(case["shape_id"])
    sc = geom.get("soundhole_center") or [None, None]
    row = {
        "case_id": case["case_id"], "base_shape_id": case["base_shape_id"],
        "shape_id": case["shape_id"], "body_type": body, "split": case["split"],
        "material_id": case["material_id"],
        "material_name": material.get("material_name", ""),
        "rel_case_dir": f"cases/{case['case_id']}",
        "rel_case_response": f"cases/{case['case_id']}/case_response.npz",
        "rel_contour": f"shapes/{name}/{geom.get('rel_contour', 'contour.npy')}",
        "rel_step": f"shapes/{name}/{geom['rel_step']}" if geom.get("rel_step") else "",
        "rel_mesh": f"shapes/{name}/{geom['rel_mesh']}" if geom.get("rel_mesh") else "",
        "shape_type": shape_meta["shape_type"], "guitar_model": shape_meta["guitar_model"],
        "thickness": shape_meta["thickness"],
        "top_plate_thickness_mm": (shape_meta["top_plate_thickness_mm"]
                                   if body == "hollow" else 0.0),
        "cavity_ratio": shape_meta["cavity_ratio"] if body == "hollow" else 0.0,
        "cavity_volume_m3": d.get("cavity_volume_m3", 0.0),
        "soundhole_type": ("round" if body == "hollow" else "none"),
        "soundhole_diameter": (shape_meta["soundhole_diameter"] if body == "hollow" else 0.0),
        "soundhole_area_m2": d.get("soundhole_area_m2", 0.0),
        "soundhole_center_x": sc[0], "soundhole_center_y": sc[1],
        "solver_backend": _solver_backend(body),
        "solver_revision": response.get("solver_revision"), "plan_hash": plan_hash_str,
        "geometry_digest": geom.get("geometry_digest"),
        "mesh_sha256": geom.get("mesh_sha256"),
        "band_covered": d.get("band_covered", ""), "coverage_ok": d.get("coverage_ok", ""),
        "eig_max_residual": d.get("max_eigen_residual", ""),
        "struct_mass_dev": d.get("structural_mass_orthonormality_max_dev", ""),
        "acoustic_mass_dev": d.get("acoustic_mass_orthonormality_max_dev", ""),
        "struct_cond": d.get("structural_reduced_mass_condition", ""),
        "acoustic_cond": d.get("acoustic_reduced_mass_condition", ""),
        "solve_time_s": round(solve_time_s, 2),
        "n_modes_used": _budget_n_modes(budget, body),
        "attempt": int(attempt), "status": "done", "qc_reason": "",
    }
    row.update(material_columns(material))
    return row


def fan_out_case(case: dict, shape_meta: dict, material: dict, response: dict,
                 freqs: np.ndarray, plan_hash_str: str, samples_root: Path,
                 geom: dict, budget: dict, solve_time_s: float, attempt: int,
                 transaction_id: str,
                 ) -> tuple[list[dict], list[tuple[Path, Path]]]:
    """Write one NN-compatible sample per bridge (commit-marker atomic) from the
    committed case response, and return the manifest rows.  Called ONLY after QC."""
    Y = np.asarray(response["admittance"], complex)          # (n_bridges, n_freq)
    bridges = response["bridges"]
    base = _base_row(case, shape_meta, material, response, geom, plan_hash_str,
                     solve_time_s, budget, attempt)
    rows = []
    staged_pairs = []
    for bi in range(case["n_bridges"]):
        sample_id = f"{case['case_id']}_b{bi}"
        final = samples_root / sample_id
        staging = samples_root / f"{sample_id}.staging_{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        req = [float(v) for v in bridges[bi]["bridge_requested_xyz"]]
        snp = [float(v) for v in bridges[bi]["bridge_snapped_xyz"]]
        snap_mm = float(bridges[bi]["snap_distance_mm"])
        np.savez(str(staging / "admittance.npz"), frequencies=freqs, admittance=Y[bi])
        params = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "sample_id": sample_id, "case_id": case["case_id"],
            "base_shape_id": case["base_shape_id"], "shape_id": case["shape_id"],
            "material_id": case["material_id"], "bridge_idx": bi,
            "split": case["split"], "body_type": case["body_type"],
            "shape_type": shape_meta["shape_type"], "guitar_model": shape_meta["guitar_model"],
            "bridge_coords": req, "bridge_requested_xyz": req,
            "bridge_snapped_xyz": snp, "snap_distance_mm": snap_mm,
            "response_at": "nearest_mesh_node_to_requested_bridge",
            "units": {
                "coordinates": "mm", "material_stiffness": "Pa",
                "density": "kg/m^3", **PRODUCTION_CONTRACT["response_units"],
            },
            "coordinate_frame": {
                "geometry": "FEM/STEP global XYZ",
                "material_axes": PRODUCTION_CONTRACT["material_axes_fem"],
                "time_convention": PRODUCTION_CONTRACT["time_convention"],
            },
            "sampled_response_role": PRODUCTION_CONTRACT["sampled_response_role"],
            "rel_case_response": f"cases/{case['case_id']}/case_response.npz",
            "rel_case_model": (
                f"cases/{case['case_id']}/"
                + ("reduced_model.npz" if case["body_type"] == "hollow"
                   else "solid_full_eigen.npz")),
            "material": {k: v for k, v in material.items()},
            "material_columns": material_columns(material),
            "thickness": shape_meta["thickness"],
            "top_plate_thickness_mm": (shape_meta["top_plate_thickness_mm"]
                                       if case["body_type"] == "hollow" else 0.0),
            "cavity_ratio": (shape_meta["cavity_ratio"] if case["body_type"] == "hollow" else 0.0),
            "cavity_volume_m3": (response.get("diagnostics", {}).get(
                "cavity_volume_m3", 0.0) if case["body_type"] == "hollow" else 0.0),
            "soundhole_type": geom.get("soundhole_type", "none"),
            "soundhole_diameter": geom.get("soundhole_diameter", 0.0),
            "soundhole_area_m2": (response.get("diagnostics", {}).get(
                "soundhole_area_m2", 0.0) if case["body_type"] == "hollow" else 0.0),
            "soundhole_center": geom.get("soundhole_center"),
            "rel_contour": base["rel_contour"], "rel_case_dir": base["rel_case_dir"],
            "solver_backend": _solver_backend(case["body_type"]),
            "solver_revision": response.get("solver_revision"),
            "plan_hash": plan_hash_str,
            "geometry_digest": geom.get("geometry_digest"),
            "mesh_sha256": geom.get("mesh_sha256"),
            "transaction_id": transaction_id,
            "case_input_hash": response.get("case_input_hash"),
            "attempt": int(attempt),
            "mode_budget": budget,
            "diagnostics": response.get("diagnostics", {}),
        }
        _atomic_write_bytes(staging / "sample_params.json",
                            json.dumps(params, indent=2, default=_json_default).encode())
        _write_marker(staging)
        staged_pairs.append((staging, final))

        row = dict(base)
        row.update({
            "sample_id": sample_id, "bridge_idx": bi,
            "rel_case_response": f"cases/{case['case_id']}/case_response.npz",
            "rel_admittance": f"samples/{sample_id}/admittance.npz",
            "rel_params": f"samples/{sample_id}/sample_params.json",
            "bridge_req_x": req[0], "bridge_req_y": req[1], "bridge_req_z": req[2],
            "bridge_snap_x": snp[0], "bridge_snap_y": snp[1], "bridge_snap_z": snp[2],
            "snap_distance_mm": round(snap_mm, 4),
            "transaction_id": transaction_id,
            "case_input_hash": response.get("case_input_hash"),
        })
        rows.append(row)
    return rows, staged_pairs


# ===========================================================================
# Resume: re-validate a committed case (never trust status=done alone)
# ===========================================================================

def _grid_ok(arr, freqs) -> bool:
    try:
        a = np.asarray(arr, float)
        return (a.shape == freqs.shape and np.all(np.isfinite(a))
                and np.allclose(a, freqs, rtol=0, atol=1e-9)
                and np.all(np.diff(a) > 0))
    except (TypeError, ValueError):
        return False


def _array_matches(actual, expected) -> bool:
    """Exact, finite array match used to bind native and canonical artifacts."""
    try:
        a = np.asarray(actual)
        e = np.asarray(expected)
        return (a.shape == e.shape and np.all(np.isfinite(a))
                and np.array_equal(a, e))
    except (TypeError, ValueError):
        return False


def _validate_native_case_artifacts(case_dir: Path, body: str, freqs: np.ndarray,
                                    Y: np.ndarray, p_bar: np.ndarray | None,
                                    U_h: np.ndarray | None, meta: dict,
                                    want_rev: str) -> bool:
    """Cross-check solver-native output against the canonical committed response.

    Keeping both is useful only if they cannot silently disagree.  This also makes
    corruption of a native file invalidate resume instead of leaving a plausible
    but internally inconsistent case directory.
    """
    try:
        model_name = (
            "reduced_model.npz" if body == "hollow" else "solid_full_eigen.npz")
        model_bridge_req = model_bridge_snp = None
        with np.load(case_dir / model_name, allow_pickle=False) as model:
            if body == "hollow":
                matrix_keys = ("K_r", "M_r", "Ka_r", "Ma_r", "G_r")
                vector_keys = ("beta_m", "q_b", "omega_s_modes_rad_s",
                               "omega_a_elastic_rad_s")
                scalar_keys = ("soundhole_area_m2", "soundhole_inertance_kg_m4",
                               "cavity_volume_m3", "air_speed_m_s",
                               "air_density_kg_m3", "rayleigh_alpha",
                               "rayleigh_beta", "force_n")
                arrays = [np.asarray(model[k]) for k in matrix_keys + vector_keys]
                K_r, M_r, Ka_r, Ma_r, G_r, beta_m, q_b = arrays[:7]
                omega_s, omega_a = arrays[7:9]
                scalars = {k: float(np.asarray(model[k]).item()) for k in scalar_keys}
                m_u = K_r.shape[0] if K_r.ndim == 2 else -1
                m_p = Ka_r.shape[0] if Ka_r.ndim == 2 else -1
                shapes_ok = (
                    m_u > 0 and m_p > 0
                    and K_r.shape == M_r.shape == (m_u, m_u)
                    and Ka_r.shape == Ma_r.shape == (m_p, m_p)
                    and G_r.shape == (m_p, m_u)
                    and beta_m.shape == (m_p,)
                    and q_b.shape == (Y.shape[0], m_u))
                if (str(np.asarray(model["schema_version"]).item())
                        != "air-coupled-reduced-model-v1"
                        or not shapes_ok
                        or not all(a.size and np.all(np.isfinite(a)) for a in arrays)
                        or omega_s.ndim != 1 or omega_a.ndim != 1
                        or np.any(omega_s < 0.0) or np.any(omega_a < 0.0)
                        or np.any(np.diff(omega_s) < 0.0)
                        or np.any(np.diff(omega_a) < 0.0)
                        or not all(np.isfinite(v) for v in scalars.values())
                        or scalars["soundhole_area_m2"] <= 0.0
                        or scalars["soundhole_inertance_kg_m4"] <= 0.0
                        or scalars["cavity_volume_m3"] <= 0.0
                        or scalars["air_speed_m_s"] <= 0.0
                        or scalars["air_density_kg_m3"] <= 0.0
                        or scalars["force_n"] <= 0.0
                        or not np.isclose(
                            scalars["soundhole_area_m2"],
                            float(meta["diagnostics"]["soundhole_area_m2"]), rtol=1e-12)
                        or not np.isclose(
                            scalars["cavity_volume_m3"],
                            float(meta["diagnostics"]["cavity_volume_m3"]), rtol=1e-12)
                        or not np.isclose(scalars["air_speed_m_s"],
                                          PRODUCTION_CONTRACT["air_speed_m_s"], rtol=0.0)
                        or not np.isclose(scalars["air_density_kg_m3"],
                                          PRODUCTION_CONTRACT["air_density_kg_m3"], rtol=0.0)
                        or not np.isclose(scalars["rayleigh_alpha"],
                                          PRODUCTION_CONTRACT["rayleigh_alpha"], rtol=0.0)
                        or not np.isclose(scalars["rayleigh_beta"],
                                          PRODUCTION_CONTRACT["rayleigh_beta"], rtol=0.0)
                        or not np.isclose(scalars["force_n"],
                                          PRODUCTION_CONTRACT["force_n"], rtol=0.0)
                        or str(np.asarray(model["time_convention"]).item())
                           != PRODUCTION_CONTRACT["time_convention"]
                        or str(np.asarray(model["response_units"]).item())
                           != "Y:m/s/N;p_bar:Pa;U_h:m^3/s"):
                    return False
            else:
                eigenfreq = np.asarray(model["eigenfrequency_hz"], float)
                residues = np.asarray(model["bridge_residue_inv_kg"], float)
                zeta = np.asarray(model["zeta"], float)
                candidate_f = np.asarray(model["candidate_frequency_hz"], float)
                candidate_zeta = np.asarray(model["candidate_zeta"], float)
                candidate_eligible = np.asarray(model["candidate_eligible"], bool)
                candidate_y = np.asarray(model["candidate_admittance"], complex)
                model_bridge_req = np.asarray(model["bridge_requested_xyz_mm"], float)
                model_bridge_snp = np.asarray(model["bridge_snapped_xyz_mm"], float)
                force_n = float(np.asarray(model["force_n"]).item())
                alpha = float(np.asarray(model["rayleigh_alpha"]).item())
                beta = float(np.asarray(model["rayleigh_beta"]).item())
                if (str(np.asarray(model["schema_version"]).item())
                        != "solid-full-eigen-aux-v1"
                        or eigenfreq.ndim != 1 or eigenfreq.size == 0
                        or residues.shape != (Y.shape[0], eigenfreq.size)
                        or zeta.shape != eigenfreq.shape
                        or candidate_f.ndim != 1
                        or candidate_zeta.shape != candidate_f.shape
                        or candidate_eligible.shape != (Y.shape[0], candidate_f.size)
                        or candidate_y.shape != (Y.shape[0], candidate_f.size)
                        or not np.all(np.isfinite(eigenfreq))
                        or not np.all(np.isfinite(residues))
                        or not np.all(np.isfinite(zeta))
                        or not np.all(np.isfinite(candidate_f))
                        or not np.all(np.isfinite(candidate_zeta))
                        or not np.all(np.isfinite(candidate_y))
                        or np.any(eigenfreq <= 0.0) or np.any(residues < 0.0)
                        or np.any(zeta <= 0.0) or np.any(candidate_f <= 0.0)
                        or np.any(candidate_zeta <= 0.0)
                        or np.any(np.diff(eigenfreq) <= 0.0)
                        or np.any(np.diff(candidate_f) <= 0.0)
                        or float(np.max(residues)) <= 0.0
                        or model_bridge_req.shape != (Y.shape[0], 3)
                        or model_bridge_snp.shape != (Y.shape[0], 3)
                        or not np.all(np.isfinite(model_bridge_req))
                        or not np.all(np.isfinite(model_bridge_snp))
                        or not np.isclose(force_n, PRODUCTION_CONTRACT["force_n"], rtol=0.0)
                        or not np.isclose(alpha, PRODUCTION_CONTRACT["rayleigh_alpha"],
                                          rtol=0.0)
                        or not np.isclose(beta, PRODUCTION_CONTRACT["rayleigh_beta"],
                                          rtol=0.0)
                        or str(np.asarray(model["time_convention"]).item())
                           != PRODUCTION_CONTRACT["time_convention"]
                        or str(np.asarray(model["response_units"]).item()) != "m/s/N"):
                    return False
        if body == "hollow":
            reconstruction_error = _compact_reconstruction_error(
                case_dir / model_name, body, freqs, Y, p_bar, U_h)
            if (not np.isfinite(reconstruction_error)
                    or reconstruction_error > QC_MODEL_RECONSTRUCTION_TOL):
                return False
        timing = json.loads((case_dir / "timing.json").read_text(encoding="utf-8"))
        case_status = json.loads(
            (case_dir / "case_run_status.json").read_text(encoding="utf-8"))
        worker = json.loads(
            (case_dir / "worker_result.json").read_text(encoding="utf-8"))
        if (case_status.get("status") != "complete"
                or not isinstance(timing, dict)
                or not _numeric_tree_is_finite(timing, nonnegative=True)
                or timing != meta.get("timing", timing)):
            return False

        multi_name = ("admittance_air_modal_coupled_multi.npz"
                      if body == "hollow" else "admittance_harmonic_multi.npz")
        if (worker.get("run_status") != "complete"
                or worker.get("solver_revision") != want_rev
                or worker.get("multi_npz") != multi_name
                or worker.get("peak_npz") != "peak_labels.npz"
                or worker.get("case_input_hash") != meta.get("case_input_hash")
                or not _numeric_tree_is_finite(worker.get("diagnostics", {}))
                or not _numeric_tree_is_finite(worker.get("timing", {}),
                                               nonnegative=True)
                or worker.get("timing", {}) != timing):
            return False
        canonical_diag = dict(meta.get("diagnostics", {}))
        canonical_diag.pop("conformal_interface_verified", None)
        worker_diag = dict(worker.get("diagnostics", {}))
        worker_diag.pop("conformal_interface_verified", None)
        if (worker_diag != canonical_diag
                or worker.get("bridges", []) != meta.get("bridges", [])):
            return False

        bridges = meta.get("bridges", [])
        req = np.asarray([b["bridge_requested_xyz"] for b in bridges], float)
        snp = np.asarray([b["bridge_snapped_xyz"] for b in bridges], float)
        snap = np.asarray([b["snap_distance_mm"] for b in bridges], float)
        if body == "solid" and (not _array_matches(model_bridge_req, req)
                                or not _array_matches(model_bridge_snp, snp)):
            return False
        with np.load(case_dir / multi_name) as native:
            if (not _grid_ok(native["frequencies"], freqs)
                    or not _array_matches(native["admittance"], Y)):
                return False
            if body == "hollow":
                if (not _array_matches(native["p_bar"], p_bar)
                        or not _array_matches(native["U_h"], U_h)
                        or not _array_matches(native["bridge_requested_xyz"], req)
                        or not _array_matches(native["bridge_snapped_xyz"], snp)
                        or not _array_matches(native["snap_distance_mm"], snap)):
                    return False
            elif (not _array_matches(native["bridge_requested"], req)
                  or not _array_matches(native["bridge_snapped"], snp)
                  or not _array_matches(native["snap_distance_mm"], snap)):
                return False

        if body == "hollow":
            with np.load(case_dir / "admittance_air_modal_coupled.npz") as single:
                if (not _grid_ok(single["frequencies"], freqs)
                        or not _array_matches(single["admittance"], Y[0])):
                    return False
            with np.load(case_dir / "pressure_mean_cavity.npz") as single:
                if (not _grid_ok(single["frequencies"], freqs)
                        or not _array_matches(single["p_bar"], p_bar[0])):
                    return False
            with np.load(case_dir / "soundhole_volume_velocity.npz") as single:
                if (not _grid_ok(single["frequencies"], freqs)
                        or not _array_matches(single["U_h"], U_h[0])):
                    return False
            native_meta = json.loads(
                (case_dir / "modal_coupled_metadata.json").read_text(encoding="utf-8"))
            native_status = json.loads(
                (case_dir / "run_status.json").read_text(encoding="utf-8"))
            a0 = json.loads(
                (case_dir / "A0_estimated_vs_observed.json").read_text(encoding="utf-8"))
            canonical_a0 = json.loads(
                (case_dir / "A0.json").read_text(encoding="utf-8"))
            observed = canonical_a0.get("A0_observed_hz")
            if observed is None:
                observed_ok = (canonical_a0.get("A0_detected") is False
                               and a0.get("A0_detected") is False
                               and a0.get("A0_observed_hz") is None)
            else:
                try:
                    observed_ok = (
                        canonical_a0.get("A0_detected") is True
                        and a0.get("A0_detected") is True
                        and np.isfinite(float(observed))
                        and np.isclose(float(a0.get("A0_observed_hz")),
                                       float(observed), rtol=1e-12))
                except (TypeError, ValueError):
                    observed_ok = False
            if (native_status.get("status") != "complete"
                    or native_status.get("solver_revision") != want_rev
                    or _canonical_json(canonical_a0)
                       != _canonical_json(worker.get("A0", {}))
                    or not observed_ok
                    or not np.isclose(
                        float(a0.get("A0_estimated_hz", float("nan"))),
                        float(canonical_a0.get("A0_estimated_hz", float("nan"))),
                        rtol=1e-12)
                    or not np.isclose(float(a0.get("cavity_volume_m3", float("nan"))),
                                      float(meta["diagnostics"]["cavity_volume_m3"]),
                                      rtol=1e-12)
                    or not np.isclose(float(a0.get("soundhole_area_m2", float("nan"))),
                                      float(meta["diagnostics"]["soundhole_area_m2"]),
                                      rtol=1e-12)
                    or int(a0.get("n_freq_points", -1)) != int(freqs.size)
                    or float(a0.get("freq_min_hz", float("nan"))) != float(freqs[0])
                    or float(a0.get("freq_max_hz", float("nan"))) != float(freqs[-1])
                    or native_meta.get("solver_revision") != want_rev
                    or native_meta.get("structural_basis") != "craig-bampton"
                    or int(native_meta.get("n_bridges", -1)) != int(Y.shape[0])
                    or float(native_meta.get("analysis_freq_min_hz", float("nan")))
                       != float(freqs[0])
                    or float(native_meta.get("analysis_freq_max_hz", float("nan")))
                       != float(freqs[-1])
                    or int(native_meta.get("analysis_freq_points", -1)) != int(freqs.size)
                    or not _numeric_tree_is_finite(native_meta)
                    or not _numeric_tree_is_finite({
                        k: a0.get(k) for k in (
                            "A0_estimated_hz", "cavity_volume_m3",
                            "soundhole_area_m2", "n_freq_points",
                            "freq_min_hz", "freq_max_hz") if k in a0
                    })):
                return False
        else:
            with np.load(case_dir / "admittance.npz") as single:
                if (not _grid_ok(single["frequencies"], freqs)
                        or not _array_matches(single["admittance"], Y[0])):
                    return False
            native_meta = json.loads(
                (case_dir / "admittance_meta.json").read_text(encoding="utf-8"))
            coverage = native_meta.get("coverage", {})
            if (not _numeric_tree_is_finite(native_meta)
                    or coverage.get("solver_revision") != want_rev
                    or coverage.get("coverage_ok") is not True
                    or float(coverage.get("freq_min", float("nan"))) != float(freqs[0])
                    or float(coverage.get("freq_max", float("nan"))) != float(freqs[-1])
                    or int(coverage.get("freq_points", -1)) != int(freqs.size)):
                return False
        return True
    except Exception:
        return False


def _load_committed_geometry(output_dir: Path, case: dict, plan: dict) -> dict | None:
    shape_dir = output_dir / "shapes" / _shape_dir_name(case["shape_id"])
    try:
        if not _is_committed(shape_dir):
            return None
        geom = json.loads((shape_dir / "geometry.json").read_text())
        shape_meta = next(s for s in plan["_base_shapes"]
                          if int(s["base_shape_id"]) == int(case["base_shape_id"]))
        expected_digest = geometry_digest(shape_meta, case["body_type"])
        hashes = _geometry_artifact_hashes(shape_dir, geom)
        if (geom.get("plan_hash") != plan["plan_hash"]
                or geom.get("geometry_digest") != expected_digest
                or geom.get("body_type") != case["body_type"]
                or int(geom.get("shape_id", -1)) != int(case["shape_id"])
                or hashes != geom.get("artifact_sha256", {})
                or geom.get("mesh_sha256") != hashes.get("rel_mesh")
                or not _geometry_metadata_matches_plan(
                    shape_dir, geom, shape_meta, case["body_type"])):
            return None
        return _resolve_geom_paths(geom, shape_dir)
    except Exception:
        return None


def validate_case_complete(output_dir: Path, case: dict, plan: dict,
                           freqs: np.ndarray, status: dict | None,
                           geometry_cache: dict | None = None) -> bool:
    """Return True only if the case is FULLY and correctly on disk: status done +
    plan_hash + solver revision + committed case artifact (grid/shape/finite) +
    metadata bridge count + every bridge sample (grid/finite + params)."""
    if (not status or status.get("status") != "done"
            or not _numeric_tree_is_finite(status)):
        return False
    ph = plan["plan_hash"]
    body = case["body_type"]
    want_rev = HOLLOW_SOLVER_REVISION if body == "hollow" else SOLID_SOLVER_REVISION
    n_b = case["n_bridges"]
    case_dir = output_dir / "cases" / case["case_id"]
    try:
        cache_key = (int(case["shape_id"]), plan["plan_hash"])
        if geometry_cache is not None and cache_key in geometry_cache:
            geom = geometry_cache[cache_key]
        else:
            geom = _load_committed_geometry(output_dir, case, plan)
            if geometry_cache is not None:
                geometry_cache[cache_key] = geom
        if geom is None:
            return False
        if (status.get("case_id") != case["case_id"]
                or status.get("plan_hash") != ph
                or status.get("geometry_digest") != geom.get("geometry_digest")
                or status.get("mesh_sha256") != geom.get("mesh_sha256")):
            return False
        if not _is_committed(case_dir):
            return False
        meta = json.loads((case_dir / "case_meta.json").read_text())
        transaction_id = status.get("transaction_id")
        if (meta.get("plan_hash") != ph or meta.get("solver_revision") != want_rev
                or meta.get("case_id") != case["case_id"]
                or meta.get("body_type") != body
                or meta.get("material_id") != case["material_id"]
                or meta.get("run_status") != "complete"
                or meta.get("geometry_digest") != geom.get("geometry_digest")
                or meta.get("mesh_sha256") != geom.get("mesh_sha256")
                or not isinstance(transaction_id, str) or len(transaction_id) != 32
                or meta.get("transaction_id") != transaction_id
                or int(meta.get("attempt", -1)) != int(status.get("attempt", -2))
                or meta.get("production_contract") != PRODUCTION_CONTRACT):
            return False
        native_model_name = (
            "reduced_model.npz" if body == "hollow" else "solid_full_eigen.npz")
        if (meta.get("schema_version") != DATASET_SCHEMA_VERSION
                or meta.get("output_schema") != PRODUCTION_CONTRACT["output_schema"]
                or meta.get("nn_response") != "case_response.npz"
                or meta.get("units") != PRODUCTION_CONTRACT["response_units"]
                or meta.get("coordinate_units") != "mm"
                or meta.get("native_model") != native_model_name
                or meta.get("native_model_sha256")
                   != _file_sha256(case_dir / native_model_name)):
            return False
        if int(meta.get("n_bridges", -1)) != n_b or len(meta.get("bridges", [])) != n_b:
            return False
        required = ["case_admittance.npz", "case_response.npz", "peak_labels.npz",
                    "case_meta.json", "timing.json",
                    "solve_spec.json",
                    "case_run_status.json", "worker_result.json", "worker.log"]
        required += (["admittance_air_modal_coupled_multi.npz",
                      "admittance_air_modal_coupled.npz",
                      "reduced_model.npz",
                      "modal_coupled_metadata.json", "A0_estimated_vs_observed.json",
                      "pressure_mean_cavity.npz", "soundhole_volume_velocity.npz",
                      "peak_search_response.npz", "run_status.json", "A0.json"]
                     if body == "hollow" else
                     ["admittance_harmonic_multi.npz", "admittance.npz",
                      "admittance_meta.json", "solid_full_eigen.npz"])
        if any(not (case_dir / name).is_file() for name in required):
            return False
        shape_meta = next(s for s in plan["_base_shapes"]
                          if int(s["base_shape_id"]) == int(case["base_shape_id"]))
        material = next(m for m in plan["_materials"]
                        if m["material_id"] == case["material_id"])
        budget = meta.get("mode_budget") or initial_mode_budget(
            case, plan["plan_body"]["config"])
        expected_input_hash = case_input_hash(case, material, geom, freqs, budget)
        if (status.get("case_input_hash") != expected_input_hash
                or meta.get("case_input_hash") != expected_input_hash
                or meta.get("material") != material
                or meta.get("base_shape_id") != case["base_shape_id"]
                or meta.get("shape_id") != case["shape_id"]
                or meta.get("split") != case["split"]
                or meta.get("n_modes_used") != _budget_n_modes(budget, body)):
            return False
        spec = json.loads((case_dir / "solve_spec.json").read_text(encoding="utf-8"))
        spec_material = {k: v for k, v in material.items()
                         if isinstance(v, (int, float, str))}
        budget_key = "hollow" if body == "hollow" else "solid"
        expected_solver_cfg = dict(budget)
        if body == "hollow":
            expected_solver_cfg["t_hole_mm"] = geom.get("t_hole_mm")
        expected_mesh_rel = (Path("../..") / "shapes" /
                             _shape_dir_name(case["shape_id"]) /
                             geom["rel_mesh"]).as_posix()
        if (spec.get("body_type") != body
                or spec.get("material") != spec_material
                or spec.get("production_contract") != PRODUCTION_CONTRACT
                or spec.get("plan_hash") != ph
                or spec.get("geometry_digest") != geom.get("geometry_digest")
                or spec.get("mesh_sha256") != geom.get("mesh_sha256")
                or spec.get("case_input_hash") != expected_input_hash
                or spec.get(budget_key) != expected_solver_cfg
                or not _grid_ok(spec.get("freqs", []), freqs)
                or not _array_matches(spec.get("bridge_coords", []),
                                      geom.get("bridge_coords", []))
                or spec.get("msh_path") != expected_mesh_rel
                or (case_dir / spec.get("msh_path", "")).resolve()
                   != Path(geom["msh_path"]).resolve()
                or spec.get("out_dir") != "."):
            return False
        with np.load(case_dir / "case_admittance.npz") as d:
            if not _grid_ok(d["frequencies"], freqs):
                return False
            Y = np.asarray(d["admittance"], complex).copy()
            p_bar = U_h = None
            if body == "hollow":
                p_bar = np.asarray(d["p_bar"], complex).copy()
                U_h = np.asarray(d["U_h"], complex).copy()
        with np.load(case_dir / "case_response.npz", allow_pickle=False) as d:
            if (str(np.asarray(d["schema_version"]).item())
                    != PRODUCTION_CONTRACT["output_schema"]
                    or not _grid_ok(d["frequencies_hz"], freqs)):
                return False
            log_mag = np.asarray(d["log_magnitude_db"], float).copy()
            peaks = {key: np.asarray(d[key]).copy() for key in (
                "peak_frequency_hz", "peak_amplitude_db", "peak_q", "peak_mask",
                "peak_count_total", "peak_truncated")}
        from peak_labels import magnitude_db, validate_peak_batch
        if (not _array_matches(log_mag, magnitude_db(Y))
                or validate_peak_batch(
                    peaks, n_b, int(PRODUCTION_CONTRACT["top_k_peaks"]),
                    float(freqs[0]), float(freqs[-1]),
                    min_count=int(
                        PRODUCTION_CONTRACT["min_peak_count_per_bridge"]))):
            return False
        with np.load(case_dir / "peak_labels.npz", allow_pickle=False) as native_peaks:
            if any(not _array_matches(native_peaks[key], value)
                   for key, value in peaks.items()):
                return False
        timing = json.loads((case_dir / "timing.json").read_text(encoding="utf-8"))
        response = {
            "run_status": "complete", "solver_revision": want_rev,
            "frequencies": np.asarray(freqs, float), "admittance": Y,
            "bridges": meta.get("bridges", []),
            "diagnostics": meta.get("diagnostics", {}),
            "timing": timing,
            "case_input_hash": meta.get("case_input_hash"),
            "artifact_dir": str(case_dir),
            "peaks": peaks,
        }
        if body == "hollow":
            response.update(
                p_bar=p_bar,
                U_h=U_h,
                A0=json.loads((case_dir / "A0.json").read_text(encoding="utf-8")),
            )
        ok, _reasons = qc_case(case, response, freqs, expected_geom=geom,
                               expected_budget=budget,
                               expected_input_hash=expected_input_hash)
        if not ok:
            return False
        if not _validate_native_case_artifacts(
                case_dir, body, freqs, Y, p_bar, U_h, meta, want_rev):
            return False
        samples_root = output_dir / "samples"
        expected_ids = [f"{case['case_id']}_b{bi}" for bi in range(n_b)]
        rows = status.get("rows")
        if (not isinstance(rows, list) or len(rows) != n_b
                or [r.get("sample_id") for r in rows] != expected_ids):
            return False
        attempts_log = status.get("attempts_log")
        if (not isinstance(attempts_log, list)
                or len(attempts_log) != int(status.get("attempt", -1))
                or not attempts_log or attempts_log[-1].get("ok") is not True
                or attempts_log[-1].get("mode_budget") != budget
                or attempts_log[-1].get("case_input_hash") != expected_input_hash):
            return False
        expected_attempt_budget = initial_mode_budget(case, plan["plan_body"]["config"])
        for ai, logged in enumerate(attempts_log):
            if (logged.get("attempt") != ai + 1
                    or logged.get("mode_budget") != expected_attempt_budget
                    or logged.get("case_input_hash") != case_input_hash(
                        case, material, geom, freqs, expected_attempt_budget)
                    or (ai < len(attempts_log) - 1 and logged.get("ok") is not False)):
                return False
            expected_attempt_budget = escalate_mode_budget(expected_attempt_budget, body)
        expected_base = _base_row(
            case, shape_meta, material, response, geom, ph, 0.0, budget,
            int(status["attempt"]))
        for bi in range(n_b):
            sample_id = expected_ids[bi]
            sd = samples_root / sample_id
            if not _is_committed(sd):
                return False
            with np.load(sd / "admittance.npz") as ds:
                if not _grid_ok(ds["frequencies"], freqs):
                    return False
                ys = np.asarray(ds["admittance"], complex).copy()
            if ys.shape != (freqs.size,) or not np.all(np.isfinite(ys)):
                return False
            if not np.array_equal(ys, Y[bi]):
                return False
            p = json.loads((sd / "sample_params.json").read_text())
            bridge = meta["bridges"][bi]
            if (p.get("sample_id") != sample_id or p.get("case_id") != case["case_id"]
                    or p.get("schema_version") != DATASET_SCHEMA_VERSION
                    or p.get("units", {}).get("coordinates") != "mm"
                    or p.get("units", {}).get("material_stiffness") != "Pa"
                    or p.get("units", {}).get("density") != "kg/m^3"
                    or {k: p.get("units", {}).get(k)
                        for k in PRODUCTION_CONTRACT["response_units"]}
                       != PRODUCTION_CONTRACT["response_units"]
                    or p.get("coordinate_frame", {}).get("material_axes")
                       != PRODUCTION_CONTRACT["material_axes_fem"]
                    or p.get("coordinate_frame", {}).get("time_convention")
                       != PRODUCTION_CONTRACT["time_convention"]
                    or p.get("sampled_response_role")
                       != PRODUCTION_CONTRACT["sampled_response_role"]
                    or p.get("rel_case_model")
                       != (f"cases/{case['case_id']}/"
                           + ("reduced_model.npz" if body == "hollow"
                              else "solid_full_eigen.npz"))
                    or p.get("rel_case_response")
                       != f"cases/{case['case_id']}/case_response.npz"
                    or p.get("plan_hash") != ph or int(p.get("bridge_idx", -1)) != bi
                    or p.get("material_id") != case["material_id"]
                    or p.get("body_type") != body
                    or p.get("geometry_digest") != geom.get("geometry_digest")
                    or p.get("mesh_sha256") != geom.get("mesh_sha256")
                    or p.get("solver_revision") != want_rev
                    or p.get("transaction_id") != transaction_id
                    or p.get("case_input_hash") != expected_input_hash
                    or int(p.get("attempt", -1)) != int(status.get("attempt", -2))
                    or p.get("mode_budget") != budget
                    or p.get("material") != material
                    or p.get("material_columns") != material_columns(material)
                    or p.get("base_shape_id") != case["base_shape_id"]
                    or p.get("shape_id") != case["shape_id"]
                    or p.get("split") != case["split"]
                    or p.get("shape_type") != shape_meta["shape_type"]
                    or p.get("guitar_model") != shape_meta["guitar_model"]
                    or not _array_matches(p.get("bridge_requested_xyz", []),
                                          bridge["bridge_requested_xyz"])
                    or not _array_matches(p.get("bridge_snapped_xyz", []),
                                          bridge["bridge_snapped_xyz"])
                    or float(p.get("snap_distance_mm", float("inf")))
                       != float(bridge["snap_distance_mm"])
                    or p.get("diagnostics") != meta.get("diagnostics", {})
                    or not _numeric_tree_is_finite(p)):
                return False
            row = rows[bi]
            if any(row.get(key) != value for key, value in expected_base.items()
                   if key != "solve_time_s"):
                return False
            expected_bridge_row = {
                "sample_id": sample_id, "bridge_idx": bi,
                "rel_case_response": f"cases/{case['case_id']}/case_response.npz",
                "rel_admittance": f"samples/{sample_id}/admittance.npz",
                "rel_params": f"samples/{sample_id}/sample_params.json",
                "bridge_req_x": float(bridge["bridge_requested_xyz"][0]),
                "bridge_req_y": float(bridge["bridge_requested_xyz"][1]),
                "bridge_req_z": float(bridge["bridge_requested_xyz"][2]),
                "bridge_snap_x": float(bridge["bridge_snapped_xyz"][0]),
                "bridge_snap_y": float(bridge["bridge_snapped_xyz"][1]),
                "bridge_snap_z": float(bridge["bridge_snapped_xyz"][2]),
                "snap_distance_mm": round(float(bridge["snap_distance_mm"]), 4),
                "transaction_id": transaction_id,
                "case_input_hash": expected_input_hash,
            }
            if any(row.get(key) != value for key, value in expected_bridge_row.items()):
                return False
            if (row.get("case_id") != case["case_id"] or row.get("plan_hash") != ph
                    or row.get("status") != "done"
                    or row.get("geometry_digest") != geom.get("geometry_digest")
                    or row.get("mesh_sha256") != geom.get("mesh_sha256")
                    or row.get("transaction_id") != transaction_id
                    or row.get("case_input_hash") != expected_input_hash
                    or row.get("body_type") != body
                    or row.get("material_id") != case["material_id"]
                    or int(row.get("bridge_idx", -1)) != bi
                    or row.get("solver_revision") != want_rev
                    or int(row.get("attempt", -1)) != int(status.get("attempt", -2))
                    or row.get("rel_case_response")
                       != f"cases/{case['case_id']}/case_response.npz"
                    or row.get("rel_admittance") != f"samples/{sample_id}/admittance.npz"
                    or row.get("rel_params") != f"samples/{sample_id}/sample_params.json"
                    or not _numeric_tree_is_finite(row)):
                return False
    except Exception:
        return False
    return True


# ===========================================================================
# Manifest (regenerated deterministically from committed case statuses)
# ===========================================================================

def _valid_terminal_failure_status(status: dict, case: dict, plan_hash_value: str) -> bool:
    """Return True only for a well-formed, auditable terminal case failure."""
    if not isinstance(status, dict):
        return False
    attempt = status.get("attempt")
    reasons = status.get("qc_reason")
    attempts_log = status.get("attempts_log")
    if (status.get("case_id") != case.get("case_id")
            or status.get("plan_hash") != plan_hash_value
            or status.get("status") != "failed"
            or isinstance(attempt, bool) or not isinstance(attempt, int)
            or attempt < 0
            or not isinstance(reasons, list) or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or not isinstance(attempts_log, list)
            or status.get("rows") not in (None, [])):
        return False
    if attempt > 0 and not attempts_log:
        return False
    for entry in attempts_log:
        if (not isinstance(entry, dict)
                or isinstance(entry.get("attempt"), bool)
                or not isinstance(entry.get("attempt"), int)
                or entry["attempt"] < 1):
            return False
    return True


def _manifest_rows(case_status: dict, plan: dict, only_cases=None,
                   include_non_done: bool = True) -> list[dict]:
    rows = []
    for c in plan["_cases"]:
        if only_cases is not None and c["case_id"] not in only_cases:
            continue
        st = case_status.get(c["case_id"])
        if st and st.get("status") == "done" and st.get("rows"):
            rows.extend(st["rows"])
        elif include_non_done:
            rows.append({
                "sample_id": "", "case_id": c["case_id"],
                "base_shape_id": c["base_shape_id"], "shape_id": c["shape_id"],
                "body_type": c["body_type"], "material_id": c["material_id"],
                "split": c["split"], "solver_backend": _solver_backend(c["body_type"]),
                "plan_hash": plan["plan_hash"], "attempt": (st or {}).get("attempt", 0),
                "status": (st or {}).get("status", "pending"),
                "qc_reason": ";".join((st or {}).get("qc_reason", []))[:300],
            })
    return rows


def _write_manifest(path: Path, rows: list[dict]):
    import io
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=MANIFEST_COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    _atomic_write_bytes(path, sio.getvalue().encode())


def rebuild_manifest(output_dir: Path, case_status: dict, plan: dict):
    """Regenerate the training manifest from committed successful cases only."""
    _write_manifest(output_dir / "manifest.csv", _manifest_rows(
        case_status, plan, include_non_done=False))


def _invalidate_dataset_ready(output_dir: Path, quarantine_manifest: bool = False):
    """Remove the completion certificate before any mutation or failed merge."""
    output_dir = Path(output_dir)
    (output_dir / DATASET_READY_FILENAME).unlink(missing_ok=True)
    (output_dir / DATASET_INVENTORY_FILENAME).unlink(missing_ok=True)
    # A previous successful summary is no longer authoritative once generation or
    # revalidation starts.  The next merge writes a fresh one.
    (output_dir / "merge_summary.json").unlink(missing_ok=True)
    if quarantine_manifest:
        manifest = output_dir / "manifest.csv"
        if manifest.exists():
            os.replace(manifest, output_dir / "manifest.invalid.csv")


def _artifact_inventory_paths(output_dir: Path, plan: dict,
                              included_case_ids: set[str] | None = None,
                              failed_case_ids: set[str] | None = None) -> list[Path]:
    """Return every committed artifact certified by dataset readiness.

    The inventory deliberately covers the persisted geometry, canonical/native
    case output, per-bridge samples, and the status that certifies each transaction.
    Lock files, failed attempts, shard manifests, and run logs outside committed
    directories are operational state rather than dataset content.
    """
    output_dir = Path(output_dir).resolve()
    included_case_ids = (set(included_case_ids) if included_case_ids is not None
                         else {str(c["case_id"]) for c in plan["_cases"]})
    failed_case_ids = set(failed_case_ids or ())
    known_case_ids = {str(c["case_id"]) for c in plan["_cases"]}
    if ((included_case_ids | failed_case_ids) - known_case_ids
            or included_case_ids & failed_case_ids):
        raise RuntimeError("artifact inventory case selection is invalid")

    roots: list[Path] = []
    shape_ids = sorted({int(c["shape_id"]) for c in plan["_cases"]
                        if str(c["case_id"]) in included_case_ids})
    for sid in shape_ids:
        roots.append(output_dir / "shapes" / _shape_dir_name(sid))
    for case in plan["_cases"]:
        cid = str(case["case_id"])
        if cid in failed_case_ids:
            roots.append(output_dir / "case_status" / f"{cid}.json")
            continue
        if cid not in included_case_ids:
            continue
        roots.append(output_dir / "cases" / cid)
        roots.append(output_dir / "case_status" / f"{cid}.json")
        for bi in range(int(case["n_bridges"])):
            roots.append(output_dir / "samples" / f"{cid}_b{bi}")

    files: set[Path] = set()
    for root in roots:
        root = root.resolve()
        try:
            root.relative_to(output_dir)
        except ValueError as exc:
            raise RuntimeError(f"artifact path escapes dataset root: {root}") from exc
        if root.is_symlink():
            raise RuntimeError(f"dataset artifact must not be a symlink: {root}")
        if root.is_file():
            files.add(root)
            continue
        if not root.is_dir():
            raise RuntimeError(f"required committed artifact is missing: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"dataset artifact must not be a symlink: {path}")
            if path.is_file():
                files.add(path.resolve())
    return sorted(files, key=lambda p: p.relative_to(output_dir).as_posix())


def _write_artifact_inventory(output_dir: Path, plan: dict,
                              included_case_ids: set[str] | None = None,
                              failed_case_ids: set[str] | None = None) -> dict:
    output_dir = Path(output_dir).resolve()
    entries = []
    for path in _artifact_inventory_paths(
            output_dir, plan, included_case_ids=included_case_ids,
            failed_case_ids=failed_case_ids):
        entries.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _file_sha256(path),
        })
    payload = {
        "schema_version": "mixed-artifact-inventory-v1",
        "plan_hash": plan["plan_hash"],
        "n_entries": len(entries),
        "entries": entries,
    }
    inventory_path = output_dir / DATASET_INVENTORY_FILENAME
    _atomic_write_bytes(inventory_path, json.dumps(
        payload, indent=2, sort_keys=True).encode())
    return {"sha256": _file_sha256(inventory_path), "n_entries": len(entries)}


def _validate_artifact_inventory(output_dir: Path, plan: dict, ready: dict):
    output_dir = Path(output_dir).resolve()
    inventory_path = output_dir / DATASET_INVENTORY_FILENAME
    if not inventory_path.is_file():
        raise RuntimeError("mixed dataset artifact inventory is missing")
    if ready.get("inventory_sha256") != _file_sha256(inventory_path):
        raise RuntimeError("mixed dataset artifact inventory hash is invalid")
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("mixed dataset artifact inventory is malformed") from exc
    entries = payload.get("entries")
    if (payload.get("schema_version") != "mixed-artifact-inventory-v1"
            or payload.get("plan_hash") != plan["plan_hash"]
            or not isinstance(entries, list)
            or int(payload.get("n_entries", -1)) != len(entries)
            or int(ready.get("n_inventory_entries", -1)) != len(entries)):
        raise RuntimeError("mixed dataset artifact inventory metadata is invalid")

    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("mixed dataset artifact inventory entry is malformed")
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or entry["path"] in seen:
            raise RuntimeError("mixed dataset artifact inventory path is invalid")
        seen.add(entry["path"])
        path = (output_dir / rel).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as exc:
            raise RuntimeError("mixed dataset artifact inventory escapes root") from exc
        if (path.is_symlink() or not path.is_file()
                or int(entry.get("size_bytes", -1)) != int(path.stat().st_size)
                or entry.get("sha256") != _file_sha256(path)):
            raise RuntimeError(
                f"mixed dataset artifact inventory mismatch: {entry['path']}")


def _publish_dataset_ready(output_dir: Path, plan: dict, summary: dict,
                           included_case_ids: set[str],
                           failed_case_ids: set[str]) -> dict:
    """Publish the completion certificate LAST, after the canonical manifest."""
    output_dir = Path(output_dir)
    manifest = output_dir / "manifest.csv"
    if not summary.get("complete") or not manifest.is_file():
        raise RuntimeError("cannot certify an incomplete dataset")
    manifest_sha = _file_sha256(manifest)
    with open(manifest, newline="", encoding="utf-8") as f:
        row_count = sum(1 for _ in csv.DictReader(f))
    if row_count != int(summary["n_expected_samples"]):
        raise RuntimeError(
            f"manifest row count {row_count} != expected {summary['n_expected_samples']}")
    inventory = _write_artifact_inventory(
        output_dir, plan, included_case_ids=included_case_ids,
        failed_case_ids=failed_case_ids)
    ready = {
        "status": "complete", "generator_version": GENERATOR_VERSION,
        "completion_policy": COMPLETION_POLICY,
        "plan_hash": plan["plan_hash"], "manifest_sha256": manifest_sha,
        "n_planned_cases": int(summary["n_planned_cases"]),
        "n_cases": int(summary["n_done"]),
        "n_failed_cases": int(summary["n_failed"]),
        "n_planned_samples": int(summary["n_planned_samples"]),
        "n_samples": row_count,
        "inventory_sha256": inventory["sha256"],
        "n_inventory_entries": inventory["n_entries"],
        "published_at_unix_s": float(time.time()),
    }
    _atomic_write_bytes(output_dir / DATASET_READY_FILENAME,
                        json.dumps(ready, indent=2).encode())
    return ready


def validate_dataset_ready(output_dir: Path) -> dict | None:
    """Validate a mixed dataset certificate without importing the NN stack."""
    output_dir = Path(output_dir)
    plan_path = output_dir / "dataset_plan.json"
    if not plan_path.exists():
        manifest_path = output_dir / "manifest.csv"
        mixed_trace = ((output_dir / DATASET_READY_FILENAME).exists()
                       or (output_dir / DATASET_INVENTORY_FILENAME).exists()
                       or (output_dir / "case_status").exists()
                       or (output_dir / "cases").exists())
        if manifest_path.is_file():
            try:
                with open(manifest_path, newline="", encoding="utf-8") as f:
                    columns = set(next(csv.reader(f), []))
                mixed_trace = mixed_trace or bool(
                    {"transaction_id", "case_input_hash", "plan_hash"} & columns)
            except Exception:
                mixed_trace = True
        if mixed_trace:
            raise RuntimeError("mixed dataset plan is missing")
        return None
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("mixed dataset plan is malformed") from exc
    config = plan.get("config", {})
    if (plan.get("generator_version") != GENERATOR_VERSION
            or config.get("generator_version") != GENERATOR_VERSION):
        raise RuntimeError("mixed dataset plan generator version is invalid")
    manifest = output_dir / "manifest.csv"
    ready_path = output_dir / DATASET_READY_FILENAME
    if not ready_path.exists():
        raise RuntimeError(
            "mixed dataset is not certified complete (dataset_ready.json missing)")
    if not manifest.is_file():
        raise RuntimeError("mixed dataset readiness certificate has no manifest.csv")
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("mixed dataset readiness certificate is malformed") from exc
    recorded_plan_hash = plan.get("plan_hash")
    if recorded_plan_hash != _stored_plan_hash(plan):
        raise RuntimeError("mixed dataset plan body is corrupt")
    cases = plan.get("cases", [])
    expected_cases = len(cases)
    planned_samples = sum(int(c.get("n_bridges", 0)) for c in cases)
    with open(manifest, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    row_count = len(manifest_rows)
    common_valid = (
        ready.get("status") == "complete"
        and ready.get("generator_version") == GENERATOR_VERSION
        and ready.get("plan_hash") == recorded_plan_hash
        and ready.get("manifest_sha256") == _file_sha256(manifest)
    )
    policy = ready.get("completion_policy")
    if policy is None:
        # Backward-compatible validation for already-certified all-success v4 data.
        valid = (common_valid
                 and int(ready.get("n_cases", -1)) == expected_cases
                 and int(ready.get("n_samples", -1)) == planned_samples
                 and row_count == planned_samples)
    elif policy == COMPLETION_POLICY:
        case_by_id = {str(c.get("case_id")): c for c in cases}
        if len(case_by_id) != expected_cases:
            raise RuntimeError("mixed dataset plan contains duplicate case IDs")
        statuses = _load_case_status(output_dir / "case_status")
        if set(statuses) != set(case_by_id):
            raise RuntimeError("mixed dataset terminal case-status set is invalid")

        done_case_ids: set[str] = set()
        failed_case_ids: set[str] = set()
        expected_sample_ids: set[str] = set()
        for cid, case in case_by_id.items():
            status = statuses[cid]
            if status.get("status") == "done":
                rows = status.get("rows")
                n_bridges = int(case.get("n_bridges", 0))
                ids = ([row.get("sample_id") for row in rows]
                       if isinstance(rows, list) else [])
                expected_ids = {f"{cid}_b{bi}" for bi in range(n_bridges)}
                if (status.get("case_id") != cid
                        or status.get("plan_hash") != recorded_plan_hash
                        or len(ids) != n_bridges or set(ids) != expected_ids
                        or any(not isinstance(row, dict)
                               or row.get("case_id") != cid
                               or row.get("plan_hash") != recorded_plan_hash
                               or row.get("status") != "done" for row in rows)):
                    raise RuntimeError("mixed dataset completed case status is invalid")
                done_case_ids.add(cid)
                expected_sample_ids.update(expected_ids)
            elif _valid_terminal_failure_status(status, case, recorded_plan_hash):
                failed_case_ids.add(cid)
            else:
                raise RuntimeError("mixed dataset terminal failure status is invalid")

        manifest_ids = [row.get("sample_id") for row in manifest_rows]
        manifest_consistent = (
            len(manifest_ids) == len(expected_sample_ids)
            and set(manifest_ids) == expected_sample_ids
            and all(row.get("case_id") in done_case_ids
                    and row.get("status") == "done"
                    and row.get("plan_hash") == recorded_plan_hash
                    for row in manifest_rows)
        )
        valid = (
            common_valid and manifest_consistent
            and int(ready.get("n_planned_cases", -1)) == expected_cases
            and int(ready.get("n_cases", -1)) == len(done_case_ids)
            and int(ready.get("n_failed_cases", -1)) == len(failed_case_ids)
            and len(done_case_ids) + len(failed_case_ids) == expected_cases
            and int(ready.get("n_planned_samples", -1)) == planned_samples
            and int(ready.get("n_samples", -1)) == len(expected_sample_ids)
            and row_count == len(expected_sample_ids)
        )
    else:
        valid = False
    if not valid:
        raise RuntimeError("mixed dataset readiness certificate is invalid")
    _validate_artifact_inventory(output_dir, {
        "plan_hash": recorded_plan_hash,
    }, ready)
    return ready


def _load_case_status(status_root: Path, *, strict: bool = True) -> dict:
    out = {}
    for sp in status_root.glob("*.json"):
        try:
            st = json.loads(sp.read_text())
            cid = st["case_id"]
            if not isinstance(cid, str) or not cid or sp.stem != cid:
                raise ValueError("status filename does not match case_id")
            if cid in out:
                raise ValueError("duplicate case_id in status files")
            out[cid] = st
        except Exception as exc:
            if strict:
                raise RuntimeError(f"invalid case status file: {sp}") from exc
    return out


def merge_shards(output_dir: Path, config: dict | None = None) -> dict:
    """Merge step: reload ALL committed case statuses and regenerate the single
    canonical manifest.csv, checking plan_hash consistency and duplicate/missing
    samples.  Safe to run after parallel shards finish (each shard only wrote its
    own disjoint case_status/*.json + manifest_shard*.csv)."""
    output_dir = Path(output_dir).resolve()
    publish_lock = _AdvisoryFileLock(output_dir / ".dataset_active.lock")
    if not publish_lock.acquire():
        raise RuntimeError("timed out waiting for active dataset shards before merge")
    try:
        return _merge_shards_locked(output_dir, config)
    finally:
        publish_lock.release()


def _merge_shards_locked(output_dir: Path, config: dict | None = None) -> dict:
    """Implementation of merge_shards with the exclusive dataset lease held."""
    output_dir = Path(output_dir).resolve()
    _recover_transactions(output_dir)
    _invalidate_dataset_ready(output_dir)
    plan = load_existing_plan(output_dir, config=config)
    case_status = _load_case_status(output_dir / "case_status")

    expected_cases = {c["case_id"]: c for c in plan["_cases"]}
    unexpected_cases = sorted(set(case_status) - set(expected_cases))
    invalid_cases = []
    valid_status = {}
    failed_status = {}
    freqs = plan["_frequencies"]
    geometry_cache = {}
    for cid, case in expected_cases.items():
        st = case_status.get(cid)
        if isinstance(st, dict) and st.get("status") == "done":
            if validate_case_complete(output_dir, case, plan, freqs, st,
                                      geometry_cache=geometry_cache):
                valid_status[cid] = st
            else:
                invalid_cases.append(cid)
        elif _valid_terminal_failure_status(st, case, plan["plan_hash"]):
            failed_status[cid] = st
        else:
            invalid_cases.append(cid)

    seen = {}
    dups = []
    for cid, st in valid_status.items():
        for row in st["rows"]:
            sid = row["sample_id"]
            if sid in seen:
                dups.append(sid)
            seen[sid] = cid
    expected_samples = {
        f"{cid}_b{bi}"
        for cid, c in expected_cases.items() if cid in valid_status
        for bi in range(int(c["n_bridges"]))
    }
    planned_samples = sum(int(c["n_bridges"]) for c in plan["_cases"])
    missing_samples = sorted(expected_samples - set(seen))
    extra_samples = sorted(set(seen) - expected_samples)
    summary = {
        "plan_hash": plan["plan_hash"],
        "completion_policy": COMPLETION_POLICY,
        "n_cases": len(plan["_cases"]),
        "n_planned_cases": len(plan["_cases"]),
        "n_done": len(valid_status), "n_failed": len(failed_status),
        "n_samples": len(seen),
        "n_planned_samples": planned_samples,
        "n_expected_samples": len(expected_samples),
        "duplicate_sample_ids": sorted(set(dups)),
        "invalid_or_missing_case_ids": invalid_cases,
        "unexpected_case_ids": unexpected_cases,
        "missing_sample_ids": missing_samples,
        "extra_sample_ids": extra_samples,
        "complete": not (dups or invalid_cases or unexpected_cases
                         or missing_samples or extra_samples),
    }
    if not summary["complete"]:
        _invalidate_dataset_ready(output_dir, quarantine_manifest=True)
        _atomic_write_bytes(output_dir / "merge_summary.json",
                            json.dumps(summary, indent=2).encode())
        raise RuntimeError(
            "merge refused an incomplete/inconsistent dataset: "
            f"valid cases {len(valid_status)}/{len(expected_cases)}, "
            f"samples {len(seen)}/{len(expected_samples)}, "
            f"duplicates={len(dups)}, unexpected_cases={len(unexpected_cases)}")
    rebuild_manifest(output_dir, valid_status, plan)
    summary["manifest_sha256"] = _file_sha256(output_dir / "manifest.csv")
    _atomic_write_bytes(output_dir / "merge_summary.json",
                        json.dumps(summary, indent=2).encode())
    _publish_dataset_ready(
        output_dir, plan, summary, included_case_ids=set(valid_status),
        failed_case_ids=set(failed_status))
    return summary


# ===========================================================================
# Injectable execution context
# ===========================================================================

@dataclasses.dataclass
class GenContext:
    """Boundaries so the orchestration is testable without FEniCSx.

    prepare_geometry(shape_meta, body_type, shape_dir, freqs) -> geom dict with
        paths RELATIVE to shape_dir (persisted; built once per shape x body).
    solve_case(case, shape_meta, material, geom, freqs, budget, staging) -> response
        (subprocess FEM solve; artifacts under `staging`).
    max_mode_retries: extra attempts with escalated mode counts on a coverage miss.
    """
    prepare_geometry: object = None
    solve_case: object = None
    max_mode_retries: int = 2

    def __post_init__(self):
        if self.prepare_geometry is None:
            self.prepare_geometry = _real_prepare_geometry
        if self.solve_case is None:
            self.solve_case = _real_solve_case


def _is_coverage_failure(reasons: list[str]) -> bool:
    return any(("band_covered" in r) or ("coverage_ok" in r) for r in reasons)


def _archive_failed_attempt(staging: Path | None, failed_root: Path, case_id: str,
                            attempt: int, reasons: list[str], response: dict | None):
    if staging is None or not staging.exists():
        return
    _atomic_write_bytes(staging / "attempt_failure.json", json.dumps({
        "case_id": case_id,
        "attempt": int(attempt),
        "qc_reason": list(reasons),
        "error": (response or {}).get("error"),
        "worker_returncode": (response or {}).get("worker_returncode"),
    }, indent=2, default=_json_default).encode())
    target_parent = failed_root / case_id
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / f"attempt_{attempt:02d}_{int(time.time() * 1e6)}"
    os.rename(staging, target)


def _write_non_success_status(output_dir: Path, cases_root: Path, status_root: Path,
                              case_id: str, status: dict) -> tuple[dict, bool]:
    """Write failed/stale status without clobbering a concurrent committed success.

    The successful directory transaction uses this same stable per-case lock.  A
    late failing duplicate shard therefore waits for an in-flight commit and then
    preserves its certified done status instead of invalidating completed FEM work.
    """
    lock = _AdvisoryFileLock(
        Path(output_dir) / ".transaction_locks" / f"{case_id}.lock")
    if not lock.acquire():
        raise RuntimeError(f"timed out waiting to record status for {case_id}")
    try:
        status_path = Path(status_root) / f"{case_id}.json"
        current = None
        if status_path.is_file():
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                current = None
        if (isinstance(current, dict) and current.get("case_id") == case_id
                and current.get("status") == "done"
                and _is_committed(Path(cases_root) / case_id)):
            return current, True
        _atomic_write_bytes(status_path,
                            json.dumps(status, default=_json_default).encode())
        return status, False
    finally:
        lock.release()


# ===========================================================================
# Orchestration
# ===========================================================================

def _mesh_warm_task(args):
    """Subprocess task: mesh ONE (shape, body) into the on-disk geometry cache.

    Meshing (gmsh) is CPU-bound and low-memory, so unlike the memory-bound solve it
    parallelises cleanly across PROCESSES (one gmsh model per process — never share a
    gmsh model between threads).  This only WARMS the persisted geometry cache; the
    serial Phase-1 loop stays the source of truth, so any failure here is harmless
    (the shape is simply (re)meshed / recorded there).  Returns (shape_id, ok, err).
    """
    sid, body, shape_meta, shapes_root_str, ph = args
    try:
        prepare_or_load_geometry(GenContext(), shape_meta, body, sid,
                                 Path(shapes_root_str), ph)
        return (sid, True, "")
    except BaseException:                     # never let a child crash the pool
        return (sid, False, traceback.format_exc()[-400:])


def _warm_geometry_cache_parallel(targets, shape_by_id, shapes_root, ph, mesh_workers):
    """Best-effort parallel warm of the geometry cache for targets [(sid, b, body)].

    Uses a 'spawn' pool so each worker starts gmsh from a clean interpreter (fork can
    inherit a dirty gmsh/global state).  Failures are logged, not raised: the caller's
    serial loop then loads the warmed cache or (re)meshes/records as usual.  With
    mesh_workers <= 1 this is never called, so the proven serial path is unchanged.
    """
    import multiprocessing as _mp
    tasks = [(sid, body, shape_by_id[b], str(shapes_root), ph)
             for (sid, b, body) in targets]
    n = max(1, min(int(mesh_workers), len(tasks)))
    t0 = time.time()
    ok = 0
    try:
        with _mp.get_context("spawn").Pool(processes=n) as pool:
            for sid, good, err in pool.imap_unordered(_mesh_warm_task, tasks):
                if good:
                    ok += 1
                else:
                    print(f"[mesh-warm] shape {sid} failed (serial loop will retry): "
                          f"{err}", flush=True)
    except BaseException as exc:               # fall through to serial meshing
        print(f"[mesh-warm] parallel warm aborted ({exc!r}); using serial meshing",
              flush=True)
        return
    print(f"[mesh-warm] warmed {ok}/{len(tasks)} geometries with {n} procs "
          f"in {time.time() - t0:.0f}s", flush=True)


def run_generation(config: dict, output_dir: Path, ctx: GenContext = None,
                   num_shards: int = 1, shard_index: int = 0,
                   max_cases: int = 0, resume: bool = True,
                   do_merge: bool = None, mesh_workers: int = 1,
                   run_policy: dict | None = None) -> dict:
    """Acquire a shared generation lease and release it on every exit path."""
    output_dir = Path(output_dir).resolve()
    run_lock = _AdvisoryFileLock(output_dir / ".dataset_active.lock", shared=True)
    if not run_lock.acquire():
        raise RuntimeError("timed out waiting for dataset generation lease")
    try:
        return _run_generation_locked(
            config, output_dir, ctx=ctx, num_shards=num_shards,
            shard_index=shard_index, max_cases=max_cases, resume=resume,
            do_merge=do_merge, run_lock=run_lock, mesh_workers=mesh_workers,
            run_policy=run_policy)
    finally:
        run_lock.release()


def _run_generation_locked(config: dict, output_dir: Path, ctx: GenContext = None,
                           num_shards: int = 1, shard_index: int = 0,
                           max_cases: int = 0, resume: bool = True,
                           do_merge: bool = None,
                           run_lock: _AdvisoryFileLock | None = None,
                           mesh_workers: int = 1,
                           run_policy: dict | None = None) -> dict:
    """Two-phase, fail-closed, resumable generation for this shard.

    Phase 1: build/validate the persistent geometry for each (shape, body) once.
    Phase 2: solve each shape x body x material case in a subprocess with adaptive
             mode retries; stage -> QC -> atomically commit the case artifact and
             the per-bridge samples.  A shard writes only its own manifest_shard*;
             the canonical manifest.csv is (re)built by the merge step.
    """
    ctx = ctx or GenContext()
    validate_config(config)
    validate_run_options(num_shards, shard_index, max_cases, ctx.max_mode_retries)
    output_dir = Path(output_dir).resolve()
    samples_root = output_dir / "samples"
    shapes_root = output_dir / "shapes"
    status_root = output_dir / "case_status"
    cases_root = output_dir / "cases"
    for r in (samples_root, shapes_root, status_root, cases_root):
        r.mkdir(parents=True, exist_ok=True)
    if run_lock is None or run_lock.file is None:
        raise RuntimeError("generation implementation requires an active dataset lease")
    _recover_transactions(output_dir)
    _invalidate_dataset_ready(output_dir)

    plan = load_or_create_plan(config, output_dir)
    ph = plan["plan_hash"]
    shape_by_id = {s["base_shape_id"]: s for s in plan["_base_shapes"]}
    mat_by_id = {m["material_id"]: m for m in plan["_materials"]}
    freqs = plan["_frequencies"]
    my_cases = shard_cases(plan["_cases"], num_shards, shard_index)

    case_status = _load_case_status(status_root)
    geometry_cache = {}
    disk_valid = {
        c["case_id"]: validate_case_complete(
            output_dir, c, plan, freqs, case_status.get(c["case_id"]),
            geometry_cache=geometry_cache)
        for c in my_cases
    }
    pending_cases = []
    n_done = 0
    for case in my_cases:
        cid = case["case_id"]
        status = case_status.get(cid)
        if resume and disk_valid[cid]:
            n_done += 1
            continue
        if resume and _valid_terminal_failure_status(status, case, ph):
            # A production failure is terminal by default: retain its diagnostics,
            # omit it from the training dataset, and continue with other cases.
            continue
        if not disk_valid[cid] and (status or {}).get("status") == "done":
            stale = {"case_id": cid, "status": "stale", "attempt": 0,
                     "plan_hash": ph, "qc_reason": ["resume validation failed"],
                     "attempts_log": []}
            case_status[cid] = stale
        pending_cases.append(case)
    # Scheduling only: reorder / defer pending work.  `my_cases` stays the full
    # shard set so resume validation, the manifest and the completion accounting
    # below still describe the whole plan, and anything the policy defers simply
    # remains pending for a later run.
    policy_cases = apply_run_policy(pending_cases, run_policy, mat_by_id)
    if run_policy:
        print(f"[mixed] run policy: {len(policy_cases)}/{len(pending_cases)} pending "
              f"cases selected this invocation "
              f"(skip_shapes={len(run_policy['skip_shapes'])}, "
              f"materials_per_shape={len(run_policy['materials_per_shape'])} shapes, "
              f"ordered={len(run_policy['shape_order'])} shapes)", flush=True)
    attempt_cases = policy_cases[:max_cases] if max_cases else policy_cases

    # Build only geometry needed by cases this invocation will actually attempt.
    geom_by_sid: dict[int, dict] = {}
    seen = {}
    for c in attempt_cases:
        seen.setdefault(c["shape_id"], (c["base_shape_id"], c["body_type"]))
    # Optional parallel pre-warm of the geometry cache (meshing is CPU-bound + low
    # memory, so it scales across processes where the memory-bound solve cannot).
    # Purely additive: the serial loop below still produces geom_by_sid, so
    # mesh_workers <= 1 keeps the proven path byte-identical.
    if mesh_workers and int(mesh_workers) > 1:
        warm_targets = [
            (sid, b, body) for sid, (b, body) in sorted(seen.items())
            if not (body == "hollow"
                    and not shape_by_id[b].get("soundhole_ok", False))]
        if warm_targets:
            _warm_geometry_cache_parallel(warm_targets, shape_by_id, shapes_root,
                                          ph, int(mesh_workers))
    for sid, (b, body) in sorted(seen.items()):
        shape_meta = shape_by_id[b]
        if body == "hollow" and not shape_meta.get("soundhole_ok", False):
            geom_by_sid[sid] = {"ok": False,
                                "reason": "hollow soundhole placement failed"}
            continue
        try:
            geom_by_sid[sid] = {"ok": True,
                                **prepare_or_load_geometry(ctx, shape_meta, body, sid,
                                                           shapes_root, ph)}
        except Exception:
            geom_by_sid[sid] = {"ok": False,
                                "reason": f"geometry: {traceback.format_exc()[-1000:]}"}

    failed_root = output_dir / "failed_attempts"
    failed_root.mkdir(parents=True, exist_ok=True)
    n_attempt_failures = n_attempted = 0
    for case in attempt_cases:
        cid = case["case_id"]
        n_attempted += 1
        shape_meta = shape_by_id[case["base_shape_id"]]
        material = mat_by_id[case["material_id"]]
        geom = geom_by_sid.get(case["shape_id"], {"ok": False, "reason": "no geometry"})
        old_status = case_status.get(cid)

        status = {"case_id": cid, "status": "failed", "attempt": 0,
                  "plan_hash": ph, "qc_reason": [], "attempts_log": []}
        if not geom.get("ok"):
            status["qc_reason"] = [geom.get("reason", "geometry unavailable")]
            if not disk_valid[cid]:
                stored, _preserved = _write_non_success_status(
                    output_dir, cases_root, status_root, cid, status)
                case_status[cid] = stored
            elif old_status is not None:
                case_status[cid] = old_status
            n_attempt_failures += 1
            continue

        status.update(geometry_digest=geom.get("geometry_digest"),
                      mesh_sha256=geom.get("mesh_sha256"))
        t0 = time.time()
        budget = initial_mode_budget(case, config)
        ok = False
        reasons = []
        response = None
        staging = None
        for attempt in range(int(ctx.max_mode_retries) + 1):
            staging = cases_root / f"{cid}.staging_{os.getpid()}_{attempt + 1}"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            try:
                execution_case = {**case,
                                  "_case_timeout_s": float(config["case_timeout_s"])}
                response = ctx.solve_case(execution_case, shape_meta, material, geom,
                                          freqs, budget, staging)
            except Exception:
                response = {"run_status": "failed", "diagnostics": {}, "bridges": [],
                            "error": traceback.format_exc()[-2000:]}
            if not isinstance(response, dict):
                response = {"run_status": "failed", "diagnostics": {}, "bridges": [],
                            "error": "solver returned a non-object response"}
            expected_input_hash = case_input_hash(case, material, geom, freqs, budget)
            try:
                ok, reasons = qc_case(case, response, freqs, expected_geom=geom,
                                      expected_budget=budget,
                                      expected_input_hash=expected_input_hash)
            except Exception:
                ok = False
                reasons = [f"QC exception: {traceback.format_exc()[-1000:]}"]
                if not isinstance(response, dict):
                    response = {"run_status": "failed", "diagnostics": {},
                                "bridges": [], "error": "malformed solver response"}
            status["attempt"] = attempt + 1
            status["attempts_log"].append(
                 {"attempt": attempt + 1, "mode_budget": budget, "ok": ok,
                  "case_input_hash": expected_input_hash,
                  "qc_reason": reasons, "error": response.get("error"),
                 "n_modes": _budget_n_modes(budget, case["body_type"])})
            if ok:
                break
            _archive_failed_attempt(staging, failed_root, cid, attempt + 1,
                                    reasons, response)
            staging = None
            # Escalate the mode budget ONLY when the solve actually succeeded but the
            # basis genuinely fell short of the band.  A crashed/failed worker leaves
            # empty diagnostics (coverage_ok -> None -> "coverage_ok=false"), which is
            # NOT a coverage miss; retrying it just burns compute on a doomed case.
            if (response.get("run_status") == "complete"
                    and _is_coverage_failure(reasons)
                    and attempt < int(ctx.max_mode_retries)):
                budget = escalate_mode_budget(budget, case["body_type"])
                continue
            break

        if ok:
            sample_pairs: list[tuple[Path, Path]] = []
            try:
                transaction_id = uuid.uuid4().hex
                write_case_artifact(staging, case, response, freqs, ph, geom, budget,
                                    status["attempt"], transaction_id, material)
                # Samples are promoted first while the previous committed case is
                # still intact.  Status is written last; any interruption is caught
                # by case/sample equality checks on resume.
                rows, sample_pairs = fan_out_case(
                    case, shape_meta, material, response, freqs, ph,
                    samples_root, geom, budget, time.time() - t0, status["attempt"],
                    transaction_id)
                status.update(status="done", qc_reason=[], rows=rows,
                              transaction_id=transaction_id,
                              case_input_hash=response.get("case_input_hash"),
                              solver_revision=response.get("solver_revision"),
                              solve_time_s=round(time.time() - t0, 2))
                case_status[cid] = status
                status_path = status_root / f"{cid}.json"
                pending_transaction = _commit_dirs_transaction(
                    [*sample_pairs, (staging, cases_root / cid)],
                    certification_path=status_path, transaction_id=transaction_id)
                journal_path, transaction_lock = pending_transaction
                try:
                    _atomic_write_bytes(status_path,
                                        json.dumps(status, default=_json_default).encode())
                except Exception:
                    _recover_transaction_journal(journal_path,
                                                 held_lock=transaction_lock)
                    raise
                _finalize_transaction_journal(journal_path,
                                              held_lock=transaction_lock)
                n_done += 1
            except Exception:
                # fan_out_case may fail after creating only some sample staging
                # directories, before it can return their paths to the caller.
                # They are not journalled yet, so clean this process's remnants
                # explicitly instead of leaving them to accumulate in a long run.
                _rmtree_glob(samples_root, f"{cid}_b*.staging_{os.getpid()}")
                reasons = [f"artifact promotion: {traceback.format_exc()[-1000:]}"]
                _archive_failed_attempt(staging, failed_root, cid, status["attempt"],
                                        reasons, response)
                ok = False
        if not ok:
            status.update(status="failed", qc_reason=reasons)
            stored, _preserved = _write_non_success_status(
                output_dir, cases_root, status_root, cid, status)
            case_status[cid] = stored
            n_attempt_failures += 1

    # --- manifest: per-shard file for parallel safety; canonical only on merge ---
    shard_rows = _manifest_rows(case_status, plan,
                                only_cases={c["case_id"] for c in my_cases})
    _write_manifest(output_dir / f"manifest_shard{shard_index}.csv", shard_rows)

    statuses_now = _load_case_status(status_root)
    n_failed = sum(
        _valid_terminal_failure_status(statuses_now.get(c["case_id"]), c, ph)
        for c in my_cases)
    all_statuses_terminal = all(
        (isinstance(statuses_now.get(c["case_id"]), dict)
         and statuses_now[c["case_id"]].get("status") == "done")
        or _valid_terminal_failure_status(statuses_now.get(c["case_id"]), c, ph)
        for c in plan["_cases"])

    ready_published = False
    if do_merge is None:
        # The last finishing shard publishes automatically once every planned case
        # has reached a terminal success/failure state.  No separate merge command
        # is required for the default production workflow.
        do_merge = all_statuses_terminal
    if do_merge:
        # Upgrade shared generation -> exclusive publication without deadlock.
        run_lock.release()
        merge_shards(output_dir, config)
        ready_published = True

    run_lock.release()

    summary = {"plan_hash": ph, "n_cases_assigned": len(my_cases),
               "n_done": n_done, "n_failed": int(n_failed),
               "n_attempt_failures": n_attempt_failures,
               "n_new": n_attempted,
               "n_attempted": n_attempted,
               "num_shards": num_shards, "shard_index": shard_index,
               "dataset_ready_published": ready_published,
               # audit trail: scheduling only, never part of plan_hash
               "run_policy": run_policy}
    _atomic_write_bytes(output_dir / f"run_summary_shard{shard_index}.json",
                        json.dumps(summary, indent=2).encode())
    print(f"[mixed] plan {ph[:12]}  shard {shard_index}/{num_shards}  "
          f"cases={len(my_cases)}  done={n_done}  failed={n_failed}")
    return summary


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Mixed solid/hollow guitar admittance dataset")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-base-shapes", type=int, default=100)
    p.add_argument("--body-types", nargs="+", default=["solid", "hollow"],
                   choices=["solid", "hollow"])
    p.add_argument("--n-materials", type=int, default=10)
    p.add_argument("--n-bridge-points", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--freq-min", type=float, default=DEFAULT_FREQ_MIN)
    p.add_argument("--freq-max", type=float, default=DEFAULT_FREQ_MAX)
    p.add_argument("--freq-points", type=int, default=DEFAULT_FREQ_POINTS)
    p.add_argument("--solid-eigen-fmax", "--solid-modal-fmax",
                   dest="solid_modal_fmax", type=float, default=7500.0,
                   help="Solid auxiliary eigensolve coverage cutoff [Hz].")
    p.add_argument("--solid-n-modes", type=int, default=400)
    p.add_argument("--coupled-struct-fmax", type=float, default=7500.0)
    p.add_argument("--coupled-acoustic-fmax", type=float, default=7500.0)
    p.add_argument("--coupled-n-struct-modes", type=int, default=400)
    p.add_argument("--coupled-n-acoustic-modes", type=int, default=360)
    p.add_argument("--coupled-n-attach-acoustic", type=int, default=20)
    p.add_argument("--port-end-corrections", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--mesh-workers", type=int, default=1,
                   help="Parallel processes for warming the geometry (mesh) cache "
                        "before solving (default 1 = serial, unchanged). Meshing is "
                        "CPU-bound + low-memory; the solve phase is unaffected.")
    p.add_argument("--run-policy", type=Path, default=None,
                   help="JSON scheduling policy (mixed-run-policy-v1): shape_order / "
                        "materials_per_shape / skip_shapes.  RUNTIME ONLY — it never "
                        "enters the config or plan_hash; deferred cases stay pending.")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--max-mode-retries", type=int, default=2)
    p.add_argument("--case-timeout-s", type=float, default=DEFAULT_CASE_TIMEOUT_S,
                   help="Maximum wall time for one FEM worker attempt (default: 6 h).")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Reuse only fully revalidated cases (default: enabled).")
    p.add_argument("--plan-only", action="store_true",
                   help="Build/write the immutable plan and exit (no FEM).")
    p.add_argument("--merge", action="store_true",
                   help="Reload all shard statuses and rebuild the canonical manifest.")
    return p.parse_args()


def config_from_args(args) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "n_base_shapes": args.n_base_shapes,
        "body_types": list(args.body_types),
        "n_materials": args.n_materials,
        "n_bridge_points": args.n_bridge_points,
        "seed": args.seed,
        "freq_min": args.freq_min, "freq_max": args.freq_max,
        "freq_points": args.freq_points,
        "solid_modal_fmax": args.solid_modal_fmax, "solid_n_modes": args.solid_n_modes,
        "coupled_struct_fmax": args.coupled_struct_fmax,
        "coupled_acoustic_fmax": args.coupled_acoustic_fmax,
        "coupled_n_struct_modes": args.coupled_n_struct_modes,
        "coupled_n_acoustic_modes": args.coupled_n_acoustic_modes,
        "coupled_n_attach_acoustic": args.coupled_n_attach_acoustic,
        "port_end_corrections": args.port_end_corrections,
        "case_timeout_s": args.case_timeout_s,
    }


if __name__ == "__main__":
    args = _parse_args()
    cfg = config_from_args(args)
    if args.plan_only:
        validate_run_options(args.num_shards, args.shard_index, args.max_cases,
                             args.max_mode_retries)
        plan = load_or_create_plan(cfg, args.output_dir)
        print(f"[mixed] plan: {args.output_dir/'dataset_plan.json'}  "
              f"hash={plan['plan_hash']}  cases={len(plan['_cases'])}")
    elif args.merge:
        summ = merge_shards(args.output_dir)
        print(f"[mixed] merged: done={summ['n_done']}/{summ['n_cases']}  "
              f"samples={summ['n_samples']}")
    else:
        ctx = GenContext(max_mode_retries=args.max_mode_retries)
        policy = load_run_policy(args.run_policy) if args.run_policy else None
        run_generation(cfg, args.output_dir, ctx=ctx, num_shards=args.num_shards,
                       shard_index=args.shard_index, max_cases=args.max_cases,
                       resume=args.resume, mesh_workers=args.mesh_workers,
                       run_policy=policy)
