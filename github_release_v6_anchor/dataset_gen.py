"""
dataset_gen.py
-----------------
Dataset generation using the shape library (guitar_shapes + shape_gen)
and the full FEM pipeline.

Structure
---------
dataset/
  manifest.csv
  shapes/
    shape_0001/
      contour.npy          (120,2) mm, body-centred y+=neck
      model/model.step
      mesh/mesh.msh
      params.json          body_type, soundhole, bridge_pts, geom
  samples/
    sample_0001/
      admittance.npz
      admittance.png
      sample_params.json

Usage
-----
  python dataset_gen.py --dry-run
  python dataset_gen.py --no-fem
  python dataset_gen.py
  python dataset_gen.py --output-dir dataset/ --n-shapes 50
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# ---------------------------------------------------------------------------
# Canvas constants (must match model_builder.py / frontend)
# ---------------------------------------------------------------------------
CANVAS_PX    = 600
CANVAS_MM    = 800
PX_PER_MM    = CANVAS_PX / CANVAS_MM   # 0.75 px/mm
MM_PER_PX    = CANVAS_MM / CANVAS_PX   # 4/3 mm/px
CANVAS_CX_MM = CANVAS_MM / 2.0         # 400 mm (canvas x-centre)
CANVAS_CY_MM = CANVAS_MM / 2.0         # 400 mm (canvas y-centre, y+ downward)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
N_TOTAL_SHAPES    = 50
GUITAR_RATIO      = 0.10        # 10 % guitar-model shapes (= 5)
N_BRIDGE_PTS      = 3
N_MATERIALS       = 3           # all random

# Mesh defaults
MESH_SIZE_MIN = 8.0             # mm
MESH_SIZE_MAX = 10.0            # mm

# FEM defaults
FREQ_MIN      = 20.0            # Hz
FREQ_MAX      = 5000.0          # Hz
FREQ_POINTS   = 500
RAYLEIGH_ALPHA = 0.0
RAYLEIGH_BETA  = 5e-6

# Random material parameter ranges (SI units)
MATERIAL_RANGES: dict[str, tuple[float, float]] = {
    "E1":      (8.0e9,  14.0e9),   # longitudinal Young's modulus
    "E2":      (0.8e9,   2.5e9),   # radial
    "E3":      (0.4e9,   1.0e9),   # tangential
    "G12":     (1.0e9,   1.8e9),   # shear LR
    "G13":     (1.0e9,   1.8e9),   # shear LT
    "G23":     (0.05e9,  0.3e9),   # shear RT
    "nu12":    (0.30,    0.50),
    "nu13":    (0.30,    0.50),
    "nu23":    (0.30,    0.60),
    "density": (300.0,  700.0),
}

MANIFEST_COLS = [
    "sample_id", "shape_id", "shape_type", "guitar_model",
    "body_type", "body_thickness", "cavity_depth_ratio",
    "soundhole_type", "soundhole_diameter", "soundhole_x", "soundhole_y",
    "bridge_x", "bridge_y", "bridge_idx",
    "material_name",
    "E_L", "E_R", "E_T", "G_LR", "G_LT", "G_RT",
    "nu_LR", "nu_LT", "nu_RT", "density",
    # Modal-backend metadata (blank for harmonic samples)
    "method", "modal_fmax", "n_modes_retained", "eig_freq_max_hz",
    "coverage_ok", "snap_distance_mm", "harmonic_batch",
    "status",
]


# ===========================================================================
# Coordinate helpers
# ===========================================================================

def _body_to_canvas_mm(bx: float, by: float) -> tuple[float, float]:
    """Body-centred (y+ = toward neck) -> canvas mm (y+ = downward)."""
    return bx + CANVAS_CX_MM, -by + CANVAS_CY_MM


def _body_to_fenics_coords(bx: float, by: float, thickness: float) -> tuple:
    """Body-centred mm (y+ neck) -> fenics / STEP frame coords.

    pixels_to_contour subtracts the centroid (≈ canvas centre = 400,400 mm),
    so the STEP body frame has x the same as body_x, but y is flipped.
    """
    return (bx, -by, thickness)


def _rasterize_contour(contour_mm: np.ndarray) -> list[list[int]]:
    """(N,2) body-centred contour (y+ neck) -> list of [px, py] canvas pixels."""
    import cv2
    pts_px = np.column_stack([
        contour_mm[:, 0] * PX_PER_MM + CANVAS_PX / 2,
        -contour_mm[:, 1] * PX_PER_MM + CANVAS_PX / 2,
    ]).round().astype(np.int32)
    mask = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px], 1)
    ys, xs = np.where(mask > 0)
    return [[int(x), int(y)] for x, y in zip(xs, ys)]


# ===========================================================================
# Shape generation
# ===========================================================================

def _make_all_shapes(
    rng: np.random.Generator,
    n_total: int,
    guitar_ratio: float,
) -> list[dict]:
    """Return list of shape dicts: {shape_id, shape_type, guitar_model, contour_mm}."""
    from guitar_shapes import get_guitar_contour, GUITAR_SHAPES
    from shape_gen import random_shape

    n_guitar = max(1, round(n_total * guitar_ratio))
    n_random = n_total - n_guitar

    shapes: list[dict] = []

    # Select guitar models (cycle if n_guitar > 8)
    guitar_names = []
    for i in range(n_guitar):
        guitar_names.append(GUITAR_SHAPES[i % len(GUITAR_SHAPES)])
    for name in guitar_names:
        shapes.append({
            "shape_type":  "guitar_model",
            "guitar_model": name,
            "contour_mm":  get_guitar_contour(name),
        })

    # Random shapes
    for _ in range(n_random):
        seed = int(rng.integers(0, 2 ** 31))
        shapes.append({
            "shape_type":  "random",
            "guitar_model": "",
            "contour_mm":  random_shape(seed=seed),
        })

    # Shuffle and assign IDs
    indices = rng.permutation(len(shapes))
    shapes  = [shapes[i] for i in indices]
    for i, s in enumerate(shapes):
        s["shape_id"] = i + 1

    return shapes


# ===========================================================================
# Material sampling
# ===========================================================================

def _random_material_dict(rng: np.random.Generator,
                          material_ranges: dict | None = None) -> dict:
    """Sample orthotropic material properties from material_ranges (or MATERIAL_RANGES)."""
    ranges = material_ranges if material_ranges is not None else MATERIAL_RANGES
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}


def _get_materials(rng: np.random.Generator,
                   n_materials: int = N_MATERIALS,
                   material_ranges: dict | None = None) -> list[tuple[str, dict]]:
    """Return n_materials randomly sampled orthotropic material dicts."""
    return [("random", _random_material_dict(rng, material_ranges))
            for _ in range(n_materials)]


# ===========================================================================
# Model / mesh building
# ===========================================================================

def _make_data_dict(
    contour_mm:       np.ndarray,
    body_type:        str,
    bridge_pt_body:   tuple[float, float],   # body-centred (y+ neck)
    soundhole_type:   str,
    soundhole_diam:   float,
    soundhole_center: tuple[float, float] | None,   # body-centred
    thickness:        float,
    cavity_ratio:     float,
    top_plate_thickness_mm: float | None = None,     # v5: explicit top plate (mm)
    include_raster:   bool = True,
) -> dict:
    """Construct the data dict expected by model_builder.build_model.

    ``body_contour_mm`` is the authoritative CAD input.  ``body_pixels`` is a
    legacy frontend payload and requires OpenCV; callers that only build CAD can
    set ``include_raster=False`` to avoid an unnecessary runtime dependency and
    a lossy contour rasterization that is not consumed by ``build_model``.
    """
    cx_canvas, cy_canvas = _body_to_canvas_mm(*bridge_pt_body)

    if soundhole_type != "none" and soundhole_center is not None:
        hx, hy = _body_to_canvas_mm(*soundhole_center)
        hole_center_mm = [hx, hy]
    else:
        soundhole_type = "none"
        hole_center_mm = None

    pixels = _rasterize_contour(contour_mm) if include_raster else []

    # The FEM/CadQuery frame has +y in the opposite direction from the
    # body-placement frame.  Keep the legacy raster payload for the frontend, but
    # also provide the exact contour and point coordinates.  model_builder uses
    # these direct fields when present, avoiding the old
    # contour -> bitmap -> findContours -> RDP -> spline round-trip (which could
    # change random shapes by tens of millimetres).
    contour_step = np.column_stack([contour_mm[:, 0], -contour_mm[:, 1]])
    bridge_step = [float(bridge_pt_body[0]), -float(bridge_pt_body[1])]
    hole_step = ([float(soundhole_center[0]), -float(soundhole_center[1])]
                 if soundhole_type != "none" and soundhole_center is not None
                 else None)

    params = {
        "body_type":           body_type,
        "top_type":            "flat",
        "body_thickness":      thickness,
        "cavity_depth_ratio":  cavity_ratio,
        # v5: explicit top-plate thickness [mm] (source of truth; cavity_depth_ratio
        # above is derived from it).  build_model verifies the CAD honours this.
        "top_plate_thickness_mm": top_plate_thickness_mm,
        "center_block_width":  40.0,
        "wall_thickness":      5.0,
        "hole_type":           soundhole_type,
        "hole_params": {
            "round":       {"diameter": soundhole_diam},
            "oval":        {"length": soundhole_diam * 1.5,
                            "width":  soundhole_diam * 0.6},
            "f-hole":      {"length": soundhole_diam,
                            "slot_width": 8.0, "offset_x": 30.0},
            "user-defined": {},
        },
        "bracing":        "none",
        "top_arch_height": 0.0,
        "back_arch_height": 0.0,
    }

    return {
        "canvas_px":        CANVAS_PX,
        "canvas_mm":        CANVAS_MM,
        "body_pixels":      pixels,
        "bridge_point_mm":  [cx_canvas, cy_canvas],
        "hole_center_mm":   hole_center_mm,
        "body_contour_mm":  contour_step.tolist(),
        "bridge_point_body_mm": bridge_step,
        "hole_center_body_mm": hole_step,
        "params":           params,
    }


def _build_shape(
    shape_rec:     dict,
    body_type:     str,
    thickness:     float,
    cavity_ratio:  float,
    shape_dir:     Path,
    rng:           np.random.Generator,
    verbose:       bool  = True,
    n_bridge_pts:  int   = N_BRIDGE_PTS,
    mesh_size_min: float = MESH_SIZE_MIN,
    mesh_size_max: float = MESH_SIZE_MAX,
) -> dict:
    """Build STEP + mesh for one shape. Returns shape_params dict."""
    from model_builder import build_model, BACK_PLATE_MM
    from mesh_gen import generate_mesh
    from placement_utils import random_bridge_points, random_soundhole

    contour_mm = shape_rec["contour_mm"]
    sid        = shape_rec["shape_id"]
    shape_dir.mkdir(parents=True, exist_ok=True)

    # ---------- bridge points ----------
    for min_edge in (25.0, 15.0, 10.0):
        try:
            bridge_pts = random_bridge_points(
                contour_mm, n=n_bridge_pts, min_edge_dist=min_edge, rng=rng)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"shape {sid}: failed to place {n_bridge_pts} bridge points")

    # ---------- soundhole ----------
    soundhole_type   = "none"
    soundhole_diam   = 0.0
    soundhole_center: tuple | None = None

    if body_type == "hollow":
        soundhole_diam = float(rng.uniform(30.0, 80.0))
        center = random_soundhole(
            contour_mm, bridge_pts,
            diameter_mm=soundhole_diam,
            min_bridge_dist=40.0,
            min_edge_dist=20.0,
            rng=rng,
        )
        if center is not None:
            soundhole_type   = "round"
            soundhole_center = (float(center[0]), float(center[1]))
        else:
            soundhole_diam = 0.0   # no valid position found

    # ---------- STEP ----------
    model_dir = shape_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    data = _make_data_dict(
        contour_mm, body_type,
        (float(bridge_pts[0, 0]), float(bridge_pts[0, 1])),
        soundhole_type, soundhole_diam, soundhole_center,
        thickness, cavity_ratio,
    )
    if verbose:
        print(f"  [A] Building STEP (body_type={body_type}) ...")
    model_params = build_model(data, model_dir)

    step_path = model_dir / "model.step"

    # ---------- bridge coords in STEP frame ----------
    bridge_coords_list = []
    for bp in bridge_pts:
        bx, by = float(bp[0]), float(bp[1])
        bridge_coords_list.append(list(_body_to_fenics_coords(bx, by, thickness)))

    # ---------- mesh (embed bridge_pts[0]) ----------
    mesh_dir = shape_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    bc0 = bridge_coords_list[0]
    if verbose:
        print(f"  [B] Meshing (bridge0={bc0}) ...")
    # For hollow bodies, refine the real top/back plate regions independently so
    # thin-plate bending modes are captured.  Geometry is asymmetric: flat top
    # plate = thickness·(1-cavity_ratio), back plate = fixed BACK_PLATE_MM.
    # Solid bodies: no refinement.
    top_plate_t  = 0.0
    back_plate_t = 0.0
    if body_type in ("hollow", "semi_hollow", "semi-hollow"):
        top_plate_t  = thickness * (1.0 - cavity_ratio)
        back_plate_t = BACK_PLATE_MM

    msh_path = generate_mesh(
        step_file=step_path,
        bridge_coords=tuple(bc0),
        mesh_size_min=mesh_size_min,
        mesh_size_max=mesh_size_max,
        output_dir=mesh_dir,
        top_plate_thickness=top_plate_t,
        back_plate_thickness=back_plate_t,
    )

    # ---------- save shape params ----------
    shape_params = {
        "shape_id":         sid,
        "shape_type":       shape_rec["shape_type"],
        "guitar_model":     shape_rec["guitar_model"],
        "body_type":        body_type,
        "body_thickness":   thickness,
        "cavity_depth_ratio": cavity_ratio,
        "soundhole_type":   soundhole_type,
        "soundhole_diameter": soundhole_diam,
        "soundhole_center": list(soundhole_center) if soundhole_center else None,
        "bridge_pts":       bridge_pts.tolist(),
        "bridge_coords":    bridge_coords_list,
        "step_path":        str(step_path),
        "msh_path":         str(msh_path),
    }
    with open(shape_dir / "params.json", "w") as f:
        json.dump(shape_params, f, indent=2)
    np.save(shape_dir / "contour.npy", contour_mm)

    return shape_params


# ===========================================================================
# FEM sample
# ===========================================================================

def _run_sample(
    msh_path:      Path,
    bridge_coords: tuple,
    material:      dict,
    sample_dir:    Path,
    freq_min:      float = FREQ_MIN,
    freq_max:      float = FREQ_MAX,
    freq_points:   int   = FREQ_POINTS,
    rayleigh_alpha: float = RAYLEIGH_ALPHA,
    rayleigh_beta:  float = RAYLEIGH_BETA,
    method:        str   = "harmonic",
    modal_fmax:    float = 7500.0,
    modal_nmodes:  int   = 400,
    damping:       str   = "rayleigh",
    zeta_const:    float = 0.01,
    save_modes:    bool  = False,
) -> None:
    from run_pipeline import _run_fenics
    sample_dir.mkdir(parents=True, exist_ok=True)
    _run_fenics(
        msh_path=msh_path,
        material=material,
        bridge_coords=bridge_coords,
        freq_min=freq_min,
        freq_max=freq_max,
        freq_points=freq_points,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        output_dir=sample_dir,
        solver="petsc",
        method=method,
        modal_fmax=modal_fmax,
        modal_nmodes=modal_nmodes,
        damping=damping,
        zeta_const=zeta_const,
        save_modes=save_modes,
    )


def _save_admittance_npz(sample_dir: Path, freqs, Y,
                         bridge_meta: dict | None = None,
                         coverage_ok: bool | None = None) -> None:
    """Write a sample's admittance.npz (+png) from arrays (modal batch path).

    Keeps the harmonic-compatible keys (frequencies, admittance) and adds the
    snapped-bridge + coverage metadata so modal samples match the single-run
    output format.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(frequencies=freqs, admittance=Y)
    if bridge_meta:
        if "bridge_requested_xyz" in bridge_meta:
            kwargs["bridge_requested_xyz"] = np.array(bridge_meta["bridge_requested_xyz"])
        if "bridge_snapped_xyz" in bridge_meta:
            kwargs["bridge_snapped_xyz"] = np.array(bridge_meta["bridge_snapped_xyz"])
        if "snap_distance_mm" in bridge_meta:
            kwargs["snap_distance_mm"] = np.array(bridge_meta["snap_distance_mm"])
    if coverage_ok is not None:
        kwargs["coverage_ok"] = np.array(coverage_ok)
    np.savez(str(sample_dir / "admittance.npz"), **kwargs)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mag_db = 20.0 * np.log10(np.abs(Y) + 1e-30)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(freqs, mag_db, lw=1.2, color="darkorange")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|Y| [dB re 1 m/s/N]")
        ax.grid(True, which="both", alpha=0.4)
        fig.tight_layout()
        fig.savefig(str(sample_dir / "admittance.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ===========================================================================
# Manifest helpers
# ===========================================================================

def _load_manifest(path: Path) -> tuple[list[dict], set[str]]:
    """Load existing manifest. Returns (rows, done_sample_ids)."""
    if not path.exists():
        return [], set()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done = {r["sample_id"] for r in rows if r.get("status") == "done"}
    return rows, done


def _append_manifest(path: Path, row: dict) -> None:
    """Append one row to manifest.csv (creates header if new)."""
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def _mat_to_manifest(name: str, m: dict) -> dict:
    """Flatten material dict to manifest column names."""
    return {
        "material_name": name,
        "E_L":  m["E1"],  "E_R": m["E2"],   "E_T":  m["E3"],
        "G_LR": m["G12"], "G_LT": m["G13"], "G_RT": m["G23"],
        "nu_LR": m["nu12"], "nu_LT": m["nu13"], "nu_RT": m["nu23"],
        "density": m["density"],
    }


# ===========================================================================
# Main dataset runner
# ===========================================================================

def run_dataset(
    output_dir:     Path   = Path("dataset"),
    n_shapes:       int    = N_TOTAL_SHAPES,
    guitar_ratio:   float  = GUITAR_RATIO,
    dry_run:        bool   = False,
    no_fem:         bool   = False,
    resume:         bool   = True,
    seed:           int    = 42,
    max_samples:    int    = 0,
    n_bridge_pts:   int    = N_BRIDGE_PTS,
    n_materials:    int    = N_MATERIALS,
    mesh_size_min:  float  = MESH_SIZE_MIN,
    mesh_size_max:  float  = MESH_SIZE_MAX,
    freq_min:       float  = FREQ_MIN,
    freq_max:       float  = FREQ_MAX,
    freq_points:    int    = FREQ_POINTS,
    rayleigh_alpha: float  = RAYLEIGH_ALPHA,
    rayleigh_beta:  float  = RAYLEIGH_BETA,
    material_ranges: dict | None = None,
    body_type_fixed: str | None = None,
    method:         str    = "harmonic",
    modal_fmax:     float  = 7500.0,
    modal_nmodes:   int    = 400,
    damping:        str    = "rayleigh",
    zeta_const:     float  = 0.01,
    save_modes:     bool   = False,
    on_coverage_fail: str  = "abort",   # "abort" | "flag"
    include_rigid_modes: bool = True,
    residual_flexibility: bool = False,
    harmonic_batch: bool = False,       # opt-in full-harmonic multi-bridge batching
    method_by_body: dict | None = None, # EXPERIMENTAL per-body method override
) -> None:
    base_method = method                # global default; per-shape may override
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes_root  = output_dir / "shapes"
    samples_root = output_dir / "samples"
    manifest_path = output_dir / "manifest.csv"

    shapes_root.mkdir(exist_ok=True)
    samples_root.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)

    # -- Load resume state ----------------------------------------------------
    existing_rows, done_ids = _load_manifest(manifest_path) if resume else ([], set())

    print(f"\n{'='*60}")
    print(f"  dataset_gen  output={output_dir}  n_shapes={n_shapes}")
    print(f"  n_bridge_pts={n_bridge_pts}  n_materials={n_materials}"
          f"  (-> {n_bridge_pts * n_materials} samples/shape)")
    print(f"  mesh={mesh_size_min}-{mesh_size_max}mm  "
          f"freq={freq_min}-{freq_max}Hz  pts={freq_points}  beta={rayleigh_beta:.0e}")
    print(f"  dry_run={dry_run}  no_fem={no_fem}  resume={resume}  seed={seed}")
    print(f"  already done: {len(done_ids)} samples")
    print(f"{'='*60}\n")

    # -- Generate all shapes --------------------------------------------------
    print("Generating shape list ...")
    all_shapes = _make_all_shapes(rng, n_shapes, guitar_ratio)

    sample_idx = 0
    n_done = 0
    n_fail = 0
    n_new = 0   # newly completed in this run (for --max-samples)

    for shape_rec in all_shapes:
        if max_samples > 0 and n_new >= max_samples:
            print(f"\nReached --max-samples {max_samples}. Stopping.")
            break
        sid       = shape_rec["shape_id"]
        shape_dir = shapes_root / f"shape_{sid:04d}"

        print(f"\n{'-'*50}")
        print(f"Shape {sid}/{n_shapes}  type={shape_rec['shape_type']}"
              f"  model={shape_rec['guitar_model'] or '-'}")

        # -- Geometry params for this shape -----------------------------------
        body_type    = body_type_fixed if body_type_fixed else rng.choice(["solid", "hollow"])
        thickness    = float(rng.uniform(40.0, 55.0))
        cavity_ratio = float(rng.uniform(0.5, 0.9)) if body_type == "hollow" else 0.0

        # EXPERIMENTAL method-by-body override (default off): pick the solver per
        # body type, e.g. {"solid":"harmonic","hollow":"modal"}.  Falls back to the
        # global `method` when unset - so default behaviour is identical.
        method = (method_by_body.get(body_type, base_method)
                  if method_by_body else base_method)

        # -- Build STEP + mesh (skip if already done and resuming) ------------
        params_json = shape_dir / "params.json"
        msh_path    = shape_dir / "mesh" / "mesh.msh"

        if resume and params_json.exists() and msh_path.exists():
            print(f"  Shape already built - loading params ...")
            with open(params_json) as f:
                shape_params = json.load(f)
            msh_path = Path(shape_params["msh_path"])
        else:
            if dry_run:
                # Dry-run: fake shape_params for manifest building
                shape_params = {
                    "shape_id": sid,
                    "shape_type": shape_rec["shape_type"],
                    "guitar_model": shape_rec["guitar_model"],
                    "body_type": body_type,
                    "body_thickness": thickness,
                    "cavity_depth_ratio": cavity_ratio,
                    "soundhole_type": "none",
                    "soundhole_diameter": 0.0,
                    "soundhole_center": None,
                    "bridge_pts": [[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]],
                    "bridge_coords": [[0.0, 0.0, thickness],
                                      [10.0, -10.0, thickness],
                                      [-10.0, 10.0, thickness]],
                    "step_path": str(shape_dir / "model" / "model.step"),
                    "msh_path":  str(shape_dir / "mesh" / "mesh.msh"),
                }
            else:
                try:
                    shape_params = _build_shape(
                        shape_rec, body_type, thickness, cavity_ratio,
                        shape_dir, rng, verbose=True,
                        n_bridge_pts=n_bridge_pts,
                        mesh_size_min=mesh_size_min,
                        mesh_size_max=mesh_size_max,
                    )
                    msh_path = Path(shape_params["msh_path"])
                except Exception:
                    err = traceback.format_exc()
                    print(f"  ERROR building shape {sid}:\n{err}")
                    shape_dir.mkdir(parents=True, exist_ok=True)
                    (shape_dir / "error.log").write_text(err)
                    n_fail += 1
                    continue

        bridge_coords_list = shape_params["bridge_coords"]
        snd_type = shape_params["soundhole_type"]
        snd_diam = shape_params["soundhole_diameter"]
        snd_ctr  = shape_params.get("soundhole_center") or [None, None]

        # -- Materials --------------------------------------------------------
        materials = _get_materials(rng, n_materials, material_ranges)

        n_bridges = min(n_bridge_pts, len(bridge_coords_list))
        # Modal batching: K and M depend only on geometry+material (not on the
        # bridge point), so for the modal backend we run ONE eigensolve per
        # material and reuse it across all bridge points of this shape.  The
        # cache is keyed by mat_idx and rebuilt per shape.  Harmonic is unchanged.
        modal_cache: dict[int, dict] = {}
        harmonic_cache: dict[int, dict] = {}    # (shape,material) -> batch result

        for bridge_idx in range(n_bridges):
            bc = tuple(bridge_coords_list[bridge_idx])

            for mat_idx, (mat_name, mat_dict) in enumerate(materials):
                sample_idx += 1
                sample_id  = f"sample_{sample_idx:04d}"
                sample_dir = samples_root / sample_id

                if sample_id in done_ids:
                    print(f"  {sample_id} already done - skip")
                    n_done += 1
                    continue

                # -- Manifest row --------------------------------------------
                row: dict = {
                    "sample_id":    sample_id,
                    "shape_id":     sid,
                    "shape_type":   shape_params["shape_type"],
                    "guitar_model": shape_params["guitar_model"],
                    "body_type":    shape_params["body_type"],
                    "body_thickness":    shape_params["body_thickness"],
                    "cavity_depth_ratio": shape_params["cavity_depth_ratio"],
                    "soundhole_type":     snd_type,
                    "soundhole_diameter": snd_diam,
                    "soundhole_x": snd_ctr[0] if snd_ctr[0] is not None else "",
                    "soundhole_y": snd_ctr[1] if snd_ctr[1] is not None else "",
                    "bridge_x":   bc[0],
                    "bridge_y":   bc[1],
                    "bridge_idx": bridge_idx,
                    **_mat_to_manifest(mat_name, mat_dict),
                    # Defaults so every row records the solver, even the plain
                    # harmonic-single and dry-run/no-fem paths (branches may
                    # overwrite these).
                    "method": method,
                    "harmonic_batch": bool(method == "harmonic" and harmonic_batch),
                    "status": "pending",
                }

                if dry_run:
                    row["status"] = "dry_run"
                    _append_manifest(manifest_path, row)
                    n_done += 1
                    continue

                # -- FEM solve -----------------------------------------------
                t0 = time.time()
                try:
                    if not no_fem and method == "modal":
                        # One eigensolve per material, reused across bridges.
                        if mat_idx not in modal_cache:
                            from run_pipeline import run_modal_batch
                            scratch = shape_dir / "_modal_scratch" / f"mat_{mat_idx}"
                            modal_cache[mat_idx] = run_modal_batch(
                                msh_path=msh_path,
                                material=mat_dict,
                                bridge_points=[list(bridge_coords_list[b])
                                               for b in range(n_bridges)],
                                scratch_dir=scratch,
                                freq_min=freq_min, freq_max=freq_max,
                                freq_points=freq_points,
                                rayleigh_alpha=rayleigh_alpha,
                                rayleigh_beta=rayleigh_beta,
                                modal_fmax=modal_fmax, modal_nmodes=modal_nmodes,
                                damping=damping, zeta_const=zeta_const,
                                save_modes=save_modes,
                                include_rigid_modes=include_rigid_modes,
                                residual_flexibility=residual_flexibility)
                        batch = modal_cache[mat_idx]
                        cov_ok = bool(batch.get("coverage_ok", True))
                        bm = (batch["bridge_meta"][bridge_idx]
                              if bridge_idx < len(batch["bridge_meta"]) else None)
                        # Record modal coverage metadata on the manifest row.
                        row["method"] = "modal"
                        row["modal_fmax"] = modal_fmax
                        row["n_modes_retained"] = batch.get("n_modes_retained", "")
                        row["eig_freq_max_hz"] = round(
                            float(batch.get("eig_freq_max_hz", 0.0)), 1)
                        row["coverage_ok"] = cov_ok
                        if bm and "snap_distance_mm" in bm:
                            row["snap_distance_mm"] = round(bm["snap_distance_mm"], 3)
                        # Coverage gate: do NOT silently mark truncated samples done.
                        if not cov_ok:
                            row["status"] = "coverage_failed"
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            _append_manifest(manifest_path, row)
                            n_fail += 1
                            msg = (f"{sample_id}: modal COVERAGE FAILED "
                                   f"(eig_freq_max={row['eig_freq_max_hz']} Hz < "
                                   f"modal_fmax={modal_fmax} Hz). Increase "
                                   f"--modal-nmodes or lower --modal-fmax.")
                            print(f"  {msg}")
                            if on_coverage_fail == "abort":
                                print(f"\n{'='*60}\nABORTING generation: modal coverage "
                                      f"failed (set --on-coverage-fail flag to continue "
                                      f"and skip instead).\n{'='*60}")
                                return
                            continue
                        Yb = batch["Y_list"][bridge_idx]
                        _save_admittance_npz(sample_dir, batch["freqs"], Yb, bm, cov_ok)
                    elif not no_fem and method == "harmonic" and harmonic_batch:
                        # Full-harmonic batch: ONE assembly + ONE factorization/freq
                        # per material, reused across all bridges of this shape.
                        if mat_idx not in harmonic_cache:
                            from run_pipeline import run_harmonic_batch
                            scratch = shape_dir / "_harmonic_scratch" / f"mat_{mat_idx}"
                            harmonic_cache[mat_idx] = run_harmonic_batch(
                                msh_path=msh_path, material=mat_dict,
                                bridge_points=[list(bridge_coords_list[b])
                                               for b in range(n_bridges)],
                                scratch_dir=scratch,
                                freq_min=freq_min, freq_max=freq_max,
                                freq_points=freq_points,
                                rayleigh_alpha=rayleigh_alpha,
                                rayleigh_beta=rayleigh_beta)
                        batch = harmonic_cache[mat_idx]
                        bm = (batch["bridge_meta"][bridge_idx]
                              if bridge_idx < len(batch["bridge_meta"]) else None)
                        row["method"] = "harmonic"
                        row["harmonic_batch"] = True
                        if bm and "snap_distance_mm" in bm:
                            row["snap_distance_mm"] = round(bm["snap_distance_mm"], 3)
                        Yb = batch["Y_list"][bridge_idx]
                        _save_admittance_npz(sample_dir, batch["freqs"], Yb, bm)
                    elif not no_fem:
                        _run_sample(msh_path, bc, mat_dict, sample_dir,
                                    freq_min=freq_min, freq_max=freq_max,
                                    freq_points=freq_points,
                                    rayleigh_alpha=rayleigh_alpha,
                                    rayleigh_beta=rayleigh_beta,
                                    method=method,
                                    modal_fmax=modal_fmax,
                                    modal_nmodes=modal_nmodes,
                                    damping=damping,
                                    zeta_const=zeta_const,
                                    save_modes=save_modes)

                    # Save sample params.  NOTE: bridge_coords is the REQUESTED
                    # (nominal) bridge point.  The FEM response is computed at the
                    # nearest mesh node; for the modal backend we also record the
                    # snapped node coords and snap distance so a downstream loader
                    # can use bridge_snapped_xyz (or at least track snap error).
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    sp = {
                        "sample_id":    sample_id,
                        "shape_id":     sid,
                        "bridge_idx":   bridge_idx,
                        "bridge_coords": list(bc),       # requested / nominal
                        "material_name": mat_name,
                        "material":     mat_dict,
                        "method":       method,
                        "response_at":  "nearest_mesh_node_to_requested_bridge",
                    }
                    if method == "modal":
                        sp["modal_fmax"] = modal_fmax
                        sp["n_modes_retained"] = batch.get("n_modes_retained")
                        sp["eig_freq_max_hz"] = batch.get("eig_freq_max_hz")
                        sp["coverage_ok"] = cov_ok
                        if bm:
                            sp["bridge_requested_xyz"] = bm.get("bridge_requested_xyz")
                            sp["bridge_snapped_xyz"] = bm.get("bridge_snapped_xyz")
                            sp["snap_distance_mm"] = bm.get("snap_distance_mm")
                    elif method == "harmonic" and harmonic_batch:
                        sp["harmonic_batch"] = True
                        if bm:
                            sp["bridge_requested_xyz"] = bm.get("bridge_requested_xyz")
                            sp["bridge_snapped_xyz"] = bm.get("bridge_snapped_xyz")
                            sp["snap_distance_mm"] = bm.get("snap_distance_mm")
                    with open(sample_dir / "sample_params.json", "w") as f:
                        json.dump(sp, f, indent=2)

                    row["status"] = "done" if not no_fem else "no_fem"
                    elapsed = time.time() - t0
                    print(f"  {sample_id}  bridge={bridge_idx}  mat={mat_name}"
                          f"  {elapsed:.1f}s  OK")
                    n_done += 1
                    n_new  += 1
                    if max_samples > 0 and n_new >= max_samples:
                        _append_manifest(manifest_path, row)
                        print(f"\nReached --max-samples {max_samples}. Stopping.")
                        return

                except Exception:
                    err = traceback.format_exc()
                    row["status"] = "failed"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    (sample_dir / "error.log").write_text(err)
                    print(f"  {sample_id}  FAILED:\n{err[:300]}")
                    n_fail += 1

                _append_manifest(manifest_path, row)

    print(f"\n{'='*60}")
    print(f"  DONE  processed={n_done}  failed={n_fail}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}\n")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="V2 guitar dataset generator")
    p.add_argument("--output-dir", default=None, type=str,
                   help="Output directory (default: ./dataset_output next to this script)")
    p.add_argument("--n-shapes",     default=N_TOTAL_SHAPES, type=int)
    p.add_argument("--guitar-ratio", default=GUITAR_RATIO,   type=float,
                   help="Fraction of shapes from the named guitar model library")
    p.add_argument("--dry-run",  action="store_true",
                   help="Build manifest only, no computation")
    p.add_argument("--no-fem",   action="store_true",
                   help="Build STEP + mesh but skip FEM (Windows-safe)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore existing manifest, restart from scratch")
    p.add_argument("--seed",        default=42,   type=int)
    p.add_argument("--max-samples",  default=0,             type=int,
                   help="Stop after generating this many new samples (0=unlimited)")
    p.add_argument("--n-bridge-pts", default=N_BRIDGE_PTS,  type=int,
                   help="Bridge point positions per shape (default: %(default)s)")
    p.add_argument("--n-materials",  default=N_MATERIALS,   type=int,
                   help="Random material draws per shape (default: %(default)s)")

    # Mesh
    p.add_argument("--mesh-size-min", default=MESH_SIZE_MIN, type=float,
                   help="Gmsh minimum element size [mm] (default: %(default)s)")
    p.add_argument("--mesh-size-max", default=MESH_SIZE_MAX, type=float,
                   help="Gmsh maximum element size [mm] (default: %(default)s)")

    # FEM
    p.add_argument("--freq-min",      default=FREQ_MIN,    type=float,
                   help="Frequency sweep start [Hz] (default: %(default)s)")
    p.add_argument("--freq-max",      default=FREQ_MAX,    type=float,
                   help="Frequency sweep end [Hz] (default: %(default)s)")
    p.add_argument("--freq-points",   default=FREQ_POINTS, type=int,
                   help="Number of frequency points (default: %(default)s)")
    p.add_argument("--rayleigh-alpha", default=RAYLEIGH_ALPHA, type=float,
                   help="Rayleigh mass damping alpha (default: %(default)s)")
    p.add_argument("--rayleigh-beta",  default=RAYLEIGH_BETA,  type=float,
                   help="Rayleigh stiffness damping beta (default: %(default)s)")

    # Solver backend
    p.add_argument("--method", default="harmonic", choices=["harmonic", "modal"],
                   help="Admittance backend (default: %(default)s). 'modal' uses "
                        "the SLEPc eigensolve reconstruction.")
    p.add_argument("--modal-fmax", default=7500.0, type=float,
                   help="Modal: keep modes up to this frequency [Hz]")
    p.add_argument("--modal-nmodes", default=400, type=int,
                   help="Modal: number of lowest eigenpairs to request")
    p.add_argument("--damping", default="rayleigh", choices=["rayleigh", "constant"],
                   help="Modal damping model (default: %(default)s)")
    p.add_argument("--zeta", default=0.01, type=float,
                   help="Constant modal damping ratio (only if --damping constant)")
    p.add_argument("--save-modes", action="store_true",
                   help="Modal: also save modes.npz per sample")
    p.add_argument("--on-coverage-fail", default="abort", choices=["abort", "flag"],
                   help="Modal: if eigensolve doesn't reach modal_fmax, either "
                        "'abort' generation (default) or 'flag' the sample "
                        "(status=coverage_failed) and continue")
    p.add_argument("--modal-no-rigid", action="store_true",
                   help="Modal ABLATION: exclude rigid/weak-spring modes "
                        "(drops free-free low-frequency mass line - worse)")
    p.add_argument("--modal-residual", action="store_true",
                   help="Modal EXPERIMENTAL: residual flexibility (unstable for "
                        "free-free; off by default)")
    p.add_argument("--harmonic-batch", action="store_true",
                   help="Full-harmonic multi-bridge batching (ONE factorization/freq "
                        "shared across bridges). Numerically identical to per-bridge; "
                        "opt-in until server-validated. Default off.")
    p.add_argument("--method-by-body", default=None, type=str,
                   help="EXPERIMENTAL: per-body-type solver as JSON, e.g. "
                        "'{\"solid\":\"harmonic\",\"hollow\":\"modal\"}'. Overrides "
                        "--method per shape. Default off (uses --method everywhere).")

    # Body type
    p.add_argument("--body-type", default=None, choices=["solid", "hollow"],
                   help="Fix body type for all shapes (default: random solid/hollow)")

    # Material ranges
    p.add_argument("--material-ranges", default=None, type=str,
                   help="JSON file or inline JSON to override MATERIAL_RANGES. "
                        "Keys: E1,E2,E3,G12,G13,G23,nu12,nu13,nu23,density. "
                        "Example: '{\"density\": [300, 500]}'")
    return p.parse_args()


DEFAULT_OUTPUT = Path(__file__).parent / "dataset_output"

if __name__ == "__main__":
    import json as _json

    args = _parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT

    # Parse --material-ranges (JSON file path or inline JSON string)
    mat_ranges = None
    if args.material_ranges:
        p = Path(args.material_ranges)
        raw = p.read_text() if p.exists() else args.material_ranges
        override = _json.loads(raw)
        mat_ranges = {**MATERIAL_RANGES, **{k: tuple(v) for k, v in override.items()}}

    run_dataset(
        output_dir      = output_dir,
        n_shapes        = args.n_shapes,
        guitar_ratio    = args.guitar_ratio,
        dry_run         = args.dry_run,
        no_fem          = args.no_fem,
        resume          = args.resume,
        seed            = args.seed,
        max_samples     = args.max_samples,
        n_bridge_pts    = args.n_bridge_pts,
        n_materials     = args.n_materials,
        mesh_size_min   = args.mesh_size_min,
        mesh_size_max   = args.mesh_size_max,
        freq_min        = args.freq_min,
        freq_max        = args.freq_max,
        freq_points     = args.freq_points,
        rayleigh_alpha  = args.rayleigh_alpha,
        rayleigh_beta   = args.rayleigh_beta,
        material_ranges  = mat_ranges,
        body_type_fixed  = args.body_type,
        method           = args.method,
        modal_fmax       = args.modal_fmax,
        modal_nmodes     = args.modal_nmodes,
        damping          = args.damping,
        zeta_const       = args.zeta,
        save_modes       = args.save_modes,
        on_coverage_fail = args.on_coverage_fail,
        include_rigid_modes  = not args.modal_no_rigid,
        residual_flexibility = args.modal_residual,
        harmonic_batch       = args.harmonic_batch,
        method_by_body       = (_json.loads(args.method_by_body)
                                if args.method_by_body else None),
    )
