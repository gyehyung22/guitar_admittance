"""
model_builder.py
----------------
guitar_model.json → CadQuery 3-D solid → output/model.step + output/model_params.json

Usage
-----
python backend/model_builder.py guitar_model.json
python backend/model_builder.py guitar_model.json --output-dir output
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import cadquery as cq

# Add backend/ to path so holes.py / bracing.py are importable
sys.path.insert(0, str(Path(__file__).parent))
from holes import hole_solid

# Thickness of the back plate (z=0 to z=BACK_PLATE_MM).
# Hollow/semi-hollow cavities are cut starting from this z, leaving a solid back plate.
BACK_PLATE_MM = 2.0


# ─── pixel → smooth 2-D contour ───────────────────────────────────────────────

def _rdp(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification (pure numpy, no extra deps)."""
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    d = end - start
    norm = np.linalg.norm(d)
    if norm == 0:
        dists = np.linalg.norm(pts - start, axis=1)
    else:
        dists = np.abs(np.cross(d, start - pts)) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left  = _rdp(pts[:idx+1], epsilon)
        right = _rdp(pts[idx:],   epsilon)
        return np.vstack([left[:-1], right])
    return np.array([start, end])


def _fit_spline_uniform(pts: np.ndarray, n_points: int = 120) -> np.ndarray:
    """Re-sample pts to n_points uniformly spaced along the arc."""
    from scipy.interpolate import splprep, splev
    pts_c = np.vstack([pts, pts[0]])          # close the loop
    tck, _ = splprep([pts_c[:, 0], pts_c[:, 1]], s=0, per=True, k=3)
    u = np.linspace(0, 1, n_points, endpoint=False)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def pixels_to_contour(
    pixel_list: list,
    canvas_px: int,
    mm_per_px: float,
    rdp_epsilon: float = 3.0,
    n_spline_pts: int = 120,
) -> np.ndarray:
    """
    Convert a list of [px, py] body pixels → smooth closed contour in mm.

    Returns ndarray of shape (n_spline_pts, 2) in mm coordinates.
    """
    bitmap = np.zeros((canvas_px, canvas_px), dtype=np.uint8)
    for px, py in pixel_list:
        if 0 <= px < canvas_px and 0 <= py < canvas_px:
            bitmap[py, px] = 255

    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour found in body_pixels.")
    contour = max(contours, key=cv2.contourArea).squeeze()   # (N, 2)

    simplified = _rdp(contour.astype(float), rdp_epsilon)
    smooth_mm  = _fit_spline_uniform(simplified, n_spline_pts) * mm_per_px

    # Center: shift so centroid is at origin (CadQuery convention)
    cx, cy = smooth_mm.mean(axis=0)
    smooth_mm -= np.array([cx, cy])

    return smooth_mm, np.array([cx, cy])   # (contour_mm, centroid_px_in_mm)


def pixels_to_hole_contour(
    pixel_list: list,
    canvas_px: int,
    mm_per_px: float,
    body_centroid_mm: np.ndarray,
    rdp_epsilon: float = 2.0,
    n_spline_pts: int = 60,
) -> np.ndarray:
    """Convert user-defined hole pixels → smooth contour in body-centred mm coords."""
    pts_mm, _ = pixels_to_contour(pixel_list, canvas_px, mm_per_px,
                                  rdp_epsilon, n_spline_pts)
    # pts_mm is relative to hole centroid; shift to body frame
    hole_centroid_px = np.mean(pixel_list, axis=0) * mm_per_px
    offset = hole_centroid_px - body_centroid_mm
    return pts_mm + offset


# ─── body outline wire ────────────────────────────────────────────────────────

def _contour_to_wire(contour_mm: np.ndarray) -> cq.Wire:
    """Build a closed CadQuery Wire from an (N,2) contour array."""
    pts = [(float(x), float(y), 0.0) for x, y in contour_mm]
    pts.append(pts[0])   # close
    return cq.Wire.makeSpline([cq.Vector(*p) for p in pts], periodic=False)


# ─── solid body ───────────────────────────────────────────────────────────────

def build_solid_flat(contour_mm: np.ndarray, thickness_mm: float) -> cq.Workplane:
    """Simple extrusion — solid flat-top body."""
    pts = [(float(x), float(y)) for x, y in contour_mm]
    return (cq.Workplane("XY")
            .spline(pts, periodic=True, includeCurrent=False)
            .close()
            .extrude(thickness_mm))


def _sphere_cap(
    pts: list,
    arch_height: float,
    span: float,
    base_z: float,
    direction: float,
) -> cq.Workplane:
    """
    Return a convex spherical cap solid.

    The cap protrudes by `arch_height` mm beyond `base_z` in `direction`
    (+1 = upward / top arch,  -1 = downward / back arch).
    It is clipped to the body outline prism so no material extends past the edges.

    Geometry derivation
    -------------------
    Sphere of radius R centered at z_c.
    At the outline edge (r = span): cap surface must be at z = base_z.
    At the body center  (r = 0):    cap surface is at z = base_z + direction*arch_height.

    Solving:   R = (span² + arch_height²) / (2 * arch_height)
               z_c = base_z + direction * (arch_height - R)
                   = base_z - direction * (R - arch_height)
    """
    R    = (span ** 2 + arch_height ** 2) / (2.0 * arch_height)
    z_c  = base_z - direction * (R - arch_height)

    sphere = cq.Workplane("XY").workplane(offset=z_c).sphere(R)

    # Clip the sphere to the region that protrudes beyond base_z
    margin = arch_height + 1.0
    if direction > 0:
        clip_z0 = base_z - 0.05          # tiny overlap avoids zero-thickness seam
        clip_h  = arch_height + 0.1
    else:
        clip_z0 = base_z - margin
        clip_h  = margin + 0.05

    clip = (cq.Workplane("XY")
            .workplane(offset=clip_z0)
            .spline(pts, periodic=True, includeCurrent=False)
            .close()
            .extrude(clip_h))

    cap = sphere.intersect(clip)

    # Trim the tiny overlap below base_z to get a clean base face
    if direction > 0 and clip_z0 < base_z:
        trim = (cq.Workplane("XY")
                .workplane(offset=clip_z0)
                .spline(pts, periodic=True, includeCurrent=False)
                .close()
                .extrude(base_z - clip_z0))
        cap = cap.cut(trim)

    return cap


def build_solid_archtop(
    contour_mm: np.ndarray,
    thickness_mm: float,
    top_arch_height_mm: float,
    back_arch_height_mm: float,
) -> cq.Workplane:
    """
    Archtop guitar body built from a flat extrusion + spherical-cap domes.

    Top face  (z = thickness): convex dome rising +top_arch_height_mm upward.
    Back face (z = 0):         convex dome dropping -back_arch_height_mm downward.

    The spherical-cap approach preserves the correct outline silhouette at the
    waist and provides a physically realistic arched surface for FEM.
    """
    pts  = [(float(x), float(y)) for x, y in contour_mm]
    xmin, xmax = contour_mm[:, 0].min(), contour_mm[:, 0].max()
    ymin, ymax = contour_mm[:, 1].min(), contour_mm[:, 1].max()
    span = max(xmax - xmin, ymax - ymin) / 2.0

    # ── flat base ──────────────────────────────────────────────────────────────
    body = (cq.Workplane("XY")
            .spline(pts, periodic=True, includeCurrent=False)
            .close()
            .extrude(thickness_mm))

    # ── top arch (convex upward) ──────────────────────────────────────────────
    if top_arch_height_mm > 0:
        cap = _sphere_cap(pts, top_arch_height_mm, span,
                          base_z=thickness_mm, direction=+1)
        body = body.union(cap)

    # ── back arch (convex downward) ───────────────────────────────────────────
    if back_arch_height_mm > 0:
        cap = _sphere_cap(pts, back_arch_height_mm, span,
                          base_z=0.0, direction=-1)
        body = body.union(cap)

    return body


# ─── hollow / semi-hollow cavities ───────────────────────────────────────────

def _inset_contour(contour_mm: np.ndarray, inset_mm: float) -> np.ndarray:
    """
    Shrink contour by a uniform inset_mm using a true parallel offset.

    Uses shapely.buffer (accurate for non-convex guitar shapes).
    Falls back to centroid scaling if shapely is unavailable.
    """
    try:
        from shapely.geometry import Polygon
        poly = Polygon(contour_mm)
        inner_poly = poly.buffer(-inset_mm, join_style=2, mitre_limit=5.0)
        if inner_poly.is_empty or not inner_poly.is_valid:
            raise ValueError("Inset makes contour empty — wall_thickness too large.")
        coords = np.array(inner_poly.exterior.coords)[:-1]  # drop duplicate closing pt
        return coords
    except ImportError:
        # Fallback: centroid-based uniform scaling
        cx, cy = contour_mm.mean(axis=0)
        centered = contour_mm - [cx, cy]
        max_r = np.linalg.norm(centered, axis=1).max()
        scale = max(0.05, 1.0 - inset_mm / max_r)
        return centered * scale + [cx, cy]


def _cavity_depth(
    thickness_mm: float,
    cavity_depth_ratio: float,
    top_type: str,
    top_arch_height_mm: float,
    back_arch_height_mm: float,
) -> float:
    """
    Cavity depth measured from z=0 (back face) upward.

    For archtop, the total body height = thickness + top_arch + back_arch,
    but the cavity is still measured from the flat back (z=0) so that the
    top-plate thickness (from cavity floor to top surface) is preserved.

        cavity_depth = (thickness + back_arch) * cavity_depth_ratio

    This keeps the top-plate thickness = thickness * (1 - ratio) + top_arch,
    which is physically reasonable for a carved archtop.
    """
    effective_height = thickness_mm
    if top_type == "archtop":
        effective_height += back_arch_height_mm   # back arch extends below z=0
    return effective_height * cavity_depth_ratio


def apply_hollow_cavity(
    body: cq.Workplane,
    contour_mm: np.ndarray,
    thickness_mm: float,
    cavity_depth_ratio: float,
    wall_thickness_mm: float,
    top_type: str = "flat",
    top_arch_height_mm: float = 0.0,
    back_arch_height_mm: float = 0.0,
) -> cq.Workplane:
    """
    Cut hollow cavity from z=BACK_PLATE_MM upward (back plate is preserved).

    Side walls:  uniform `wall_thickness_mm` via parallel offset.
    Back plate:  solid slab from z=0 to z=BACK_PLATE_MM (2 mm).
    Top plate:   remaining solid above cavity ceiling.
    """
    inner = _inset_contour(contour_mm, wall_thickness_mm)
    depth = _cavity_depth(thickness_mm, cavity_depth_ratio,
                          top_type, top_arch_height_mm, back_arch_height_mm)
    pts = [(float(x), float(y)) for x, y in inner]
    # Start cavity at BACK_PLATE_MM so back plate remains solid
    cavity = (cq.Workplane("XY")
              .workplane(offset=BACK_PLATE_MM)
              .spline(pts, periodic=True, includeCurrent=False)
              .close()
              .extrude(depth - BACK_PLATE_MM))
    body = body.cut(cavity)

    top_plate_mm = thickness_mm - depth + back_arch_height_mm
    print(f"  Cavity depth={depth:.1f} mm  top-plate={top_plate_mm:.1f} mm  "
          f"back-plate={BACK_PLATE_MM:.1f} mm  side-wall={wall_thickness_mm:.1f} mm")
    return body


def apply_semi_hollow_cavity(
    body: cq.Workplane,
    contour_mm: np.ndarray,
    thickness_mm: float,
    cavity_depth_ratio: float,
    center_block_width_mm: float,
    wall_thickness_mm: float,
    top_type: str = "flat",
    top_arch_height_mm: float = 0.0,
    back_arch_height_mm: float = 0.0,
) -> cq.Workplane:
    """
    Semi-hollow: hollow cavity + solid center block running neck-to-tail.

    Strategy
    --------
    1. Cut the full inner cavity.
    2. Union back a center block of width `center_block_width_mm` (X axis)
       spanning the full inner Y range at the same depth.
    """
    inner = _inset_contour(contour_mm, wall_thickness_mm)
    depth = _cavity_depth(thickness_mm, cavity_depth_ratio,
                          top_type, top_arch_height_mm, back_arch_height_mm)
    pts = [(float(x), float(y)) for x, y in inner]

    # Full inner cavity — starts at BACK_PLATE_MM to preserve back plate
    cavity = (cq.Workplane("XY")
              .workplane(offset=BACK_PLATE_MM)
              .spline(pts, periodic=True, includeCurrent=False)
              .close()
              .extrude(depth - BACK_PLATE_MM))
    body = body.cut(cavity)

    # Center block: neck-to-tail (Y span of inner contour), limited width (X)
    # Starts at BACK_PLATE_MM so it sits on top of the back plate
    ymin = float(inner[:, 1].min())
    ymax = float(inner[:, 1].max())
    block_height = ymax - ymin + 2.0   # +2 mm to ensure clean intersection
    cavity_h = depth - BACK_PLATE_MM
    center_block = (cq.Workplane("XY")
                    .workplane(offset=BACK_PLATE_MM)
                    .transformed(offset=cq.Vector(0.0, (ymin + ymax) / 2.0, 0.0))
                    .rect(center_block_width_mm, block_height)
                    .extrude(cavity_h))

    # Clip center block to the inner contour shape
    inner_solid = (cq.Workplane("XY")
                   .workplane(offset=BACK_PLATE_MM)
                   .spline(pts, periodic=True, includeCurrent=False)
                   .close()
                   .extrude(cavity_h))
    block_clipped = center_block.intersect(inner_solid)
    body = body.union(block_clipped)

    top_plate_mm = thickness_mm - depth + back_arch_height_mm
    print(f"  Cavity depth={depth:.1f} mm  top-plate={top_plate_mm:.1f} mm  "
          f"side-wall={wall_thickness_mm:.1f} mm  center-block={center_block_width_mm:.1f} mm")
    return body


# ─── internal air cavity solid (for conformal air-coupling mesh) ──────────────

def build_air_cavity_solid(
    contour_mm: np.ndarray,
    thickness_mm: float,
    cavity_depth_ratio: float,
    wall_thickness_mm: float,
    body_type: str = "hollow",
    center_block_width_mm: float = 40.0,
    top_type: str = "flat",
    top_arch_height_mm: float = 0.0,
    back_arch_height_mm: float = 0.0,
) -> "cq.Workplane | None":
    """Build the EXACT internal air cavity solid (the negative space of the wood).

    This is the same geometry the structural builder CUTS OUT of the wood
    (apply_hollow_cavity), exported separately so the conformal air mesh has a
    real, watertight AIR_INTERNAL volume instead of reverse-engineering it from
    the wood STEP (which is unstable — see docs/air_coupling_theory.md §Domains).

    The air volume = cavity region z ∈ [BACK_PLATE_MM, cavity_floor]; for
    semi-hollow the center block is subtracted (it is wood, not air).  The
    soundhole NECK is NOT included here — the default lumped impedance port models
    the neck inertance analytically, so AIR_INTERNAL is the cavity only.

    Returns the cq.Workplane air solid, or None for a solid body (no cavity).
    """
    if body_type not in ("hollow", "semi-hollow", "semi_hollow"):
        return None

    inner = _inset_contour(contour_mm, wall_thickness_mm)
    depth = _cavity_depth(thickness_mm, cavity_depth_ratio,
                          top_type, top_arch_height_mm, back_arch_height_mm)
    cavity_h = depth - BACK_PLATE_MM
    if cavity_h <= 0:
        return None
    pts = [(float(x), float(y)) for x, y in inner]
    air = (cq.Workplane("XY")
           .workplane(offset=BACK_PLATE_MM)
           .spline(pts, periodic=True, includeCurrent=False)
           .close()
           .extrude(cavity_h))

    if body_type in ("semi-hollow", "semi_hollow"):
        # Subtract the solid center block (wood) so air excludes it.
        ymin = float(inner[:, 1].min()); ymax = float(inner[:, 1].max())
        block_height = ymax - ymin + 2.0
        center_block = (cq.Workplane("XY")
                        .workplane(offset=BACK_PLATE_MM)
                        .transformed(offset=cq.Vector(0.0, (ymin + ymax) / 2.0, 0.0))
                        .rect(center_block_width_mm, block_height)
                        .extrude(cavity_h))
        inner_solid = (cq.Workplane("XY")
                       .workplane(offset=BACK_PLATE_MM)
                       .spline(pts, periodic=True, includeCurrent=False)
                       .close()
                       .extrude(cavity_h))
        block_clipped = center_block.intersect(inner_solid)
        air = air.cut(block_clipped)

    return air


# ─── bracing ─────────────────────────────────────────────────────────────────

def apply_bracing(
    body: cq.Workplane,
    brace_type: str,
    contour_mm: np.ndarray,
    thickness_mm: float,
    cavity_depth_ratio: float,
) -> cq.Workplane:
    if brace_type == "none":
        return body
    try:
        from bracing import get_bracing_solids
        solids = get_bracing_solids(
            brace_type, contour_mm, thickness_mm, cavity_depth_ratio,
            back_plate_mm=BACK_PLATE_MM,
        )
        for s in solids:
            body = body.union(s)
    except ImportError:
        print("WARNING: bracing.py not found — skipping bracing.")
    return body


# ─── main builder ─────────────────────────────────────────────────────────────

def build_model(data: dict, output_dir: Path, export_air: bool = False) -> dict:
    """
    Build the 3-D guitar body from guitar_model.json data.

    Returns model_params dict (bridge coords, bounding box, etc.).

    If `export_air` is True and the body is hollow/semi-hollow, ALSO export the
    internal air cavity solid to output_dir/model_air.step and record the air
    geometry (cavity floor z, soundhole center/radius, top-plate thickness) under
    model_params["air"].  This is OFF by default and never alters the wood STEP.
    """
    canvas_px  = data["canvas_px"]
    canvas_mm  = data["canvas_mm"]
    mm_per_px  = canvas_mm / canvas_px
    p          = data["params"]

    body_type          = p["body_type"]
    top_type           = p["top_type"]
    thickness          = float(p["body_thickness"])
    cavity_ratio       = float(p["cavity_depth_ratio"])
    center_block_w     = float(p.get("center_block_width", 40.0))
    wall_thickness     = float(p.get("wall_thickness", 5.0))
    hole_type          = p["hole_type"]
    hole_params        = p["hole_params"]
    bracing            = p.get("bracing", "none")
    top_arch_h         = float(p.get("top_arch_height", 0.0))
    back_arch_h        = float(p.get("back_arch_height", 0.0))

    # ── Step 1: exact contour (preferred) or legacy raster conversion ──
    direct_contour = data.get("body_contour_mm")
    if direct_contour is not None:
        contour_mm = np.asarray(direct_contour, dtype=float)
        if (contour_mm.ndim != 2 or contour_mm.shape[0] < 4
                or contour_mm.shape[1] != 2
                or not np.all(np.isfinite(contour_mm))):
            raise ValueError("body_contour_mm must be a finite (N>=4, 2) array")
        if np.linalg.norm(contour_mm[0] - contour_mm[-1]) < 1e-9:
            contour_mm = contour_mm[:-1]
        centroid_mm = np.zeros(2, dtype=float)
        print(f"[1/5] Using direct body contour ({len(contour_mm)} points) ...")
    else:
        print("[1/5] Converting body pixels to smooth contour ...")
        contour_mm, centroid_mm = pixels_to_contour(
            data["body_pixels"], canvas_px, mm_per_px
        )

    # ── Step 2: build base solid ─────────────────────────────────────────────
    print(f"[2/5] Building {body_type} / {top_type} body ({thickness} mm) ...")
    if top_type == "flat":
        body = build_solid_flat(contour_mm, thickness)
    else:
        body = build_solid_archtop(contour_mm, thickness, top_arch_h, back_arch_h)

    # ── Step 3: hollow cavity ────────────────────────────────────────────────
    _cavity_kwargs = dict(
        wall_thickness_mm   = wall_thickness,
        top_type            = top_type,
        top_arch_height_mm  = top_arch_h,
        back_arch_height_mm = back_arch_h,
    )
    if body_type == "hollow":
        print("[3/5] Applying hollow cavity ...")
        body = apply_hollow_cavity(body, contour_mm, thickness, cavity_ratio, **_cavity_kwargs)
    elif body_type == "semi-hollow":
        print("[3/5] Applying semi-hollow cavity + center block ...")
        body = apply_semi_hollow_cavity(
            body, contour_mm, thickness, cavity_ratio, center_block_w, **_cavity_kwargs)
    else:
        print("[3/5] Solid body - no cavity.")

    # ── Step 4: sound hole ───────────────────────────────────────────────────
    print(f"[4/5] Sound hole: {hole_type}")
    # Captured for the optional air-cavity export (body-centred mm, radius).
    soundhole_center_body = (0.0, 0.0)
    soundhole_radius_mm = 0.0
    cavity_floor_z_mm = 0.0
    if hole_type != "none":
        # hole_center in body-centred mm coords
        if data.get("hole_center_body_mm") is not None:
            hole_center_body = tuple(np.asarray(data["hole_center_body_mm"], float))
        elif data.get("hole_center_mm"):
            hc_px_mm = np.array(data["hole_center_mm"])
            hole_center_body = tuple(hc_px_mm - centroid_mm)
        else:
            hole_center_body = (0.0, 0.0)
        soundhole_center_body = (float(hole_center_body[0]), float(hole_center_body[1]))
        if hole_type == "round":
            soundhole_radius_mm = float(hole_params.get("round", {}).get("diameter", 0.0)) / 2.0

        user_contour = None
        if hole_type == "user-defined" and data.get("soundhole_pixels"):
            user_contour = pixels_to_hole_contour(
                data["soundhole_pixels"], canvas_px, mm_per_px, centroid_mm
            ).tolist()

        # For hollow / semi-hollow bodies the sound hole must pierce the TOP
        # plate only, leaving the back plate intact.  Restrict the cut solid to
        # z ∈ [cavity_floor, thickness] (cavity floor = cavity depth).  Solid
        # bodies keep the default through-cut.
        z_start = z_height = None
        if body_type in ("hollow", "semi-hollow"):
            depth = _cavity_depth(thickness, cavity_ratio, top_type,
                                  top_arch_h, back_arch_h)
            cavity_floor_z_mm = float(depth)
            margin = 1.0                       # overshoot top surface for clean cut
            z_start  = max(BACK_PLATE_MM, depth - margin)
            z_height = (thickness - z_start) + margin
            print(f"  Top-plate-only cut: z ∈ [{z_start:.1f}, {thickness:.1f}] mm "
                  f"(cavity floor={depth:.1f}, back plate {BACK_PLATE_MM:.1f} preserved)")

        hs = hole_solid(hole_type, hole_params, hole_center_body, thickness,
                        user_contour, z_start=z_start, z_height=z_height)
        if hs is not None:
            body = body.cut(hs)

    # ── Step 5: bracing ──────────────────────────────────────────────────────
    print(f"[5/5] Bracing: {bracing}")
    if body_type in ("hollow", "semi-hollow"):
        body = apply_bracing(body, bracing, contour_mm, thickness, cavity_ratio)

    # ── Export STEP ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    step_path = output_dir / "model.step"
    cq.exporters.export(body, str(step_path))
    print(f"\nSTEP written: {step_path}")

    # ── Bridge coords in body frame ───────────────────────────────────────────
    if data.get("bridge_point_body_mm") is not None:
        bridge_body_xy = np.asarray(data["bridge_point_body_mm"], dtype=float)
    else:
        bp_px_mm = np.array(data["bridge_point_mm"])       # in canvas mm
        bridge_body_xy = bp_px_mm - centroid_mm            # centred coords

    # For archtop: bridge sits on the dome surface, not the flat top face.
    # The dome height at radial distance r from center:
    #   z(r) = thickness + sqrt(R² - r²) - (R - top_arch)
    # At r=0 (center): z = thickness + top_arch
    # The bridge is typically near the body center, so use the arch apex.
    if top_type == "archtop" and p.get("top_arch_height", 0) > 0:
        bx, by = float(bridge_body_xy[0]), float(bridge_body_xy[1])
        top_arch_h = float(p["top_arch_height"])
        xmin = float(contour_mm[:,0].min()); xmax = float(contour_mm[:,0].max())
        ymin = float(contour_mm[:,1].min()); ymax = float(contour_mm[:,1].max())
        span = max(xmax - xmin, ymax - ymin) / 2.0
        R = (span**2 + top_arch_h**2) / (2.0 * top_arch_h)
        r2 = bx**2 + by**2
        dome_z = thickness + top_arch_h - (R - math.sqrt(max(R**2 - r2, 0.0)))
        bridge_z = float(dome_z)
    else:
        bridge_z = float(thickness)

    xmin = float(contour_mm[:,0].min())
    xmax = float(contour_mm[:,0].max())
    ymin = float(contour_mm[:,1].min())
    ymax = float(contour_mm[:,1].max())

    model_params = {
        "bridge_coords":   [float(bridge_body_xy[0]), float(bridge_body_xy[1]), bridge_z],
        "body_type":       body_type,
        "top_type":        top_type,
        "body_thickness":  thickness,
        "cavity_depth_ratio": cavity_ratio,
        "bounding_box_mm": [xmin, xmax, ymin, ymax, 0.0, thickness],
        "step_file":       str(step_path),
        "geometry_input":  "direct_contour" if direct_contour is not None else "raster",
        "actual_contour_mm": contour_mm.tolist(),
    }

    # ── Optional: export the internal air cavity solid (conformal air mesh) ────
    if export_air and body_type in ("hollow", "semi-hollow", "semi_hollow"):
        try:
            air = build_air_cavity_solid(
                contour_mm, thickness, cavity_ratio, wall_thickness,
                body_type=body_type, center_block_width_mm=center_block_w,
                top_type=top_type, top_arch_height_mm=top_arch_h,
                back_arch_height_mm=back_arch_h,
            )
            if air is not None:
                air_step = output_dir / "model_air.step"
                cq.exporters.export(air, str(air_step))
                try:
                    cavity_volume_mm3 = float(air.val().Volume())
                except Exception:
                    cavity_volume_mm3 = None
                # Cavity floor depends only on the cavity, NOT on the soundhole.
                # Compute it here so hollow bodies with hole_type="none" still get
                # the correct floor/top-plate (the soundhole block may not have run).
                cavity_floor_z_mm = float(_cavity_depth(
                    thickness, cavity_ratio, top_type, top_arch_h, back_arch_h))
                top_plate_mm = thickness - cavity_floor_z_mm  # = t_hole (neck length)
                # v5: if an explicit top-plate thickness was requested, the CAD cavity
                # MUST realise it (cavity_ratio is derived from it upstream).  Fail loud.
                requested_top = p.get("top_plate_thickness_mm")
                if (requested_top is not None
                        and abs(top_plate_mm - float(requested_top)) > 1e-6):
                    raise RuntimeError(
                        f"CAD top-plate {top_plate_mm:.6f} mm != requested "
                        f"{float(requested_top):.6f} mm (thickness={thickness}, "
                        f"cavity_ratio={cavity_ratio})")
                model_params["air"] = {
                    "has_internal_air": True,
                    "air_step_file": str(air_step),
                    "cavity_floor_z_mm": cavity_floor_z_mm,
                    "back_plate_z_mm": float(BACK_PLATE_MM),
                    "top_plate_thickness_mm": float(top_plate_mm),
                    "soundhole_center_body_mm": list(soundhole_center_body),
                    "soundhole_radius_mm": float(soundhole_radius_mm),
                    "cavity_volume_mm3": cavity_volume_mm3,
                    "wall_thickness_mm": float(wall_thickness),
                }
                print(f"Air STEP written: {air_step}  "
                      f"(cavity floor z={cavity_floor_z_mm:.1f}mm, "
                      f"top plate={top_plate_mm:.1f}mm)")
        except Exception as exc:
            print(f"WARNING: air cavity export failed ({exc}); "
                  f"continuing with wood STEP only.")
            model_params["air"] = {"has_internal_air": False, "error": str(exc)}

    params_path = output_dir / "model_params.json"
    with open(params_path, "w") as f:
        json.dump(model_params, f, indent=2)
    print(f"Params written: {params_path}")

    return model_params


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="guitar_model.json → STEP")
    parser.add_argument("json_file", type=Path, help="Path to guitar_model.json")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"ERROR: {args.json_file} not found"); sys.exit(1)

    with open(args.json_file) as f:
        data = json.load(f)

    params = build_model(data, args.output_dir)

    print("\n" + "="*55)
    print("Model build complete.")
    print(f"  Bridge coords : {params['bridge_coords']}")
    print(f"  Bounding box  : {params['bounding_box_mm']}")
    print(f"  STEP          : {params['step_file']}")
    print("="*55)
    print("\nNext step:")
    print(f"  python run_pipeline.py {params['step_file']} \\")
    bc = params['bridge_coords']
    print(f"    --bridge {bc[0]:.1f} {bc[1]:.1f} {bc[2]:.1f} \\")
    if params.get("body_type") in ("hollow", "semi_hollow", "semi-hollow"):
        print(f"    --body-thickness {params['body_thickness']:.3f} \\")
        print(f"    --cavity-depth-ratio {params['cavity_depth_ratio']:.6f} \\")
    print(f"    --material engelmann_spruce")


if __name__ == "__main__":
    main()
