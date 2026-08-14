"""Build a CONTROLLED anchor dataset plan on top of mixed-v6's geometry.

Why this exists
---------------
mixed-v5 and mixed-v6 share a seed, so they carry the SAME 100 contours and
byte-identical thickness / top-plate / cavity values; only their materials and
bridge points differ.  Combining them therefore adds nothing on the axis that
actually predicts validation error -- nearest-neighbour distance in the geometry
scalars (partial Spearman ~0.3-0.45 after every control) -- while the material
axis it does extend is uncorrelated with error (rho ~ 0.0).

Worse, in v6 the size scalars (width, height, area) are DERIVED from the
contour, so "more contours" and "wider geometry coverage" cannot be separated
within that design.  Three blocks fix that:

  C  scalar density   the same contours, resampled thickness / top plate /
                      cavity / soundhole.  New coverage on the predictive axis,
                      no new outline and no rescaling -- the cheapest cases
                      available.
  A  scale isolation  the same contour at several scales.  Geometric similarity
                      makes the expected shift exact (f ~ 1/s), so this is a
                      test of the model as well as a coverage probe, and it is
                      the ONLY way to move size without changing the outline.
  B  contour isolation different outlines held at one characteristic length,
                      with everything else anchored.  Contour identity is still
                      an open question -- the evidence so far says the model
                      does not USE it, not that it does not matter.

Two anchor materials, byte-identical on every shape, so material is a fixed
effect rather than another source of variation.  Bridge points are shared in
NORMALISED coordinates, without which a comparison across scales would be
comparing different drive points as well as different sizes.

Nothing here modifies dataset_gen_mixed.py or any other fingerprinted source --
it imports them.  mixed-v6 therefore stays resumable: stop it, generate this,
resume it later, and its plan hash and implementation fingerprint still match.

Splits are INHERITED from mixed-v6 by base_shape_id.  A contour that is a v6
validation body stays a validation body here; anything else would put the same
geometry on both sides of the split the moment the two datasets are trained on
together, which is exactly what nn_dataset's split-consistency check refuses.

  python -u plan_anchor_dataset.py --source results/mixed_v6_full \
      --out results/mixed_anchor_v1 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import dataset_gen_mixed as dg
from placement_utils import random_bridge_points, random_soundhole


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Lengths the CAD builder holds FIXED in millimetres, whatever the body size:
# the cavity side wall (dataset_gen.py "wall_thickness"), the back plate
# (backend/model_builder.BACK_PLATE_MM) and the centre block
# (dataset_gen.py "center_block_width").  A scaled HOLLOW body is therefore not
# geometrically similar to its source -- only the solid one is.  Block A is
# still valid coverage either way, but the exact 1/scale prediction belongs to
# its solid half alone.
FIXED_INTERNAL_LENGTHS_MM = {"wall_thickness": 5.0, "back_plate": 2.0,
                             "center_block_width": 40.0}


def characteristic_length(contour: np.ndarray) -> float:
    """sqrt(plan area) -- one number for "how big", independent of aspect."""
    x, y = np.asarray(contour, float).T
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return float(np.sqrt(max(area, 1e-12)))


def frame(contour: np.ndarray):
    pts = np.asarray(contour, float)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]) * 1.1)
    return centre, span


def scaled(contour: np.ndarray, factor: float) -> np.ndarray:
    """Scale about the bounding-box centre, so the frame stays concentric."""
    pts = np.asarray(contour, float)
    centre, _span = frame(pts)
    return (pts - centre) * float(factor) + centre


def to_normalised(points, contour) -> np.ndarray:
    centre, span = frame(contour)
    return (np.asarray(points, float)[:, :2] - centre) / (span / 2.0)


def from_normalised(points, contour) -> np.ndarray:
    centre, span = frame(contour)
    return np.asarray(points, float) * (span / 2.0) + centre


def distance_to_boundary(points, contour) -> np.ndarray:
    pts = np.asarray(points, float)[:, :2]
    poly = np.asarray(contour, float)
    start, end = poly, np.roll(poly, -1, axis=0)
    edge = end - start
    length2 = np.maximum((edge ** 2).sum(1), 1e-30)
    delta = pts[:, None, :] - start[None, :, :]
    t = np.clip((delta * edge[None, :, :]).sum(-1) / length2[None, :], 0.0, 1.0)
    closest = start[None, :, :] + t[:, :, None] * edge[None, :, :]
    return np.linalg.norm(pts[:, None, :] - closest, axis=-1).min(1)


def self_intersects(contour: np.ndarray) -> bool:
    """Does the closed polygon cross itself?  Scaling about the centre cannot
    introduce a crossing, but a future edit to the contour handling could."""
    pts = np.asarray(contour, float)
    n = len(pts)

    def crosses(a, b, c, d):
        def side(p, q, r):
            return np.sign((q[0] - p[0]) * (r[1] - p[1])
                           - (q[1] - p[1]) * (r[0] - p[0]))
        return (side(a, b, c) * side(a, b, d) < 0
                and side(c, d, a) * side(c, d, b) < 0)

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        # Skip the neighbours, which share an endpoint by construction.
        for j in range(i + 2, n - (1 if i == 0 else 0)):
            if crosses(a, b, pts[j], pts[(j + 1) % n]):
                return True
    return False


def inside(points, contour) -> np.ndarray:
    from matplotlib.path import Path as MplPath
    poly = np.asarray(contour, float)
    return MplPath(np.vstack([poly, poly[0]])).contains_points(
        np.asarray(points, float)[:, :2])


def cavity_ratio_from(thickness: float, top_plate: float) -> float:
    """The generator's own relation between the cavity and the plates.

    The CAD builder derives the top plate back out of ``cavity_ratio``, and it
    rejects the body when the two disagree, so getting this wrong fails every
    hollow case at the geometry stage with a message about a plate twice the
    requested thickness.  ``verify_cavity_relation`` pins it to the source plan
    rather than to a comment.
    """
    return float(np.clip(1.0 - top_plate / max(thickness, 1e-9), 0.0, 0.99))


def verify_cavity_relation(source_plan) -> tuple:
    """Check the relation against every source shape; return the sampled bands.

    A hard failure here is better than 87 shapes of hollow FEM failing one at a
    time: the check costs nothing and the mistake costs days.
    """
    thickness = np.array([s["thickness"] for s in source_plan["base_shapes"]])
    plate = np.array([s["top_plate_thickness_mm"]
                      for s in source_plan["base_shapes"]])
    stored = np.array([s["cavity_ratio"] for s in source_plan["base_shapes"]])
    predicted = np.array([cavity_ratio_from(t, p)
                          for t, p in zip(thickness, plate)])
    error = float(np.abs(predicted - stored).max())
    if error > 1e-9:
        raise SystemExit(
            f"cavity_ratio relation does not reproduce the source plan "
            f"(max error {error:.3g}). The generator's convention changed; fix "
            f"cavity_ratio_from before generating anything.")
    ratio = plate / thickness
    print(f"cavity relation verified against {len(stored)} source shapes "
          f"(max error {error:.1e})")
    print(f"  source bands: thickness {thickness.min():.1f}-{thickness.max():.1f} mm, "
          f"top plate {plate.min():.2f}-{plate.max():.2f} mm, "
          f"plate/thickness {ratio.min():.4f}-{ratio.max():.4f}")
    return ((float(thickness.min()), float(thickness.max())),
            (float(plate.min()), float(plate.max())),
            (float(ratio.min()), float(ratio.max())))


# ---------------------------------------------------------------------------
# Shape construction
# ---------------------------------------------------------------------------

def place_bridges(contour, template_norm, min_edge_mm: float, seed: int,
                  n_points: int):
    """Bridges at SHARED normalised positions, or a fresh draw if they do not fit.

    Sharing them is what makes a cross-scale or cross-contour comparison a
    comparison of geometry rather than of drive points.  A template point that
    lands outside a particular outline, or too near its edge, invalidates that
    body rather than being nudged: a silently moved bridge would break the
    very anchoring the block exists to provide.
    """
    if template_norm is not None:
        candidate = from_normalised(template_norm, contour)
        ok = inside(candidate, contour)
        clear = distance_to_boundary(candidate, contour) >= float(min_edge_mm)
        if bool(np.all(ok & clear)):
            return np.asarray(candidate, float), True
    for min_edge in (min_edge_mm, 15.0, 10.0):
        try:
            return np.asarray(random_bridge_points(
                contour, n=int(n_points), min_edge_dist=float(min_edge),
                rng=np.random.default_rng(int(seed))), float), False
        except RuntimeError:
            continue
    raise RuntimeError("could not place bridge points on an anchor contour")


def hole_is_valid(centre_mm, diameter_mm, contour, bridge_pts,
                  min_edge_mm: float = 20.0) -> bool:
    """The WHOLE circle inside, clear of the edge, and clear of every bridge.

    The generator's own placement enforces all three; an anchored hole bypasses
    that placement, so it has to be held to the same standard or a block whose
    point is a fixed hole position will quietly produce invalid bodies.
    """
    centre = np.asarray(centre_mm, float).reshape(1, 2)
    if not bool(inside(centre, contour)[0]):
        return False
    radius = float(diameter_mm) / 2.0
    # Distance from the CENTRE to the boundary must exceed the radius plus the
    # rim clearance, which is what puts the whole circle inside with margin.
    if float(distance_to_boundary(centre, contour)[0]) < radius + min_edge_mm:
        return False
    gap = np.linalg.norm(np.asarray(bridge_pts, float)[:, :2] - centre, axis=1)
    return bool(gap.min() >= radius + dg.BRIDGE_SOUNDHOLE_MARGIN_MM)


def place_soundhole(contour, bridge_pts, diameter_mm: float, seed: int,
                    hole_norm=None):
    """A hole of the requested size, clear of every bridge; None if impossible.

    ``hole_norm`` anchors it to a normalised position instead of drawing one.
    Blocks A and B need that: re-drawing the hole for every scale turns "the
    same body, larger" into "a different body", and the measured drift was up
    to 1.39 in normalised units -- larger than the effect the block exists to
    isolate.
    """
    margin = dg.BRIDGE_SOUNDHOLE_MARGIN_MM
    if hole_norm is not None:
        centre = from_normalised(np.asarray(hole_norm, float).reshape(1, 2),
                                 contour)[0]
        if hole_is_valid(centre, diameter_mm, contour, bridge_pts):
            return [float(centre[0]), float(centre[1])], 0
        return None, -1            # anchored placement failed: do NOT fall back
    for attempt in range(128):
        centre = random_soundhole(
            contour, bridge_pts, diameter_mm=float(diameter_mm),
            min_bridge_dist=float(diameter_mm) / 2.0 + margin,
            min_edge_dist=20.0,
            rng=np.random.default_rng(int(seed) + attempt))
        if centre is not None:
            return [float(centre[0]), float(centre[1])], attempt + 1
    return None, 128


def make_shape(shape_id: int, source: dict, *, contour, thickness,
               top_plate, soundhole_diameter, template_norm, n_bridges,
               seed: int, block: str, provenance: dict,
               body_types=("solid", "hollow"), hole_norm=None) -> dict | None:
    """One anchor base shape, or None when it cannot be placed validly."""
    contour = np.asarray(contour, float)
    bridge_pts, shared = place_bridges(contour, template_norm, 25.0,
                                       seed, n_bridges)
    hole_centre, attempts = place_soundhole(contour, bridge_pts,
                                            soundhole_diameter, seed,
                                            hole_norm=hole_norm)
    if hole_centre is None:
        return None
    cavity_ratio = cavity_ratio_from(thickness, top_plate)
    return {
        "base_shape_id": int(shape_id),
        "shape_type": source["shape_type"],
        "guitar_model": source["guitar_model"],
        "contour": contour,
        "thickness": float(thickness),
        "top_plate_thickness_mm": float(top_plate),
        "cavity_ratio": cavity_ratio,
        "bridge_pts_body": np.asarray(bridge_pts, float),
        "soundhole_ok": True,
        "soundhole_diameter": float(soundhole_diameter),
        "soundhole_center": hole_centre,
        "soundhole_attempts": int(attempts),
        # Not hashed by the generator; carried so the analysis can group anchor
        # rows by block and by which source body they came from.
        "_anchor": {"block": block, "shared_bridges": bool(shared),
                    "anchored_hole": hole_norm is not None,
                    "body_types": list(body_types), **provenance},
    }


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def pick_sources(source_plan, n: int, splits, seed: int) -> list[dict]:
    """Source contours, drawn so the v6 train/val/test proportions are kept.

    Anchor rows have to land in validation as well as training: a block that
    exists only in train can measure a fit but never a generalisation.
    """
    by_split: dict[str, list] = {}
    for shape in source_plan["base_shapes"]:
        by_split.setdefault(splits[str(shape["base_shape_id"])],
                            []).append(shape)
    rng = np.random.default_rng(int(seed))
    total = sum(len(v) for v in by_split.values())
    # Largest remainder, not per-split rounding: rounding each split
    # independently loses shapes (asking for 16 returned 15), and a block whose
    # size silently depends on a rounding artefact cannot be reasoned about.
    splits = sorted(by_split)
    exact = {s: n * len(by_split[s]) / total for s in splits}
    take = {s: int(np.floor(exact[s])) for s in splits}
    for s in sorted(splits, key=lambda s: exact[s] - take[s], reverse=True):
        if sum(take.values()) >= n:
            break
        take[s] += 1
    chosen: list[dict] = []
    for split in splits:
        pool = sorted(by_split[split], key=lambda s: s["base_shape_id"])
        count = min(take[split], len(pool))
        index = rng.permutation(len(pool))[:count]
        chosen += [pool[int(i)] for i in sorted(index)]
    if len(chosen) != n:
        raise SystemExit(
            f"asked for {n} source contours but could take only {len(chosen)} "
            f"(per split: { {s: len(by_split[s]) for s in splits} })")
    return chosen


def pick_from(items, n: int, splits, seed: int, key) -> list:
    """Take ``n`` items keeping the source split proportions (largest remainder)."""
    by_split: dict[str, list] = {}
    for item in items:
        by_split.setdefault(splits[str(key(item))], []).append(item)
    if not by_split:
        return []
    total = sum(len(v) for v in by_split.values())
    names = sorted(by_split)
    exact = {s: min(n, total) * len(by_split[s]) / total for s in names}
    take = {s: int(np.floor(exact[s])) for s in names}
    for s in sorted(names, key=lambda s: exact[s] - take[s], reverse=True):
        if sum(take.values()) >= min(n, total):
            break
        take[s] += 1
    rng = np.random.default_rng(int(seed))
    chosen: list = []
    for split in names:
        pool = sorted(by_split[split], key=key)
        index = rng.permutation(len(pool))[:min(take[split], len(pool))]
        chosen += [pool[int(i)] for i in sorted(index)]
    return chosen


def build_anchor_shapes(source_plan, args) -> list[dict]:
    splits = source_plan["splits"]
    shapes: list[dict] = []
    next_id = 0

    def emit(**kwargs):
        nonlocal next_id
        shape = make_shape(next_id, **kwargs)
        if shape is None:
            print(f"  skip: {kwargs['block']} "
                  f"{kwargs['provenance']} (placement failed)")
            return
        shape["_anchor"]["source_split"] = splits[
            str(kwargs["provenance"]["source_base_shape_id"])]
        shapes.append(shape)
        next_id += 1

    # -- Block C: scalar density on existing outlines --------------------
    sources = pick_sources(source_plan, args.c_contours, splits, args.seed)
    print(f"block C: {len(sources)} contours x {len(args.c_thickness_mm)} "
          f"thickness x {len(args.c_top_plate)} top plate x "
          f"{len(args.c_soundhole)} hole")
    for source in sources:
        contour = np.asarray(source["contour"], float)
        char = characteristic_length(contour)
        template = to_normalised(np.asarray(source["bridge_pts_body"], float),
                                 contour)
        for thickness in args.c_thickness_mm:
            first = True
            for plate_ratio in args.c_top_plate:
                for hole_ratio in args.c_soundhole:
                    # The solid problem depends only on (contour, thickness), so
                    # it is emitted once per thickness and the remaining grid
                    # points are hollow-only.
                    bodies = ("solid", "hollow") if first else ("hollow",)
                    first = False
                    emit(source=source, contour=contour,
                         thickness=thickness,
                         top_plate=max(plate_ratio * thickness,
                                       args.min_top_plate_mm),
                         soundhole_diameter=hole_ratio * char,
                         template_norm=template, n_bridges=args.n_bridges,
                         seed=args.seed + next_id * 977,
                         block="C_scalar_density", body_types=bodies,
                         provenance={
                             "source_base_shape_id": source["base_shape_id"],
                             "thickness_mm": float(thickness),
                             "top_plate_ratio": plate_ratio,
                             "soundhole_ratio": hole_ratio, "scale": 1.0})

    # -- Block A: scale isolation ----------------------------------------
    sources = pick_sources(source_plan, args.a_contours, splits, args.seed + 1)
    print(f"block A: {len(sources)} contours x {len(args.a_scales)} scales "
          f"(geometrically similar: thickness and hole scale too)")
    for source in sources:
        base = np.asarray(source["contour"], float)
        template = to_normalised(np.asarray(source["bridge_pts_body"], float),
                                 base)
        hole_norm = to_normalised(
            np.asarray(source["soundhole_center"], float).reshape(1, 2), base)
        for scale in args.a_scales:
            # Geometric similarity of the SOURCE body: every length scales by
            # the same factor, so the response is predicted to shift as 1/scale
            # exactly, and scale 1.0 reproduces a body the mesher has already
            # accepted -- an internal control that costs nothing.  Deriving the
            # thickness from sqrt(area) instead would tie it to body size, which
            # gave 17-114 mm bodies far outside anything ever meshed.
            emit(source=source, contour=scaled(base, scale),
                 thickness=source["thickness"] * scale,
                 top_plate=source["top_plate_thickness_mm"] * scale,
                 soundhole_diameter=source["soundhole_diameter"] * scale,
                 template_norm=template, hole_norm=hole_norm,
                 n_bridges=args.n_bridges,
                 seed=args.seed + next_id * 977, block="A_scale",
                 provenance={"source_base_shape_id": source["base_shape_id"],
                             "scale": float(scale),
                             # Recorded because they do NOT scale, so an
                             # analysis of this block can account for them.
                             "fixed_lengths_mm": FIXED_INTERNAL_LENGTHS_MM})

    # -- Block B: contour isolation at one characteristic length ---------
    #
    # Every body in this block must be identical except for its outline, so a
    # candidate is only admitted when BOTH anchors hold: the shared normalised
    # bridge template fits it, and the shared normalised hole is valid on it.
    # An outline that needs its own bridges is not comparable -- its differences
    # would be drive-point differences -- so it is dropped rather than included
    # with a footnote.  Candidates are drawn from the whole source plan and then
    # filtered, so the requested count is met if it can be.
    thickness = args.b_thickness_mm
    top_plate = max(args.b_top_plate_ratio * thickness, args.min_top_plate_mm)
    diameter = args.b_soundhole_ratio * args.b_char_length
    # The template is SEARCHED, not picked.  Any one shape's bridge layout sits
    # near its own edges and falls outside most other outlines -- shape 0's
    # admitted 10 of 100.  Every source layout is tried, each also contracted
    # toward the centroid by a few factors: contraction keeps the PATTERN (which
    # is what has to be shared) while pulling the points away from the boundary,
    # and the best combination is the one the most outlines can accept.
    normalised = []
    for source in source_plan["base_shapes"]:
        base = np.asarray(source["contour"], float)
        scale = args.b_char_length / characteristic_length(base)
        normalised.append((source, scale, scaled(base, scale)))

    def admits(template):
        out = []
        for source, scale, contour in normalised:
            bridges = from_normalised(template, contour)
            if not bool(np.all(inside(bridges, contour))):
                continue
            if float(distance_to_boundary(bridges, contour).min()) < 25.0:
                continue
            out.append((source, scale, contour, bridges))
        return out

    bridge_template, candidates = None, []
    for source in source_plan["base_shapes"]:
        base = np.asarray(source["contour"], float)
        layout = to_normalised(np.asarray(source["bridge_pts_body"], float), base)
        for shrink in (1.0, 0.85, 0.7, 0.55):
            admitted = admits(layout * shrink)
            if len(admitted) > len(candidates):
                bridge_template, candidates = layout * shrink, admitted
            if len(candidates) >= len(normalised):
                break
    if bridge_template is None:
        raise SystemExit("no bridge template fits any outline at this size")

    # The hole position that is valid on the most candidates.  A hand-picked
    # coordinate rejected 15 of 32 outlines; different shapes at equal area do
    # not all admit the same hole, and shrinking the block silently is worse
    # than searching for a position that does not.
    grid = np.linspace(-0.45, 0.45, 19)
    best, best_valid = np.asarray(args.b_hole_norm, float), []
    for y in grid:
        for x in grid:
            candidate = np.array([[x, y]])
            valid = [c for c in candidates
                     if hole_is_valid(from_normalised(candidate, c[2])[0],
                                      diameter, c[2], c[3])]
            if len(valid) > len(best_valid):
                best, best_valid = candidate.reshape(2), valid
    hole_norm = best.reshape(1, 2)
    print(f"block B: {len(candidates)}/{len(source_plan['base_shapes'])} "
          f"outlines take the shared bridge template; shared hole at "
          f"{tuple(np.round(best, 3))} is valid on {len(best_valid)} of them")

    keep = pick_from(best_valid, args.b_contours, splits, args.seed + 2,
                     key=lambda c: c[0]["base_shape_id"])
    print(f"  keeping {len(keep)} (requested {args.b_contours}), "
          f"split-proportional")
    for source, scale, contour, _bridges in keep:
        emit(source=source, contour=contour,
             thickness=thickness, top_plate=top_plate,
             soundhole_diameter=diameter,
             template_norm=bridge_template, hole_norm=hole_norm,
             n_bridges=args.n_bridges,
             seed=args.seed + next_id * 977, block="B_contour",
             provenance={"source_base_shape_id": source["base_shape_id"],
                         "scale": float(scale),
                         "thickness_mm": float(thickness)})
    return shapes


def anchor_materials(source_plan, shape_ids, n_materials: int) -> list[dict]:
    """The SAME material vectors on every shape, taken from v6's SPD-verified bank.

    Reusing vectors the generator already accepted avoids re-deriving the SPD
    rejection criterion here, and picking them by spread over longitudinal
    modulus keeps the two anchors from being near-duplicates.
    """
    pool = sorted(source_plan["materials"], key=lambda m: m["E1"])
    picks = [pool[int(round(q * (len(pool) - 1)))]
             for q in np.linspace(0.25, 0.75, int(n_materials))]
    bank = []
    for shape_id in sorted(shape_ids):
        for slot, template in enumerate(picks):
            material = {k: float(v) for k, v in template.items()
                        if isinstance(v, (int, float))
                        and k not in ("base_shape_id", "material_slot",
                                      "spd_attempts")}
            material.update(material_id=f"anchor_{shape_id:04d}_{slot:02d}",
                            material_name=f"anchor_{slot:02d}",
                            base_shape_id=int(shape_id),
                            material_slot=int(slot), spd_attempts=1)
            bank.append(material)
    return bank


# ---------------------------------------------------------------------------
# Plan assembly -- mirrors dataset_gen_mixed.build_plan exactly
# ---------------------------------------------------------------------------

def assemble(shapes, materials, config, splits_by_shape) -> dict:
    freqs = np.geomspace(config["freq_min"], config["freq_max"],
                         config["freq_points"])
    by_shape: dict[int, list[dict]] = {}
    for material in materials:
        by_shape.setdefault(int(material["base_shape_id"]), []).append(material)

    cases = []
    for shape in shapes:
        b = shape["base_shape_id"]
        # A solid body has no cavity and no hole, so two anchor shapes that
        # differ ONLY in top-plate or soundhole produce the identical solid
        # problem.  Emitting both would spend a day of FEM on duplicates, so
        # each shape declares which bodies it is actually needed for.
        for body_type in shape["_anchor"].get("body_types",
                                              config["body_types"]):
            shape_id = b * 2 + (0 if body_type == "solid" else 1)
            for material in by_shape[b]:
                mid = material["material_id"]
                cases.append({
                    "case_id": f"s{b:04d}_{body_type}_{mid}",
                    "base_shape_id": b, "shape_id": int(shape_id),
                    "body_type": body_type, "material_id": mid,
                    "split": splits_by_shape[b],
                    "n_bridges": int(shape["bridge_pts_body"].shape[0]),
                    "case_seed": int.from_bytes(hashlib.sha256(
                        f"{config['seed']}|case|{b}|{body_type}|{mid}".encode())
                        .digest()[:8], "little"),
                })
    if len({c["case_id"] for c in cases}) != len(cases):
        raise RuntimeError("plan contains duplicate case IDs")

    serial, digests = [], {}
    for shape in shapes:
        serial.append({
            "base_shape_id": int(shape["base_shape_id"]),
            "shape_type": shape["shape_type"],
            "guitar_model": shape["guitar_model"],
            "contour": np.asarray(shape["contour"], float).tolist(),
            "thickness": float(shape["thickness"]),
            "top_plate_thickness_mm": float(shape["top_plate_thickness_mm"]),
            "cavity_ratio": float(shape["cavity_ratio"]),
            "bridge_pts_body": np.asarray(shape["bridge_pts_body"], float).tolist(),
            "soundhole_ok": True,
            "soundhole_diameter": float(shape["soundhole_diameter"]),
            "soundhole_center": shape["soundhole_center"],
            "soundhole_attempts": int(shape["soundhole_attempts"]),
            "anchor": shape["_anchor"],
        })
        digests[shape["base_shape_id"]] = hashlib.sha256(
            dg._canonical_json({
                "contour": np.round(shape["contour"], 6),
                "thickness": round(float(shape["thickness"]), 6),
                "top_plate_thickness_mm": round(
                    float(shape["top_plate_thickness_mm"]), 6),
                "cavity_ratio": round(float(shape["cavity_ratio"]), 6),
                "bridge": np.round(shape["bridge_pts_body"], 6),
                "soundhole_ok": True,
                "soundhole_diameter": round(
                    float(shape["soundhole_diameter"]), 6),
                "soundhole_center": shape["soundhole_center"],
            }).encode()).hexdigest()

    body = {
        "generator_version": dg.GENERATOR_VERSION,
        "schema_version": dg.DATASET_SCHEMA_VERSION,
        "config": config,
        "solid_solver_revision": dg.SOLID_SOLVER_REVISION,
        "hollow_solver_revision": dg.HOLLOW_SOLVER_REVISION,
        "production_contract": dg.PRODUCTION_CONTRACT,
        # Computed by the generator's own function, over the generator's own
        # source list -- so a plan built here is bound to exactly the same
        # solver code a mixed-v6 plan is.
        "implementation_fingerprint": dg._implementation_fingerprint(),
        "frequencies": np.round(freqs, 9).tolist(),
        "materials": materials,
        "base_shapes": serial,
        "splits": {str(s["base_shape_id"]): splits_by_shape[s["base_shape_id"]]
                   for s in shapes},
        "shape_digests": {str(k): v for k, v in digests.items()},
        "cases": [{k: c[k] for k in ("case_id", "base_shape_id", "shape_id",
                                     "body_type", "material_id", "split",
                                     "n_bridges", "case_seed")} for c in cases],
    }
    return {"plan_hash": dg.plan_hash(body), "plan_body": body,
            "_materials": materials, "_base_shapes": shapes,
            "_frequencies": freqs, "_cases": cases}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="results/mixed_v6_full", type=Path,
                   help="Plan whose contours and SPLITS are reused.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", default=20260805, type=int)
    p.add_argument("--n-bridges", default=10, type=int)
    p.add_argument("--n-materials", default=2, type=int)
    # Block C -- thickness in ABSOLUTE mm.  Tying it to sqrt(area) would make
    # thickness a function of body size, which is precisely the coupling this
    # block exists to break, and it produced 91 mm bodies far outside anything
    # the mesher has accepted.  The default straddles the source band (40-55 mm)
    # on both sides, so the block is new coverage rather than a re-sample.
    p.add_argument("--c-contours", default=8, type=int)
    p.add_argument("--c-thickness-mm", default=[32.0, 46.0, 64.0], type=float,
                   nargs="+", metavar="MM")
    # Plate as a fraction of thickness, inside the band the mesher has accepted
    # (0.057-0.151); the cavity ratio follows from it exactly.
    p.add_argument("--c-top-plate", default=[0.10], type=float,
                   nargs="+", metavar="R")
    p.add_argument("--c-soundhole", default=[0.12, 0.22], type=float,
                   nargs="+", metavar="R")
    # Block A -- every length of the SOURCE body scales together.  Upward only:
    # a factor below 1 would take the top plate under the thinnest the mesher
    # has accepted, and similarity forbids clamping it back.
    p.add_argument("--a-contours", default=8, type=int)
    p.add_argument("--a-scales", default=[1.0, 1.18, 1.40], type=float,
                   nargs="+")
    # Block B -- one characteristic length, one thickness, many outlines.
    p.add_argument("--b-contours", default=32, type=int,
                   help="Block B's independent unit is the CONTOUR, so this is "
                        "its real sample size. 16 gave 2 validation contours, "
                        "which cannot support any generalisation claim.")
    p.add_argument("--b-char-length", default=280.0, type=float, metavar="MM")
    p.add_argument("--b-thickness-mm", default=46.0, type=float, metavar="MM")
    p.add_argument("--b-top-plate-ratio", default=0.10, type=float)
    p.add_argument("--b-soundhole-ratio", default=0.17, type=float)
    p.add_argument("--b-template-shape", default=0, type=int, metavar="I",
                   help="Index into the source plan's base_shapes whose bridge "
                        "layout becomes the SHARED normalised template for "
                        "block B. Outlines that cannot take it are dropped: an "
                        "outline with its own bridges is not comparable.")
    p.add_argument("--b-hole-norm", default=[0.0, -0.30], type=float, nargs=2,
                   metavar=("X", "Y"),
                   help="Soundhole position for block B, in the normalised "
                        "frame, SHARED by every outline. Without it the hole "
                        "moves from body to body and the block compares hole "
                        "position as much as outline.")
    p.add_argument("--min-top-plate-mm", default=3.0, type=float, metavar="MM",
                   help="Floor on the absolute top-plate thickness. Defaults to "
                        "the thinnest plate the mesher accepted in the source "
                        "plan; a thinner one is a mesh failure waiting to "
                        "happen, and the cavity ratio follows from it.")
    p.add_argument("--expect-cases", default=None, type=int, metavar="N",
                   help="Fail unless the plan has exactly N cases. The block "
                        "sizes depend on several flags at once; asserting the "
                        "total is how a silent change gets caught.")
    p.add_argument("--emit-blocks", default=None, type=Path, metavar="PATH",
                   help="Write base_shape_id -> block as JSON. With --dry-run "
                        "this recovers the map for a dataset generated before "
                        "the map was persisted.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the plan and validate it, but write nothing.")
    args = p.parse_args()

    source_plan = json.loads(
        (args.source / "dataset_plan.json").read_text(encoding="utf-8"))
    print(f"source: {args.source}  ({len(source_plan['base_shapes'])} contours, "
          f"generator {source_plan['config']['generator_version']})")

    bands = verify_cavity_relation(source_plan)
    shapes = build_anchor_shapes(source_plan, args)
    if not shapes:
        raise SystemExit("no anchor shapes could be placed")
    splits_by_shape = {s["base_shape_id"]: s["_anchor"]["source_split"]
                       for s in shapes}
    materials = anchor_materials(source_plan,
                                 [s["base_shape_id"] for s in shapes],
                                 args.n_materials)

    config = copy.deepcopy(source_plan["config"])
    config.update(n_base_shapes=len(shapes), n_materials=args.n_materials,
                  n_bridge_points=args.n_bridges, seed=int(args.seed))
    plan = assemble(shapes, materials, config, splits_by_shape)

    # -- report ----------------------------------------------------------
    import collections
    cases = plan["_cases"]
    by_block = collections.Counter(s["_anchor"]["block"] for s in shapes)
    by_body = collections.Counter(c["body_type"] for c in cases)
    by_split = collections.Counter(c["split"] for c in cases)
    shared = sum(1 for s in shapes if s["_anchor"]["shared_bridges"])
    print(f"\nplan_hash {plan['plan_hash'][:16]}")
    print(f"shapes {len(shapes)}  ({dict(by_block)})")
    print(f"  bridges at the shared normalised template: {shared}/{len(shapes)}")
    print(f"cases  {len(cases)}  ({dict(by_split)}, {dict(by_body)})")
    print(f"samples {len(cases) * args.n_bridges}")
    print(f"materials {len({m['material_id'][-2:] for m in materials})} distinct "
          f"vectors, repeated on every shape")

    # -- validation ------------------------------------------------------
    # Anchor bodies are MEANT to sit outside the sampled bands -- that is the
    # new coverage -- but a plate far below anything the mesher has accepted is
    # a failure waiting to happen, so every excursion is reported.
    (t_lo, t_hi), (p_lo, p_hi), (r_lo, r_hi) = bands
    outside = []
    for shape in shapes:
        thickness = float(shape["thickness"])
        plate = float(shape["top_plate_thickness_mm"])
        ratio = plate / max(thickness, 1e-9)
        tags = []
        if not t_lo <= thickness <= t_hi:
            tags.append(f"thickness {thickness:.1f}")
        if not p_lo <= plate <= p_hi:
            tags.append(f"plate {plate:.2f}")
        if not r_lo <= ratio <= r_hi:
            tags.append(f"ratio {ratio:.4f}")
        if tags:
            outside.append((shape["base_shape_id"], shape["_anchor"]["block"],
                            ", ".join(tags)))
    if outside:
        print(f"\noutside the source bands: {len(outside)}/{len(shapes)} "
              f"shapes (intended for thickness, a risk for the plate)")
        for shape_id, block, tags in outside[:8]:
            print(f"  s{shape_id:04d} {block}: {tags}")
        thin = [o for o in outside if "plate" in o[2]
                and float(o[2].split("plate ")[1].split(",")[0]) < p_lo]
        if thin:
            print(f"  {len(thin)} shapes have a top plate BELOW the thinnest the "
                  f"mesher has ever accepted ({p_lo:.2f} mm); raise "
                  f"--c-top-plate or --min-top-plate-mm if those fail.")

    problems = []
    vectors = {}
    for material in materials:
        key = material["material_slot"]
        value = tuple(round(material[k], 9) for k in dg.MATERIAL_RANGES)
        vectors.setdefault(key, set()).add(value)
    for slot, values in vectors.items():
        if len(values) != 1:
            problems.append(f"anchor material slot {slot} is not identical "
                            f"across shapes ({len(values)} variants)")
    margins = {"hole_edge": [], "bridge_edge": [], "bridge_rim": []}
    for shape in shapes:
        contour = np.asarray(shape["contour"], float)
        bridges = np.asarray(shape["bridge_pts_body"], float)
        radius = float(shape["soundhole_diameter"]) / 2.0
        hole = np.asarray(shape["soundhole_center"], float).reshape(1, 2)
        tag = f"shape {shape['base_shape_id']} ({shape['_anchor']['block']})"
        if self_intersects(contour):
            problems.append(f"{tag}: contour self-intersects")
        if not bool(np.all(inside(bridges, contour))):
            problems.append(f"{tag}: a bridge point is outside its contour")
        bridge_edge = float(distance_to_boundary(bridges, contour).min())
        margins["bridge_edge"].append(bridge_edge)
        if bridge_edge < 10.0:
            problems.append(f"{tag}: bridge only {bridge_edge:.2f} mm from the "
                            f"edge (the generator's own floor is 10 mm)")
        if not bool(inside(hole, contour)[0]):
            problems.append(f"{tag}: soundhole centre outside the contour")
        # The WHOLE circle, not just its centre -- the earlier check passed a
        # hole whose rim crossed the boundary.
        hole_edge = float(distance_to_boundary(hole, contour)[0]) - radius
        margins["hole_edge"].append(hole_edge)
        if hole_edge < 20.0:
            problems.append(f"{tag}: soundhole rim only {hole_edge:.2f} mm from "
                            f"the edge (placement requires 20 mm)")
        rim = float(np.linalg.norm(bridges[:, :2] - hole, axis=1).min()) - radius
        margins["bridge_rim"].append(rim)
        if rim < dg.BRIDGE_SOUNDHOLE_MARGIN_MM:
            problems.append(f"{tag}: bridge only {rim:.2f} mm from the soundhole "
                            f"rim (contract requires "
                            f"{dg.BRIDGE_SOUNDHOLE_MARGIN_MM})")
        # The cavity is inset from the outline by the side wall; a body whose
        # inset collapses would produce a fragmented or empty cavity.
        if float(distance_to_boundary(hole, contour)[0]) <=                 FIXED_INTERNAL_LENGTHS_MM["wall_thickness"]:
            problems.append(f"{tag}: soundhole sits inside the cavity side wall")
        air_height = (float(shape["thickness"])
                      - float(shape["top_plate_thickness_mm"])
                      - FIXED_INTERNAL_LENGTHS_MM["back_plate"])
        if air_height <= 0.0:
            problems.append(f"{tag}: no air cavity left "
                            f"(height {air_height:.2f} mm)")
    for slot, template in enumerate(sorted({m["material_slot"] for m in materials})):
        sample = next(m for m in materials if m["material_slot"] == template)
        if not dg.material_is_spd({k: sample[k] for k in dg.MATERIAL_RANGES}):
            problems.append(f"anchor material slot {template} is not SPD")
    if args.expect_cases is not None and len(cases) != int(args.expect_cases):
        problems.append(f"expected {args.expect_cases} cases, plan has "
                        f"{len(cases)}")
    # Splits must agree with the source for every reused geometry.
    for shape in shapes:
        source_id = shape["_anchor"]["source_base_shape_id"]
        if splits_by_shape[shape["base_shape_id"]] != \
                source_plan["splits"][str(source_id)]:
            problems.append(f"shape {shape['base_shape_id']}: split does not "
                            f"match source shape {source_id}")
    if problems:
        print("\nVALIDATION FAILED")
        for problem in problems[:20]:
            print(f"  {problem}")
        raise SystemExit(1)
    print(f"\nvalidation OK -- whole hole inside with >=20 mm rim clearance, "
          f"bridges >=10 mm from the edge and "
          f">={dg.BRIDGE_SOUNDHOLE_MARGIN_MM} mm from the rim, no "
          f"self-intersection, air cavity non-empty, anchor materials SPD and "
          f"identical across shapes, splits inherited")
    print(f"  tightest margins: hole-edge {min(margins['hole_edge']):.2f} mm, "
          f"bridge-edge {min(margins['bridge_edge']):.2f} mm, "
          f"bridge-rim {min(margins['bridge_rim']):.2f} mm")
    anchored = sum(1 for s in shapes if s["_anchor"].get("anchored_hole"))
    print(f"  soundhole anchored to a fixed normalised position: "
          f"{anchored}/{len(shapes)} shapes (blocks A and B require it)")

    scales = sorted({s["_anchor"]["scale"] for s in shapes
                     if s["_anchor"]["block"] == "A_scale"})
    print(f"\nblock A scales {scales}")
    print("  SOLID bodies are exactly similar: eigenfrequencies scale as 1/s "
          "and |Y| as 1/s^2, i.e. a shift along log-f plus a constant dB "
          "offset.  That is an exact check on the model -- EXCEPT for damping: "
          "Rayleigh beta is fixed in seconds, so zeta = beta*omega/2 changes "
          "with scale and peak heights and widths do not transfer.")
    print(f"  HOLLOW bodies are NOT similar.  The CAD holds "
          f"{FIXED_INTERNAL_LENGTHS_MM} fixed in millimetres whatever the body "
          f"size, so a scaled hollow body has proportionally thinner walls. It "
          f"is still new scale coverage, but the 1/s prediction does not apply "
          f"to it and any analysis must say so.")

    block_map = {
        "blocks": {str(s["base_shape_id"]): s["_anchor"]["block"]
                   for s in shapes},
        "anchor": {str(s["base_shape_id"]): s["_anchor"] for s in shapes},
    }
    if args.emit_blocks:
        # Recovering the map for a dataset generated before it was persisted:
        # the planner is deterministic given the same source plan and the same
        # args, so rebuilding reproduces the original assignment.
        Path(args.emit_blocks).write_text(
            json.dumps(block_map, indent=2, default=str), encoding="utf-8")
        print("block map written: %s" % args.emit_blocks)
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    dg.save_plan(plan, args.out)
    # The generator's serialisation keeps only the fields that DEFINE a
    # geometry and drops ``_anchor``, so without writing the block map here the
    # block a finished case belongs to is recorded nowhere -- and per-block
    # analysis is the entire reason this dataset exists.
    (args.out / "anchor_design.json").write_text(
        json.dumps({"source": str(args.source), "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()}, **block_map},
            indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out / 'dataset_plan.json'}")
    print(f"Generate with the SAME command mixed-v6 uses, pointed at {args.out}.")


if __name__ == "__main__":
    main()
