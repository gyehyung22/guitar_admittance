"""PyTorch dataset for the mixed magnitude-and-peak surrogate target."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


# Generator versions this loader can consume together.  They share the
# mixed-magnitude-peaks-v1 schema and the 500-point logarithmic frequency grid;
# mixed-v3 used a LINEAR grid and is deliberately absent.
SUPPORTED_GENERATOR_VERSIONS = ("mixed-v4", "mixed-v5", "mixed-v6")


MATERIAL_COLS = ["E_L", "E_R", "E_T", "G_LR", "G_LT", "G_RT",
                 "nu_LR", "nu_LT", "nu_RT", "density"]
# v5: cavity_ratio (derived) replaced by the directly-sampled top-plate thickness.
# Solid rows carry top_plate_thickness_mm = 0.0.  These are the columns read
# straight out of the manifest; SIZE_COLS below are derived from the contour and
# appended, so the geometry VECTOR is len(GEOMETRY_COLS) + len(SIZE_COLS).
GEOMETRY_COLS = ["thickness", "top_plate_thickness_mm", "cavity_volume_m3",
                 "soundhole_diameter", "soundhole_area_m2",
                 "soundhole_center_x", "soundhole_center_y"]
# The rasterised shape channel is scale-normalised (see contour_to_grid), so the
# body's ABSOLUTE plan size has to reach the network some other way.  It matters:
# plate modal frequencies go as f ~ 1/L^2, and the dataset spans a 12.7x plan-area
# range.  Solid rows cannot recover it from anything else — their cavity_volume_m3
# and every soundhole_* column are 0.0.
SIZE_COLS = ["width_mm", "height_mm", "area_mm2"]
# The soundhole is never drawn into the shape channel — contour.npy is a single
# closed polygon and cannot carry a hole — so it reaches the network only as
# scalars.  Its centre is stored in absolute mm, which no longer locates it on a
# scale-normalised image, so the same normalised pair given for the bridge is
# derived here.  Solid rows have no hole and carry zeros throughout.
SOUNDHOLE_COLS = ["soundhole_center_x_norm", "soundhole_center_y_norm"]
DERIVED_COLS = SIZE_COLS + SOUNDHOLE_COLS
GEOMETRY_DIM = len(GEOMETRY_COLS) + len(DERIVED_COLS)
# bridge = (x, y, z) in mm PLUS (x, y) in the same normalised frame as the shape
# grid and z as a fraction of body thickness.  The normalised pair is what locates
# the drive point on the mode shape the CNN actually sees.
BRIDGE_DIM = 6
BODY_TYPE_DIM = 2
# Scalar inputs consumed by PhysicsEncoder / the physics-only baseline.
PHYSICS_INPUT_DIM = (len(MATERIAL_COLS) + BRIDGE_DIM + GEOMETRY_DIM
                     + BODY_TYPE_DIM)
# Names of the concatenated physics vector, in the order PhysicsEncoder builds it
# (material, bridge, geometry, body type).  Ablations that suppress a subset of
# these need indices, and hard-coding them would silently rot the first time a
# column is added anywhere above.
BRIDGE_NAMES = ["bridge_x_mm", "bridge_y_mm", "bridge_z_mm",
                "bridge_x_norm", "bridge_y_norm", "bridge_z_frac"]
BODY_TYPE_NAMES = ["is_solid", "is_hollow"]
PHYSICS_INPUT_NAMES = (list(MATERIAL_COLS) + BRIDGE_NAMES
                       + list(GEOMETRY_COLS) + list(DERIVED_COLS)
                       + BODY_TYPE_NAMES)
assert len(PHYSICS_INPUT_NAMES) == PHYSICS_INPUT_DIM
# The three columns that duplicate, as scalars, information the shape raster is
# also supposed to carry.  They identify a base geometry essentially uniquely
# (over the 100 planned shapes the triple is unique, minimum standardized
# separation 0.078), so a network can index a TRAINING shape from them without
# ever consulting the image.  Named here so the shortcut ablation is one flag.
GLOBAL_GEOMETRY_NAMES = list(SIZE_COLS)


def physics_input_index(names=None) -> list[int]:
    """Indices of ``names`` inside the concatenated physics vector."""
    if names is None:
        return list(range(PHYSICS_INPUT_DIM))
    lookup = {name: i for i, name in enumerate(PHYSICS_INPUT_NAMES)}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise KeyError(f"not physics inputs: {missing}; "
                       f"available: {PHYSICS_INPUT_NAMES}")
    return [lookup[n] for n in names]
DEFAULT_GRID_SIZE = 96
# Fixed peak-label slots per bridge, mirroring
# ``dataset_gen_mixed.PRODUCTION_CONTRACT["top_k_peaks"]``.  Slots are prefix
# packed (``peak_mask`` marks the used prefix) and stored in ASCENDING frequency
# order, which is what makes slot-wise supervision well defined: slot k always
# means "the k-th lowest labelled resonance", so no set matching is needed.
PEAK_SLOTS = 32


def contour_frame(contour: np.ndarray, pad_frac: float = 0.05):
    """Return ``(center_xy, span_mm)`` of the ISOTROPIC normalisation frame.

    The frame is a square of side ``span_mm`` centred on the contour's bounding-box
    centre, so a shape maps into it without distorting its aspect ratio.  Both
    ``contour_to_grid`` and the normalised bridge coordinates use this frame, which
    is what keeps the drive point aligned with the rasterised shape.
    """
    pts = np.asarray(contour, float)
    if (pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3
            or not np.all(np.isfinite(pts))):
        raise ValueError("contour must be a finite (N,2) polygon")
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], float)
    span = max(x_max - x_min, y_max - y_min) * (1.0 + 2.0 * float(pad_frac))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("contour has a degenerate bounding box")
    return center, float(span)


def contour_size_features(contour: np.ndarray) -> np.ndarray:
    """Absolute ``(width_mm, height_mm, area_mm2)`` — the scale the grid drops."""
    pts = np.asarray(contour, float)
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return np.asarray([width, height, area], dtype=np.float64)


def contour_to_grid(contour: np.ndarray, grid_size: int = DEFAULT_GRID_SIZE,
                    pad_frac: float = 0.05) -> np.ndarray:
    """Rasterize an ``(N,2)`` contour to a binary occupancy grid.

    The frame is ISOTROPIC (a square of side ``max(width, height)``): x and y are
    divided by the same number, so the aspect ratio survives.  Normalising each
    axis to its own extent — as this function used to — mapped every shape onto a
    filled square, making a 0.46-aspect body and a 2.00-aspect body pixel-identical
    even though their mode shapes differ.  Absolute size is deliberately NOT in
    this channel; it is supplied by ``SIZE_COLS`` instead, because ``ShapeEncoder``
    ends in ``AdaptiveAvgPool2d(1)`` and a common-mm-canvas rasterisation would make
    the pooled embedding track canvas occupancy (a 12.8x range) rather than form.
    """
    from matplotlib.path import Path as MplPath

    pts = np.asarray(contour, float)
    center, span = contour_frame(pts, pad_frac)
    axis = np.linspace(-span / 2.0, span / 2.0, int(grid_size))
    xx, yy = np.meshgrid(axis + center[0], axis + center[1])
    query = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = MplPath(np.vstack([pts, pts[0]])).contains_points(query)
    return inside.reshape(int(grid_size), int(grid_size)).astype(np.float32)


# Scalars that express the SPATIAL RELATIONSHIPS a raster is normally brought in
# for: where the drive point sits relative to the plate edge and the hole, how
# wide the body is at those places.  They exist so a spatial CNN branch has to
# beat a handful of distances before it earns its cost -- if these are enough,
# the relationship was the missing thing, not the image.
RELATIONAL_NAMES = [
    "bridge_edge_distance",        # to the outer boundary, in span/2
    "bridge_hole_distance",        # to the soundhole rim, in span/2
    "hole_edge_distance",          # hole centre to the outer boundary
    "bridge_hole_dx", "bridge_hole_dy",
    "body_width_at_bridge_y",      # chord length across the body at the bridge
    "body_width_at_hole_y",
    "bridge_radial_fraction",      # |bridge - centroid| / |edge - centroid|
]
# A second tier, off by default so the eight above stay exactly what the runs
# that measured them saw.  These describe the boundary AS SEEN FROM THE BRIDGE
# and the body's own principal frame -- the same kind of information a spatial
# branch would have to learn to extract, written down instead.
N_BOUNDARY_RAYS = 8


def _extended_names(n_rays: int) -> list[str]:
    return ([f"bridge_ray_{k}" for k in range(int(n_rays))]
            + ["bridge_centroid_distance", "bridge_u", "bridge_v",
               "moment_uu", "moment_vv", "moment_uv",
               "edge_curvature_at_bridge"])


RELATIONAL_EXTENDED_NAMES = _extended_names(N_BOUNDARY_RAYS)
# Ray count is a knob because the rays are the part that measurably worked:
# replacing the original eight scalars with 8 rays plus the principal-frame
# terms moved validation MSE by -0.0145 (t = -16.5).  Whether a finer angular
# sampling of the boundary keeps paying is the obvious next question, and it is
# answered by adding sets rather than by editing a constant.
RELATIONAL_RAY_COUNTS = {"extended": 8, "extended16": 16, "extended32": 32}
RELATIONAL_SETS = {"basic": RELATIONAL_NAMES}
for _name, _n in RELATIONAL_RAY_COUNTS.items():
    RELATIONAL_SETS[_name] = RELATIONAL_NAMES + _extended_names(_n)
RELATIONAL_DIM = len(RELATIONAL_NAMES)
RELATIONAL_EXTENDED_DIM = len(RELATIONAL_SETS["extended"])


def _chord_width(contour: np.ndarray, y_value: float, span: float) -> float:
    """Horizontal extent of the polygon at height ``y``, in units of span/2."""
    pts = np.asarray(contour, float)
    start, end = pts, np.roll(pts, -1, axis=0)
    y0, y1 = start[:, 1], end[:, 1]
    crosses = ((y0 <= y_value) & (y1 > y_value)) | ((y1 <= y_value) & (y0 > y_value))
    if not crosses.any():
        return 0.0
    t = (y_value - y0[crosses]) / np.where(
        np.abs(y1[crosses] - y0[crosses]) < 1e-12, 1e-12, y1[crosses] - y0[crosses])
    x = start[crosses, 0] + t * (end[crosses, 0] - start[crosses, 0])
    return float((x.max() - x.min()) / (span / 2.0))


def _principal_frame(contour: np.ndarray):
    """Centroid and principal axes of the plan area's second-moment tensor.

    Gives the body its own frame, so a bridge position can be expressed the way
    the mode shapes are organised rather than in the mesh's global XY.  Computed
    from the polygon's area moments, not from the vertices, so a denser stretch
    of boundary points does not tilt the axes.
    """
    pts = np.asarray(contour, float)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-12:
        return pts.mean(0), np.eye(2), 1.0
    centroid = np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) \
        / (6.0 * area)
    cx, cy = x - centroid[0], y - centroid[1]
    cx1, cy1 = np.roll(cx, -1), np.roll(cy, -1)
    cross_c = cx * cy1 - cx1 * cy
    ixx = (cross_c * (cy ** 2 + cy * cy1 + cy1 ** 2)).sum() / 12.0
    iyy = (cross_c * (cx ** 2 + cx * cx1 + cx1 ** 2)).sum() / 12.0
    ixy = (cross_c * (cx * cy1 + 2 * cx * cy + 2 * cx1 * cy1
                      + cx1 * cy)).sum() / 24.0
    tensor = np.array([[iyy, ixy], [ixy, ixx]]) / abs(area)
    _values, vectors = np.linalg.eigh(tensor)
    # eigh returns ascending; take the major axis first so the frame is stable.
    axes = vectors[:, ::-1]
    if axes[1, 0] < 0:                      # fix the sign so it cannot flip
        axes = -axes
    return centroid, axes, abs(area)


def _ray_boundary_distances(origin, contour, axes, n_rays: int) -> np.ndarray:
    """Distance from ``origin`` to the boundary along ``n_rays`` fixed directions.

    Directions are taken in the body's PRINCIPAL frame, so ray k means the same
    thing on every body regardless of how the mesh happens to be oriented.  This
    is the boundary as the drive point sees it -- the quantity a spatial branch
    would have to learn to read off the image.
    """
    pts = np.asarray(contour, float)
    start, end = pts, np.roll(pts, -1, axis=0)
    edge = end - start
    out = np.empty(int(n_rays))
    for k in range(int(n_rays)):
        angle = 2.0 * np.pi * k / float(n_rays)
        direction = axes @ np.array([np.cos(angle), np.sin(angle)])
        # Ray-segment intersection: solve origin + t*d = start + u*edge.
        denominator = direction[0] * edge[:, 1] - direction[1] * edge[:, 0]
        safe = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
        delta = start - origin
        t = (delta[:, 0] * edge[:, 1] - delta[:, 1] * edge[:, 0]) / safe
        u = (delta[:, 0] * direction[1] - delta[:, 1] * direction[0]) / safe
        hit = t[(t > 1e-9) & (u >= 0.0) & (u <= 1.0) & np.isfinite(t)]
        out[k] = float(hit.min()) if hit.size else np.nan
    if np.isnan(out).any():                 # a degenerate ray: fall back to the
        finite = out[np.isfinite(out)]      # nearest boundary distance
        fill = float(finite.mean()) if finite.size else 0.0
        out = np.where(np.isfinite(out), out, fill)
    return out


def _edge_curvature(point, contour: np.ndarray, window: int = 3) -> float:
    """Discrete curvature of the boundary at the point nearest ``point``.

    Signed by the local turn, normalised by the polygon's own scale, so a body
    whose nearest edge is a tight waist reads differently from one with a flat
    side even when the distance to it is identical.
    """
    pts = np.asarray(contour, float)
    index = int(np.argmin(np.linalg.norm(pts - np.asarray(point, float), axis=1)))
    n = len(pts)
    a, b, c = (pts[(index - window) % n], pts[index], pts[(index + window) % n])
    ab, cb = a - b, c - b
    cross = ab[0] * cb[1] - ab[1] * cb[0]
    norms = np.linalg.norm(ab) * np.linalg.norm(cb)
    if norms < 1e-12:
        return 0.0
    # sin of the turn angle; +-1 is a hairpin, 0 is a straight edge.
    return float(np.clip(cross / norms, -1.0, 1.0))


def _relational_extended(contour, bridge_mm, half: float,
                         n_rays: int = N_BOUNDARY_RAYS) -> np.ndarray:
    pts = np.asarray(contour, float)
    centroid, axes, area = _principal_frame(pts)
    origin = np.asarray(bridge_mm, float)
    rays = _ray_boundary_distances(origin, pts, axes, int(n_rays)) / half
    offset = origin - centroid
    local = axes.T @ offset                      # bridge in the principal frame
    x, y = pts[:, 0] - centroid[0], pts[:, 1] - centroid[1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    ixx = (cross * (y ** 2 + y * y1 + y1 ** 2)).sum() / 12.0
    iyy = (cross * (x ** 2 + x * x1 + x1 ** 2)).sum() / 12.0
    ixy = (cross * (x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y)).sum() / 24.0
    # Normalised by area^2, which makes them dimensionless SHAPE descriptors
    # rather than another way of writing the body's size.
    scale = max(area ** 2, 1e-12)
    return np.concatenate([
        rays,
        [float(np.linalg.norm(offset) / half), float(local[0] / half),
         float(local[1] / half), float(iyy / scale), float(ixx / scale),
         float(ixy / scale), _edge_curvature(origin, pts)],
    ])


def _relational_features(contour, bridge, hole_norm, hole_diameter,
                         size_features, thickness_mm, has_hole,
                         extended: bool = False,
                         n_rays: int = N_BOUNDARY_RAYS) -> np.ndarray:
    """The relational-scalar control set, all in units of span/2 or unitless."""
    pts = np.asarray(contour, float)
    centre, span = contour_frame(pts)
    half = span / 2.0
    bridge_mm = np.asarray(bridge[:2], float)
    bridge_norm = np.asarray(bridge[3:5], float)
    edge = float(_point_segment_distance(bridge_mm[None, :], pts)[0] / half)
    hole_radius = float(hole_diameter) / 2.0 / half if has_hole else 0.0
    if has_hole:
        offset = bridge_norm - np.asarray(hole_norm, float)
        hole_distance = float(np.linalg.norm(offset)) - hole_radius
        hole_mm = np.asarray(hole_norm, float) * half + centre
        hole_edge = float(_point_segment_distance(hole_mm[None, :], pts)[0] / half)
        hole_y_mm = float(hole_mm[1])
    else:
        # No hole: distances are set to 1.0 (the frame half-width), which is
        # farther than any real rim, and the offsets to 0.  Constant across every
        # solid row, so the column carries no spurious variation.
        offset = np.zeros(2)
        hole_distance, hole_edge, hole_y_mm = 1.0, 1.0, float(centre[1])
    centroid = pts.mean(0)
    direction = bridge_mm - centroid
    reach = np.linalg.norm(direction)
    if reach > 1e-9:
        far = centroid + direction / reach * span
        boundary = float(_point_segment_distance(far[None, :], pts)[0])
        radial = float(reach / max(reach + boundary, 1e-9))
    else:
        radial = 0.0
    basic = np.asarray([
        edge, hole_distance, hole_edge, offset[0], offset[1],
        _chord_width(pts, float(bridge_mm[1]), span),
        _chord_width(pts, hole_y_mm, span),
        radial,
    ], dtype=np.float64)
    if not extended:
        return basic
    return np.concatenate([basic,
                           _relational_extended(pts, bridge_mm, half, n_rays)])


def _point_segment_distance(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to the closed polygon's boundary."""
    start = contour
    end = np.roll(contour, -1, axis=0)
    edge = end - start                                        # (M, 2)
    length2 = np.maximum((edge ** 2).sum(1), 1e-30)           # (M,)
    delta = points[:, None, :] - start[None, :, :]            # (N, M, 2)
    t = np.clip((delta * edge[None, :, :]).sum(-1) / length2[None, :], 0.0, 1.0)
    closest = start[None, :, :] + t[:, :, None] * edge[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(1)


def contour_to_sdf(contour: np.ndarray, grid_size: int = DEFAULT_GRID_SIZE,
                   pad_frac: float = 0.05) -> np.ndarray:
    """Signed distance to the outline, POSITIVE inside, in units of span/2.

    Same isotropic frame as ``contour_to_grid``, so the two channels stack.  A
    binary occupancy grid is flat everywhere except at the boundary, so every
    gradient the encoder can get about the outline lives in one pixel-wide ring;
    a signed distance field spreads that information over the whole plate, which
    is also where the modes it controls actually live.

    The scale is span/2 rather than millimetres deliberately: absolute size is
    carried by SIZE_COLS, and putting it back into the image would undo the
    normalisation ``contour_to_grid`` exists to apply.
    """
    from matplotlib.path import Path as MplPath

    pts = np.asarray(contour, float)
    center, span = contour_frame(pts, pad_frac)
    axis = np.linspace(-span / 2.0, span / 2.0, int(grid_size))
    xx, yy = np.meshgrid(axis + center[0], axis + center[1])
    query = np.stack([xx.ravel(), yy.ravel()], axis=1)
    distance = _point_segment_distance(query, pts)
    inside = MplPath(np.vstack([pts, pts[0]])).contains_points(query)
    signed = np.where(inside, distance, -distance) / (span / 2.0)
    return signed.reshape(int(grid_size), int(grid_size)).astype(np.float32)


def soundhole_to_sdf(center_xy, diameter_mm: float, contour: np.ndarray,
                     grid_size: int = DEFAULT_GRID_SIZE,
                     signed: bool = True) -> np.ndarray:
    """Signed distance to the soundhole rim, POSITIVE inside the hole.

    The soundhole is not part of ``contour.npy`` -- that file is a single closed
    polygon and cannot represent a hole -- so the solid and hollow images of one
    base shape are pixel-identical and the hole reaches the network only as
    scalars.  This is the channel that changes that.  Solid rows have no hole
    and get a constant -1 (everywhere outside a hole of zero size).
    """
    _center, span = contour_frame(np.asarray(contour, float))
    grid = int(grid_size)
    if center_xy is None or not np.isfinite(diameter_mm) or diameter_mm <= 0:
        return np.full((grid, grid), -1.0, dtype=np.float32)
    hole = normalized_xy(center_xy, contour)                  # in [-1, 1]
    radius = float(diameter_mm) / 2.0 / (span / 2.0)
    axis = np.linspace(-1.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis)
    distance = np.hypot(xx - hole[0], yy - hole[1])
    field = radius - distance if signed else (distance <= radius).astype(float)
    return np.clip(field, -1.0, 1.0).astype(np.float32)


# Channel stacks the shape encoder can be given.  "occupancy" is the historical
# single binary channel and stays the default so every existing checkpoint and
# every earlier number remains reproducible.
# How a row is paired with an image.  ``correct`` is the real one; the other
# two are controls for what a shuffled-raster ablation actually shows.
SHAPE_PAIRINGS = ("correct", "fixed_shuffle", "constant")

# How the TARGET is standardized.  "global" is one scalar over every bin and
# every sample (the historical default, and what every existing checkpoint and
# reported number assumes).
TARGET_NORM_MODES = ("global", "per_freq", "per_freq_body")

SHAPE_CHANNEL_SETS = {
    "occupancy": ("occupancy",),
    "sdf": ("sdf",),
    "occupancy_soundhole": ("occupancy", "soundhole_sdf"),
    "sdf_soundhole": ("sdf", "soundhole_sdf"),
    "sdf_occupancy_soundhole": ("sdf", "occupancy", "soundhole_sdf"),
}


def normalized_xy(point_xy, contour: np.ndarray) -> np.ndarray:
    """Map a millimetre point into the shape grid's [-1, 1] frame."""
    center, span = contour_frame(contour)
    return (np.asarray(point_xy, float).reshape(2) - center) / (span / 2.0)


def normalized_bridge(bridge_xyz, contour: np.ndarray,
                      thickness_mm: float) -> np.ndarray:
    """``(x, y, z)`` mm concatenated with ``(x, y)`` in the shape-grid frame and
    ``z`` as a fraction of body thickness.  The normalised pair lands in [-1, 1]
    over the grid's extent, so it indexes the rasterised shape directly."""
    bridge = np.asarray(bridge_xyz, float).reshape(-1)[:3]
    center, span = contour_frame(contour)
    xy_norm = (bridge[:2] - center) / (span / 2.0)
    thickness = float(thickness_mm)
    z_norm = bridge[2] / thickness if np.isfinite(thickness) and thickness > 0 else 0.0
    return np.concatenate([bridge, xy_norm, [z_norm]]).astype(np.float64)


class NormStats:
    """Training-split normalization for continuous inputs and magnitude target."""

    def __init__(self):
        self.mat_mean = self.mat_std = None
        self.br_mean = self.br_std = None
        self.geom_mean = self.geom_std = None
        self.y_mean = self.y_std = None
        # Peak-label normalization.  Frequency is normalized against the analysis
        # band rather than against data statistics: the band is a fixed property
        # of the production contract, so the mapping stays identical across
        # datasets and a checkpoint's peak predictions remain interpretable.
        self.f_log_min = self.f_log_max = None
        self.pk_amp_mean = self.pk_amp_std = None
        self.pk_logq_mean = self.pk_logq_std = None
        # Per-bin target statistics; None unless fit_target_mode asks for them.
        self.y_mode = "global"
        self.y_bin_mean = self.y_bin_std = None

    def fit(self, samples: list[dict]):
        if not samples:
            raise ValueError("cannot fit normalization on an empty training split")
        mats = np.asarray([s["material"] for s in samples], float)
        bridges = np.asarray([s["bridge"] for s in samples], float)
        geometry = np.asarray([s["geometry"] for s in samples], float)
        targets = np.asarray([s["log_magnitude_db"] for s in samples], float)
        self.mat_mean, self.mat_std = mats.mean(0), mats.std(0) + 1e-8
        self.br_mean, self.br_std = bridges.mean(0), bridges.std(0) + 1e-8
        self.geom_mean, self.geom_std = geometry.mean(0), geometry.std(0) + 1e-8
        self.y_mean = float(targets.mean())
        self.y_std = float(targets.std()) + 1e-8
        self._fit_peaks(samples)

    def _fit_peaks(self, samples: list[dict]):
        freqs = np.asarray(samples[0]["freqs"], float)
        self.f_log_min = float(np.log10(freqs.min()))
        self.f_log_max = float(np.log10(freqs.max()))
        amps, qs = [], []
        for s in samples:
            mask = np.asarray(s["peak_mask"], bool)
            if not mask.any():
                continue
            amps.append(np.asarray(s["peak_amplitude_db"], float)[mask])
            qs.append(np.asarray(s["peak_q"], float)[mask])
        if not amps:
            # No labelled peak anywhere in the training split.  The peak head is
            # then untrainable, but normalization must still be well defined so
            # the rest of the pipeline runs; the loss masks every slot out.
            self.pk_amp_mean, self.pk_amp_std = self.y_mean, self.y_std
            self.pk_logq_mean, self.pk_logq_std = 0.0, 1.0
            return
        amp = np.concatenate(amps)
        logq = np.log10(np.clip(np.concatenate(qs), 1e-6, None))
        self.pk_amp_mean = float(amp.mean())
        self.pk_amp_std = float(amp.std()) + 1e-8
        self.pk_logq_mean = float(logq.mean())
        self.pk_logq_std = float(logq.std()) + 1e-8

    def norm_mat(self, value):
        return (value - self.mat_mean) / self.mat_std

    def norm_bridge(self, value):
        return (value - self.br_mean) / self.br_std

    def norm_geometry(self, value):
        return (value - self.geom_mean) / self.geom_std

    def fit_target_mode(self, samples: list[dict], mode: str,
                        std_floor_fraction: float = 0.1):
        """Optional PER-BIN target statistics.

        One global scale means the loss is dominated by whichever part of the
        band happens to vary most in dB -- and the response is far more
        energetic at low frequency than at 3 kHz, so the bins where resonances
        are dense and contour-sensitive contribute least.  Standardizing each
        bin gives every part of the band comparable say.

        ``per_freq_body`` uses separate statistics for solid and hollow.  Body
        type is a known INPUT at inference, so conditioning on it is not
        leakage; the two have genuinely different spectral envelopes.

        The floor is a fraction of the MEAN per-bin std: without it a quiet bin
        gets divided by a near-zero number and its noise becomes the loss.
        """
        if mode not in TARGET_NORM_MODES:
            raise ValueError(f"target_norm must be one of {TARGET_NORM_MODES}")
        self.y_mode = str(mode)
        if mode == "global":
            self.y_bin_mean = self.y_bin_std = None
            return
        targets = np.asarray([s["log_magnitude_db"] for s in samples], float)
        bodies = np.asarray([s["body_type"] for s in samples])
        groups = ("all",) if mode == "per_freq" else ("solid", "hollow")
        self.y_bin_mean, self.y_bin_std = {}, {}
        for group in groups:
            rows = (np.ones(len(targets), bool) if group == "all"
                    else bodies == group)
            if not rows.any():                       # fall back to everything
                rows = np.ones(len(targets), bool)
            block = targets[rows]
            std = block.std(0)
            self.y_bin_mean[group] = block.mean(0)
            self.y_bin_std[group] = np.maximum(
                std, float(std_floor_fraction) * float(std.mean()) + 1e-8)

    def _bin_stats(self, body_type):
        if self.y_mode == "per_freq":
            return self.y_bin_mean["all"], self.y_bin_std["all"]
        key = body_type if body_type in self.y_bin_mean else "solid"
        return self.y_bin_mean[key], self.y_bin_std[key]

    def norm_y(self, value, body_type=None):
        if self.y_mode == "global":
            return (value - self.y_mean) / self.y_std
        mean, std = self._bin_stats(body_type)
        return (value - mean) / std

    def denorm_y(self, value, body_type=None):
        if self.y_mode == "global":
            return value * self.y_std + self.y_mean
        mean, std = self._bin_stats(body_type)
        return np.asarray(value, float) * std + mean

    @property
    def y_scale_db(self) -> float:
        """One number for reporting dB errors, whatever the mode."""
        if self.y_mode == "global":
            return float(self.y_std)
        return float(np.mean([s.mean() for s in self.y_bin_std.values()]))

    # -- peak labels --------------------------------------------------------
    # Frequency is carried as a normalized LOG frequency: resonances are spaced
    # geometrically and the analysis band spans 20 Hz - 5 kHz, so a linear-Hz
    # target would make one 4 kHz slot outweigh every slot below 400 Hz.

    def norm_peak_freq(self, value):
        span = self.f_log_max - self.f_log_min
        return (np.log10(np.clip(value, 1e-6, None)) - self.f_log_min) / span

    def denorm_peak_freq(self, value):
        span = self.f_log_max - self.f_log_min
        return 10.0 ** (np.asarray(value, float) * span + self.f_log_min)

    def norm_peak_amp(self, value):
        return (value - self.pk_amp_mean) / self.pk_amp_std

    def denorm_peak_amp(self, value):
        return np.asarray(value, float) * self.pk_amp_std + self.pk_amp_mean

    def norm_peak_logq(self, value):
        logq = np.log10(np.clip(value, 1e-6, None))
        return (logq - self.pk_logq_mean) / self.pk_logq_std

    def denorm_peak_q(self, value):
        logq = np.asarray(value, float) * self.pk_logq_std + self.pk_logq_mean
        return 10.0 ** logq

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mat_mean=self.mat_mean, mat_std=self.mat_std,
                 br_mean=self.br_mean, br_std=self.br_std,
                 geom_mean=self.geom_mean, geom_std=self.geom_std,
                 y_mean=np.asarray([self.y_mean]), y_std=np.asarray([self.y_std]),
                 y_mode=np.asarray([self.y_mode]),
                 **({} if self.y_bin_mean is None else {
                     f"y_bin_{stat}_{group}": value
                     for stat, table in (("mean", self.y_bin_mean),
                                         ("std", self.y_bin_std))
                     for group, value in table.items()}),
                 peak_scalars=np.asarray([
                     self.f_log_min, self.f_log_max,
                     self.pk_amp_mean, self.pk_amp_std,
                     self.pk_logq_mean, self.pk_logq_std]))

    @classmethod
    def load(cls, path: Path):
        result = cls()
        with np.load(path, allow_pickle=False) as data:
            result.mat_mean = data["mat_mean"].copy()
            result.mat_std = data["mat_std"].copy()
            result.br_mean = data["br_mean"].copy()
            result.br_std = data["br_std"].copy()
            result.geom_mean = data["geom_mean"].copy()
            result.geom_std = data["geom_std"].copy()
            result.y_mean = float(data["y_mean"][0])
            result.y_std = float(data["y_std"][0])
            # Written since the peak head was added.  A checkpoint trained before
            # that has no peak scalars; loading it for magnitude-only inference is
            # still valid, so this is optional rather than fatal.
            if "y_mode" in data.files:
                result.y_mode = str(np.asarray(data["y_mode"]).ravel()[0])
                if result.y_mode != "global":
                    result.y_bin_mean, result.y_bin_std = {}, {}
                    for name in data.files:
                        if name.startswith("y_bin_mean_"):
                            result.y_bin_mean[name[11:]] = data[name].copy()
                        elif name.startswith("y_bin_std_"):
                            result.y_bin_std[name[10:]] = data[name].copy()
            if "peak_scalars" in data.files:
                scalars = [float(v) for v in data["peak_scalars"]]
                (result.f_log_min, result.f_log_max,
                 result.pk_amp_mean, result.pk_amp_std,
                 result.pk_logq_mean, result.pk_logq_std) = scalars
        return result

    @property
    def has_peak_scalars(self) -> bool:
        return self.f_log_min is not None


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------
# The material axes are constant across every sample (L=+Y, R=+X, T=+Z) and an
# orthotropic stiffness tensor whose principal axes lie along the coordinate
# axes is INVARIANT under reflection about any coordinate plane.  Reflecting the
# contour, the bridge point and the soundhole together therefore produces a body
# with the identical admittance -- this is an exact symmetry of the physics, not
# an approximation, so the reflected copies carry the SAME target with no FEM
# cost.  It is the only augmentation available here that needs no argument about
# how much distortion is tolerable.
#
# Rotation and rescaling are NOT available: a rotation would take the material
# axes out of alignment, and a rescale changes every eigenfrequency.
MIRROR_TRANSFORMS = ("identity", "flip_x", "flip_y", "flip_xy")


def _mirror_contour(contour: np.ndarray, transform: str) -> np.ndarray:
    """Reflect about the contour frame's own centre, so the frame is preserved."""
    pts = np.asarray(contour, float).copy()
    centre, _span = contour_frame(pts)
    if "x" in transform:
        pts[:, 0] = 2.0 * centre[0] - pts[:, 0]
    if "y" in transform:
        pts[:, 1] = 2.0 * centre[1] - pts[:, 1]
    if transform != "identity":
        # Reflection reverses the winding; restore it so downstream area and
        # point-in-polygon tests see the same orientation they always have.
        pts = pts[::-1].copy()
    return pts


def _mirror_point(point_xy, contour: np.ndarray, transform: str) -> np.ndarray:
    centre, _span = contour_frame(contour)
    out = np.asarray(point_xy, float).copy()
    if "x" in transform:
        out[0] = 2.0 * centre[0] - out[0]
    if "y" in transform:
        out[1] = 2.0 * centre[1] - out[1]
    return out


def mirrored_record(record: dict, transform: str) -> dict:
    """A reflected copy of one sample record, target unchanged.

    Everything derived from position is recomputed from the reflected contour
    rather than patched in place: the relational scalars alone involve chord
    widths, ray casts and a principal frame, and hand-flipping each of them is
    how a sign error gets into training data that still looks plausible.
    """
    if transform == "identity":
        return record
    out = dict(record)
    contour = np.load(record["contour_path"], allow_pickle=False)
    mirrored = _mirror_contour(contour, transform)
    out["mirror"] = transform
    # A reflected body is a DIFFERENT geometry as far as grouping is concerned,
    # but it must stay on the same side of the split as its original, so the key
    # is derived from the original rather than from the reflected contour.
    out["contour_digest"] = f"{record['contour_digest']}:{transform}"
    out["mirrored_contour"] = mirrored

    bridge = np.asarray(record["bridge"], float).copy()
    bridge[:2] = _mirror_point(bridge[:2], contour, transform)
    xy_norm = normalized_xy(bridge[:2], mirrored)
    bridge[3:5] = xy_norm
    out["bridge"] = bridge

    geometry = np.asarray(record["geometry"], float).copy()
    hole_mm = record.get("soundhole_center_mm")
    if hole_mm is not None:
        hole_mm = list(_mirror_point(hole_mm, contour, transform))
        hole_norm = normalized_xy(hole_mm, mirrored)
        # GEOMETRY_COLS positions of the absolute soundhole centre, then the
        # normalised pair appended by DERIVED_COLS.
        geometry[GEOMETRY_COLS.index("soundhole_center_x")] = hole_mm[0]
        geometry[GEOMETRY_COLS.index("soundhole_center_y")] = hole_mm[1]
        geometry[-2:] = hole_norm
        out["soundhole_center_mm"] = hole_mm
        out["soundhole_key"] = (round(hole_mm[0], 6), round(hole_mm[1], 6),
                                round(record.get("soundhole_diameter_mm", 0.0), 6))
    else:
        hole_norm = np.zeros(2)
    out["geometry"] = geometry
    out["bridge_extra"] = _relational_features(
        mirrored, bridge, hole_norm,
        record.get("soundhole_diameter_mm", 0.0), np.zeros(3),
        float(geometry[GEOMETRY_COLS.index("thickness")]),
        hole_mm is not None,
        extended=(record["bridge_extra"].size > RELATIONAL_DIM),
        n_rays=max(record["bridge_extra"].size - RELATIONAL_DIM - 7, 0)
        or N_BOUNDARY_RAYS)
    out["sample_id"] = f"{record['sample_id']}#{transform}"
    return out


def augment_records(records: list[dict], transforms) -> list[dict]:
    """Expand a record list with its reflections.  Targets are untouched."""
    wanted = [t for t in transforms if t in MIRROR_TRANSFORMS]
    if wanted == ["identity"] or not wanted:
        return records
    out = []
    for record in records:
        for transform in wanted:
            out.append(mirrored_record(record, transform))
    return out


class AdmittanceDataset:
    """Return ``(inputs, normalized_log_magnitude, peak_targets)`` per bridge."""

    def __init__(self, samples: list[dict], stats: NormStats,
                 grid_size: int = DEFAULT_GRID_SIZE, *,
                 shape_vocab: dict | None = None,
                 shape_key_mode: str = "path",
                 shape_channels: str = "occupancy",
                 shape_pairing: str = "correct", pairing_seed: int = 0):
        self.samples = samples
        self.stats = stats
        self.grid_size = int(grid_size)
        # Train-shape index for the shape-ID diagnostic model.  Always present in
        # the input mapping so every model variant sees the same keys; models
        # that do not want it simply never read it.
        self.shape_vocab = dict(shape_vocab or {})
        self.shape_key_mode = str(shape_key_mode)
        self.unknown_shape_index = len(self.shape_vocab)
        if shape_channels not in SHAPE_CHANNEL_SETS:
            raise ValueError(f"shape_channels must be one of "
                             f"{sorted(SHAPE_CHANNEL_SETS)}")
        self.shape_channels = str(shape_channels)
        self.channel_names = SHAPE_CHANNEL_SETS[self.shape_channels]
        # How each row is paired with an image.  ``fixed_shuffle`` draws the
        # wrong contour ONCE and keeps it for the whole run, which separates
        # "the image is unused" from "re-drawing it every batch is a useful
        # stochastic regulariser" -- the dynamic shuffle confounds the two.
        if shape_pairing not in SHAPE_PAIRINGS:
            raise ValueError(f"shape_pairing must be one of {SHAPE_PAIRINGS}")
        self.shape_pairing = str(shape_pairing)
        self._constant_image = None
        self._pairing: dict[str, str] = {}
        if self.shape_pairing == "fixed_shuffle":
            keys = sorted({s["contour_path"] for s in samples})
            rng = np.random.default_rng(int(pairing_seed))
            shuffled = list(keys)
            for _ in range(16):                      # avoid an identity draw
                rng.shuffle(shuffled)
                if all(a != b for a, b in zip(keys, shuffled)) or len(keys) < 2:
                    break
            self._pairing = dict(zip(keys, shuffled))
        self.case_vocab = {key: index for index, key in
                           enumerate(sorted({s["case_key"] for s in samples}))}
        # One contour is shared by every bridge of every material of a shape, so
        # ~20k samples resolve to ~200 distinct grids.  Rasterising per __getitem__
        # costs ~57 s per epoch at 96x96; memoised it is ~0.6 s once, for ~7 MB.
        # Keyed by path so it stays correct if two shapes ever share a contour.
        self._grid_cache: dict[str, np.ndarray] = {}
        # Soundhole planes vary per (contour, hole) rather than per file.
        self._hole_cache: dict = {}

    def case_batches(self, cases_per_batch: int, shuffle: bool = True,
                     seed: int = 0) -> list[list[int]]:
        """Index lists that keep every case WHOLE inside one batch.

        The case/bridge decomposition subtracts a per-case mean, and a case
        split across two batches would have that mean estimated from a fraction
        of its bridges -- a different quantity in every batch, which the model
        cannot fit and the zero-mean constraint would fight.
        """
        by_case: dict = {}
        for index, sample in enumerate(self.samples):
            by_case.setdefault(sample["case_key"], []).append(index)
        cases = sorted(by_case)
        if shuffle:
            np.random.default_rng(int(seed)).shuffle(cases)
        batches, current = [], []
        for case in cases:
            current.append(case)
            if len(current) >= int(cases_per_batch):
                batches.append([i for c in current for i in by_case[c]])
                current = []
        if current:
            batches.append([i for c in current for i in by_case[c]])
        return batches

    def __len__(self):
        return len(self.samples)

    def _shape_grid(self, contour_path: str, contour=None) -> np.ndarray:
        """Contour-only channels, memoised per key (shared across ~10k rows)."""
        grid = self._grid_cache.get(contour_path)
        if grid is None:
            contour = (np.load(contour_path, allow_pickle=False)
                       if contour is None else np.asarray(contour, float))
            planes = []
            for name in self.channel_names:
                if name == "occupancy":
                    planes.append(contour_to_grid(contour, self.grid_size))
                elif name == "sdf":
                    planes.append(contour_to_sdf(contour, self.grid_size))
                else:
                    # Soundhole depends on the SAMPLE, not the contour file, so
                    # it is filled in per row by _shape_stack below.  A
                    # placeholder keeps the channel order fixed.
                    planes.append(np.zeros((self.grid_size, self.grid_size),
                                           dtype=np.float32))
            grid = np.stack(planes).astype(np.float32)
            grid.flags.writeable = False       # the cached master is never mutated
            self._grid_cache[contour_path] = grid
        # Hand back a writable copy: callers own their array, and torch.from_numpy
        # warns on (and aliases) a read-only buffer.  36 kB vs 2.9 ms to rasterize.
        return np.array(grid, copy=True)

    def set_constant_image(self, image) -> None:
        """Image every row receives under ``shape_pairing="constant"``.

        Supplied from outside (the TRAIN mean) rather than computed here, so a
        validation set cannot quietly build its own mean and make the control
        measure something different from the training condition.
        """
        self._constant_image = np.asarray(image, dtype=np.float32)

    def _contour_of(self, sample: dict):
        """The polygon this row actually uses -- reflected when augmented."""
        return sample.get("mirrored_contour")

    def _cache_key(self, sample: dict) -> str:
        mirror = sample.get("mirror")
        return (sample["contour_path"] if not mirror
                else f"{sample['contour_path']}#{mirror}")

    def _shape_stack(self, sample: dict) -> np.ndarray:
        """The full channel stack for one row, soundhole included."""
        if self.shape_pairing == "constant":
            if self._constant_image is None:
                raise RuntimeError("shape_pairing='constant' needs "
                                   "set_constant_image() first")
            return np.array(self._constant_image, copy=True)
        if self.shape_pairing == "fixed_shuffle":
            path = self._pairing.get(sample["contour_path"],
                                     sample["contour_path"])
            stack = self._shape_grid(path)
            # The soundhole is a property of the CASE, not of the borrowed
            # outline, so it stays with the row: the control swaps the contour,
            # not the body type.
            if "soundhole_sdf" in self.channel_names:
                stack[self.channel_names.index("soundhole_sdf")] =                     self._soundhole_plane(sample)
            return stack
        stack = self._shape_grid(self._cache_key(sample),
                                 self._contour_of(sample))
        if "soundhole_sdf" not in self.channel_names:
            return stack
        stack[self.channel_names.index("soundhole_sdf")] =             self._soundhole_plane(sample)
        return stack

    def _soundhole_plane(self, sample: dict) -> np.ndarray:
        key = (self._cache_key(sample), sample.get("soundhole_key"))
        plane = self._hole_cache.get(key)
        if plane is None:
            contour = self._contour_of(sample)
            if contour is None:
                contour = np.load(sample["contour_path"], allow_pickle=False)
            plane = soundhole_to_sdf(sample.get("soundhole_center_mm"),
                                     sample.get("soundhole_diameter_mm", 0.0),
                                     contour, self.grid_size)
            self._hole_cache[key] = plane
        return plane

    def numpy_item(self, index):
        """Return the exact sample tuple as NumPy arrays for schema/QC use."""
        sample = self.samples[index]
        inputs = {
            "shape": self._shape_stack(sample),
            "bridge_extra": np.asarray(
                sample["bridge_extra"], dtype=np.float32),
            "material": self.stats.norm_mat(sample["material"]).astype(np.float32),
            "bridge": self.stats.norm_bridge(sample["bridge"]).astype(np.float32),
            "geometry": self.stats.norm_geometry(sample["geometry"]).astype(np.float32),
            "body_type": sample["body_one_hot"].astype(np.float32),
            "shape_index": np.asarray(
                self.shape_vocab.get(_shape_key(sample, self.shape_key_mode),
                                     self.unknown_shape_index),
                dtype=np.int64),
            # Global case id.  The case/bridge decomposition needs to know which
            # rows share a (shape, material) case; carrying an id per row and
            # grouping inside the loss keeps the batch FLAT, so every existing
            # metric, sampler and eval path keeps working unchanged.
            "case_index": np.asarray(
                self.case_vocab.get(sample["case_key"], -1), dtype=np.int64),
        }
        magnitude = self.stats.norm_y(
            sample["log_magnitude_db"], sample["body_type"]).astype(np.float32)
        mask = sample["peak_mask"].astype(bool)
        peaks = {
            "frequency_hz": sample["peak_frequency_hz"].astype(np.float32),
            "amplitude_db": sample["peak_amplitude_db"].astype(np.float32),
            "q": sample["peak_q"].astype(np.float32),
            "mask": mask,
            "count_total": np.asarray(sample["peak_count_total"], dtype=np.int64),
            "truncated": np.asarray(sample["peak_truncated"], dtype=bool),
            # Carried alongside the labels because the peak loss has to be able to
            # gate on body type: the frequency-ordered slots are stable for solid
            # and meaningless for hollow (see SurrogateLoss.peak_term).
            "is_solid": np.asarray(
                sample["body_type"] == "solid", dtype=np.float32),
        }
        peaks.update(self._normalized_peaks(sample, mask))
        for kind in ("peak", "valley"):
            if f"{kind}_heatmap" in sample:
                peaks[f"{kind}_heatmap"] = sample[f"{kind}_heatmap"]
                peaks[f"{kind}_event_count"] = np.asarray(
                    sample[f"{kind}_event_count"], dtype=np.float32)
        return inputs, magnitude, peaks

    def _normalized_peaks(self, sample: dict, mask: np.ndarray) -> dict:
        """Normalized regression targets for the auxiliary peak head.

        Unused slots are written as 0.0 rather than left at their raw value: they
        are excluded from the loss by ``mask``, and a real number there would make
        an unmasked bug look plausible instead of obviously wrong.
        """
        stats = self.stats
        if not stats.has_peak_scalars:
            zeros = np.zeros(mask.shape, dtype=np.float32)
            return {"freq_norm": zeros, "amp_norm": zeros.copy(),
                    "logq_norm": zeros.copy()}
        keep = mask.astype(bool)
        freq = np.where(keep, sample["peak_frequency_hz"], 1.0)
        amp = np.where(keep, sample["peak_amplitude_db"], stats.pk_amp_mean)
        q = np.where(keep, sample["peak_q"], 1.0)
        return {
            "freq_norm": np.where(
                keep, stats.norm_peak_freq(freq), 0.0).astype(np.float32),
            "amp_norm": np.where(
                keep, stats.norm_peak_amp(amp), 0.0).astype(np.float32),
            "logq_norm": np.where(
                keep, stats.norm_peak_logq(q), 0.0).astype(np.float32),
        }

    def __getitem__(self, index):
        """Convert ``numpy_item`` to tensors at the PyTorch boundary."""
        import torch

        inputs, magnitude, peaks = self.numpy_item(index)
        return (
            {name: torch.from_numpy(value) for name, value in inputs.items()},
            torch.from_numpy(magnitude),
            {name: torch.from_numpy(value) for name, value in peaks.items()},
        )


EVENT_HEATMAP_KINDS = ("none", "peak", "valley", "both")


def attach_event_heatmaps(samples, *, kinds: str = "both",
                          sigma_cents: float = 50.0,
                          prominence_db: float = 1.5) -> None:
    """Precompute per-bin peak/valley heatmaps, IN DECIBELS, once.

    Two reasons this happens here and not in the loss.

    Cost: detecting events on every curve every epoch would mean a CPU round
    trip per batch.  Done once at load it is seconds.

    Correctness: a resonance is at a frequency, which is a fact about the body,
    not about the run's ``--target-norm``.  Building the heatmap from the dB
    curve makes the supervision identical across normalisation modes, so an
    arm that changes the normalisation is not also silently changing what it is
    being taught to find.  (Under ``per_freq`` the per-bin scale does not
    commute with peak detection -- that is the same bug class that made three
    arms report 11.1, 12.3 and 12.9 target peaks on identical data.)

    The heatmap is a Gaussian bump of ``sigma_cents`` at each detected event,
    clipped to 1.  It is a soft PRESENCE target, per the plan; amplitude
    regression on top of it is a later step.
    """
    if kinds not in EVENT_HEATMAP_KINDS:
        raise ValueError(f"event_heatmaps must be one of {EVENT_HEATMAP_KINDS}")
    from spectral_metrics import cents_axis, cents_per_bin, detect_events

    if not samples:
        return
    freqs = np.asarray(samples[0]["freqs"], float)
    axis = cents_axis(freqs)
    sigma = max(float(sigma_cents), 1e-6)
    wanted = ("peak", "valley") if kinds == "both" else (kinds,)
    for sample in samples:
        curve = np.asarray(sample["log_magnitude_db"], float)
        for kind in wanted:
            events = detect_events(curve, freqs, kind=kind,
                                   prominence_db=prominence_db)
            heatmap = np.zeros(freqs.size, dtype=np.float32)
            for centre in events["cents"]:
                heatmap = np.maximum(
                    heatmap,
                    np.exp(-0.5 * ((axis - centre) / sigma) ** 2))
            sample[f"{kind}_heatmap"] = heatmap.astype(np.float32)
            sample[f"{kind}_event_count"] = np.float32(len(events["cents"]))


def _finite_or_zero(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _load_contour(path: Path, cache: dict | None) -> np.ndarray:
    """Load a contour, memoising by path (contours are shared across samples)."""
    key = ("contour", str(path))
    if cache is not None and key in cache:
        return cache[key]
    contour = np.load(path, allow_pickle=False)
    if cache is not None:
        cache[key] = contour
    return contour


def contour_digest(contour: np.ndarray) -> str:
    """Content hash of a contour polygon.

    The solid and the hollow member of one base shape are written to SEPARATE
    files (``shape_0000`` / ``shape_0001``) whose contents are byte-identical:
    the soundhole is not part of the outer polygon, so both bodies rasterize to
    the same image.  Keying a "shape" by file path therefore counts one geometry
    twice, which doubles every reported shape count and makes a shape-axis
    learning curve advance in steps of half a geometry.  Hashing the array is
    what makes ``shape_key_mode="contour"`` count distinct GEOMETRIES.
    """
    array = np.ascontiguousarray(np.asarray(contour, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:32]


_RESPONSE_KEYS = ("frequencies_hz", "log_magnitude_db", "peak_frequency_hz",
                  "peak_amplitude_db", "peak_q", "peak_mask",
                  "peak_count_total", "peak_truncated")


def _load_case_response(path: Path, cache: dict | None) -> dict | None:
    """Read one case-level NN artifact, memoised ONE case deep.

    Every bridge of a case shares this file, and manifest rows arrive grouped by
    case, so a single slot removes ~10x redundant decompression while holding at
    most one case (~60 kB) in memory.
    """
    key = str(path)
    if cache is not None:
        slot = cache.get("_response_slot")
        if slot is not None and slot[0] == key:
            return slot[1]
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["schema_version"]).item()) != "magnitude-peaks-v1":
            return None
        arrays = {name: np.asarray(data[name]).copy() for name in _RESPONSE_KEYS}
    if cache is not None:
        cache["_response_slot"] = (key, arrays)
    return arrays


def _load_sample_record(dataset_dir: Path, manifest_row,
                        contour_cache: dict | None = None,
                        extended_relational: int = 0) -> dict | None:
    """Load one bridge row from its shared case-level NN artifact."""
    # Schema first, before any I/O: a column that is PRESENT but null is
    # legitimate (solid rows have no soundhole, so soundhole_center_x is None ->
    # 0.0), but a column that is ABSENT is a generator-version mismatch.  mixed-v4
    # manifests carry cavity_ratio instead of top_plate_thickness_mm, and
    # defaulting it to 0.0 would silently relabel every v4 hollow body as having
    # no top plate — so this fails loudly instead of returning None, which the
    # caller would otherwise treat as one more skipped row.
    missing = [name for name in GEOMETRY_COLS if name not in manifest_row]
    if missing:
        raise RuntimeError(
            f"manifest is missing geometry column(s) {missing}; this dataset was "
            f"produced by an incompatible generator version and needs explicit "
            f"harmonization rather than a silent default")
    rel_response = str(manifest_row.get("rel_case_response", ""))
    response_path = dataset_dir / rel_response
    if not rel_response or not response_path.is_file():
        return None
    try:
        bridge_index = int(manifest_row["bridge_idx"])
        data = _load_case_response(response_path, contour_cache)
        if data is None:
            return None
        frequencies = np.asarray(data["frequencies_hz"], float).copy()
        log_magnitude = np.asarray(
            data["log_magnitude_db"], float)[bridge_index].copy()
        peak_frequency = np.asarray(
            data["peak_frequency_hz"], float)[bridge_index].copy()
        peak_amplitude = np.asarray(
            data["peak_amplitude_db"], float)[bridge_index].copy()
        peak_q = np.asarray(data["peak_q"], float)[bridge_index].copy()
        peak_mask = np.asarray(data["peak_mask"], bool)[bridge_index].copy()
        peak_count = int(np.asarray(data["peak_count_total"])[bridge_index])
        peak_truncated = bool(np.asarray(data["peak_truncated"])[bridge_index])
    except Exception:
        return None

    contour_path = dataset_dir / str(manifest_row.get("rel_contour", ""))
    params_path = dataset_dir / str(manifest_row.get("rel_params", ""))
    if not contour_path.is_file() or not params_path.is_file():
        return None
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        material = np.asarray(
            [manifest_row[name] for name in MATERIAL_COLS], dtype=np.float64)
        # One contour is shared by every bridge of every material of a shape, so
        # read it through a cache: 20k samples resolve to ~200 distinct files.
        contour = _load_contour(contour_path, contour_cache)
        digest = contour_digest(contour)
        size_features = contour_size_features(contour)
        bridge = normalized_bridge(
            params["bridge_coords"], contour,
            _finite_or_zero(manifest_row.get("thickness", 0.0)))
    except Exception:
        return None
    body_type = str(manifest_row.get("body_type", ""))
    if body_type not in ("solid", "hollow"):
        return None
    # A solid body has no hole: its soundhole_* columns are 0.0 and its centre is
    # null, so the normalised pair is zeroed rather than mapped from (0, 0) — which
    # would otherwise read as "a hole at the middle of the plate".
    hole_center_mm = None
    if body_type == "hollow" and manifest_row.get("soundhole_center_x") is not None:
        hole_center_mm = [_finite_or_zero(manifest_row["soundhole_center_x"]),
                          _finite_or_zero(manifest_row["soundhole_center_y"])]
        hole_norm = normalized_xy(hole_center_mm, contour)
    else:
        hole_norm = np.zeros(2)
    hole_diameter = (_finite_or_zero(manifest_row.get("soundhole_diameter", 0.0))
                     if body_type == "hollow" else 0.0)
    relational = _relational_features(contour, bridge, hole_norm, hole_diameter,
                                      size_features,
                                      _finite_or_zero(manifest_row.get(
                                          "thickness", 0.0)),
                                      hole_center_mm is not None,
                                      extended=bool(extended_relational),
                                      n_rays=int(extended_relational or
                                                 N_BOUNDARY_RAYS))
    # Relational scalars, supplied as a SEPARATE input so PHYSICS_INPUT_DIM --
    # and therefore every existing checkpoint -- is untouched.
    geometry = np.concatenate([
        np.asarray([_finite_or_zero(manifest_row[name])
                    for name in GEOMETRY_COLS], dtype=np.float64),
        size_features,
        hole_norm,
    ])
    if geometry.size != GEOMETRY_DIM or bridge.size != BRIDGE_DIM:
        return None
    return {
        "sample_id": str(manifest_row["sample_id"]),
        "split": str(manifest_row["split"]),
        # The case-level artifact every bridge of one (shape, material) shares.
        # Identifies the case exactly, which split_mode="case" groups by.
        "case_key": str(response_path),
        "body_type": body_type,
        "body_one_hot": np.asarray(
            [body_type == "solid", body_type == "hollow"], dtype=np.float64),
        "contour_path": str(contour_path),
        # Content hash of the polygon.  Distinct from contour_path: solid and
        # hollow of one base shape are separate files with identical contents.
        "contour_digest": digest,
        "material": material,
        "bridge": bridge,
        # Consumed only by the spatial-query model; every other variant ignores
        # it, which is why it is not folded into ``bridge``.
        "bridge_extra": relational,
        "soundhole_center_mm": hole_center_mm,
        "soundhole_diameter_mm": hole_diameter,
        "soundhole_key": (None if hole_center_mm is None else
                          (round(hole_center_mm[0], 6),
                           round(hole_center_mm[1], 6), round(hole_diameter, 6))),
        "geometry": geometry,
        "log_magnitude_db": log_magnitude.astype(np.float32),
        "peak_frequency_hz": peak_frequency,
        "peak_amplitude_db": peak_amplitude,
        "peak_q": peak_q,
        "peak_mask": peak_mask,
        "peak_count_total": peak_count,
        "peak_truncated": peak_truncated,
        "freqs": frequencies,
    }


def _dataset_generator_version(dataset_dir: Path) -> str:
    plan_path = dataset_dir / "dataset_plan.json"
    if not plan_path.is_file():
        raise RuntimeError(f"{dataset_dir}: dataset_plan.json is missing")
    return str(json.loads(plan_path.read_text(encoding="utf-8"))
               .get("generator_version", ""))


def _validate_certificate(dataset_dir: Path, version: str) -> dict | None:
    """Certificate check that accepts any ``SUPPORTED_GENERATOR_VERSIONS``.

    For the version the generator currently produces we defer to
    ``dataset_gen_mixed.validate_dataset_ready`` — the full completion-policy
    audit.  An older published dataset was already certified by the generator
    that produced it and its plan is immutable, so here we re-verify what a
    CONSUMER can and must check: the certificate says complete, its plan hash
    matches the plan body, and manifest.csv still hashes to the certified value.
    """
    from dataset_gen_mixed import (GENERATOR_VERSION, DATASET_READY_FILENAME,
                                   _file_sha256, _stored_plan_hash,
                                   validate_dataset_ready)
    if version == GENERATOR_VERSION:
        return validate_dataset_ready(dataset_dir)

    plan = json.loads((dataset_dir / "dataset_plan.json").read_text(encoding="utf-8"))
    if plan.get("plan_hash") != _stored_plan_hash(plan):
        raise RuntimeError(f"{dataset_dir}: dataset plan body is corrupt")
    ready_path = dataset_dir / DATASET_READY_FILENAME
    manifest = dataset_dir / "manifest.csv"
    if not ready_path.is_file():
        raise RuntimeError(
            f"{dataset_dir}: not certified complete ({DATASET_READY_FILENAME} missing)")
    if not manifest.is_file():
        raise RuntimeError(f"{dataset_dir}: readiness certificate has no manifest.csv")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (ready.get("status") != "complete"
            or ready.get("generator_version") != version
            or ready.get("plan_hash") != plan.get("plan_hash")):
        raise RuntimeError(f"{dataset_dir}: readiness certificate does not match the plan")
    if ready.get("manifest_sha256") != _file_sha256(manifest):
        raise RuntimeError(f"{dataset_dir}: manifest.csv does not match its certificate")
    return ready


def _rows_from_case_status(dataset_dir: Path, plan: dict):
    """Manifest-equivalent rows for a dataset that was never certified.

    A dataset only gets ``dataset_ready.json`` once EVERY planned case is
    terminal, so a run that was stopped early (mixed-v5) or that deliberately
    defers cases (a mixed-v6 run policy) has no certificate and no manifest.csv —
    but the cases it did finish are committed and carry their manifest rows in
    ``case_status``.  This rebuilds those rows in memory; nothing is written, so
    it is safe to call while the generator is still running.

    Cheap integrity only: the plan body is re-hashed once and each case must
    carry the plan's hash.  The per-case ``validate_case_complete`` audit costs
    ~350 ms/case (35 min over v4+v5+v6) and is what merge does ONCE before
    sealing the certificate — re-running it at every training start is what the
    certificate exists to avoid.  Missing or truncated files are still caught,
    by the expected-sample-count check in the caller.
    """
    from dataset_gen_mixed import _load_case_status, _stored_plan_hash

    stored = json.loads((dataset_dir / "dataset_plan.json").read_text(encoding="utf-8"))
    plan_hash = stored.get("plan_hash")
    if plan_hash != _stored_plan_hash(stored):
        raise RuntimeError(f"{dataset_dir}: dataset plan body is corrupt")

    statuses = _load_case_status(dataset_dir / "case_status")
    rows = []
    n_done = 0
    for case in plan.get("cases", []):
        status = statuses.get(case["case_id"])
        if not status or status.get("status") != "done" or not status.get("rows"):
            continue
        if status.get("plan_hash") != plan_hash:
            raise RuntimeError(
                f"{dataset_dir}: case {case['case_id']} was produced under a "
                f"different plan ({status.get('plan_hash')})")
        n_done += 1
        rows.extend(status["rows"])
    if not rows:
        raise RuntimeError(f"{dataset_dir}: no completed cases to load")
    return rows, n_done


def _harmonize_manifest(frame, version: str, dataset_dir: Path):
    """Bring an older manifest onto the current geometry-column contract.

    mixed-v4 stored ``cavity_ratio``; v5 replaced it with the directly sampled
    ``top_plate_thickness_mm`` via ``cavity_ratio = 1 - top / thickness``.  That
    inversion is exact, so a v4 manifest is recovered rather than defaulted.
    """
    if "top_plate_thickness_mm" in frame.columns:
        return frame
    if "cavity_ratio" not in frame.columns or "thickness" not in frame.columns:
        raise RuntimeError(
            f"{dataset_dir}: {version} manifest has neither top_plate_thickness_mm "
            f"nor (cavity_ratio, thickness) to derive it from")
    frame = frame.copy()
    frame["top_plate_thickness_mm"] = np.where(
        frame["body_type"].astype(str) == "solid", 0.0,
        frame["thickness"].astype(float) * (1.0 - frame["cavity_ratio"].astype(float)))
    return frame


def _assert_splits_agree_across_datasets(plans: dict):
    """A geometry that appears in two datasets must carry the SAME split label.

    Two ways this bites in practice.  v5 and v6 were generated from the same
    seed, so all 100 contours are byte-identical between them (their splits agree,
    so they combine safely).  More subtly, the first ``N_CANONICAL`` shapes are
    the fixed guitar outlines and are the SAME in every dataset regardless of
    seed, while ``assign_splits`` is seeded — so combining two runs with different
    seeds can put a canonical body in train here and test there.

    ``shape_digests`` cannot be used for this — it folds in the bridge points,
    which v6 changed from 5 to 10 — so the contour itself is the key.
    """
    seen: dict[str, tuple[str, Path]] = {}
    for dataset_dir, plan in plans.items():
        splits = {int(k): v for k, v in plan.get("splits", {}).items()}
        for shape in plan.get("base_shapes", []):
            contour = np.round(np.asarray(shape["contour"], float), 6)
            key = hashlib.sha256(contour.tobytes()).hexdigest()
            split = splits.get(int(shape["base_shape_id"]))
            if key in seen and seen[key][0] != split:
                raise RuntimeError(
                    f"split leakage: base shape {shape['base_shape_id']} has the same "
                    f"contour in {seen[key][1]} (split={seen[key][0]}) and "
                    f"{dataset_dir} (split={split}); combining them would train on a "
                    f"held-out body")
            seen.setdefault(key, (split, dataset_dir))


SPLIT_MODES = ("shape", "case", "random")


def _reassign_splits(records: list[dict], mode: str, seed: int,
                     val_frac: float, test_frac: float):
    """Override the plan's shape-level split.  Diagnostic only — see the warning.

    ``shape``  (default) keeps the split written into ``dataset_plan.json``: a base
               geometry lives entirely in one split, so validation measures
               generalization to an UNSEEN BODY.  That is the deployment
               condition — the frontend's whole purpose is predicting a shape the
               network has never seen — so this is the only split whose number
               belongs in a paper.
    ``case``   groups by (shape, material): the 10 bridge points of one case stay
               together, but the same geometry may appear in train and val under a
               different material.  Geometry leaks; bridge position does not.
    ``random`` splits per sample.  Two bridge points of the SAME body with the
               SAME material land on opposite sides of the split, so validation is
               measuring interpolation between drive points on a body already
               memorised.  It answers a real question, but a much easier one, and
               the resulting number is not a generalization estimate.
    """
    if mode == "shape":
        return
    if mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {list(SPLIT_MODES)}")
    keys = ([r["sample_id"] for r in records] if mode == "random"
            else [r["case_key"] for r in records])
    unique = sorted(set(keys))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))
    n_val = int(round(float(val_frac) * len(unique)))
    n_test = int(round(float(test_frac) * len(unique)))
    label = {}
    for rank, index in enumerate(order):
        label[unique[int(index)]] = (
            "val" if rank < n_val
            else "test" if rank < n_val + n_test
            else "train")
    for record, key in zip(records, keys):
        record["split"] = label[key]
    print(f"[nn_dataset] WARNING split_mode={mode!r} OVERRIDES the plan's "
          f"shape-level split over {len(unique)} groups. Validation no longer "
          f"measures generalization to an unseen body; do not report this number "
          f"as one.")


SHAPE_KEY_MODES = ("path", "contour")


def _shape_key(record: dict, mode: str = "path") -> str:
    """Group key for "one shape".

    ``path``     the contour FILE.  mixed-v6 writes one file per (base shape,
                 body type), so this counts each geometry twice — but it is what
                 every earlier run used, so it stays the default and old numbers
                 remain reproducible.
    ``contour``  the contour CONTENTS.  Solid and hollow of one base shape hash
                 to the same key, so a count in this mode is a count of distinct
                 GEOMETRIES.  This is the right axis for a shape learning curve.

    Both modes are safe for splitting: the plan assigns a split per base shape,
    so the two files of a pair are never on opposite sides.
    """
    if mode == "contour":
        return record.get("contour_digest") or record["contour_path"]
    if mode != "path":
        raise ValueError(f"shape_key_mode must be one of {list(SHAPE_KEY_MODES)}")
    return record["contour_path"]


def _shape_vocabulary(records: list[dict], mode: str) -> dict:
    """Stable shape key -> contiguous index, sorted so a rerun reproduces it."""
    return {key: index for index, key in
            enumerate(sorted({_shape_key(r, mode) for r in records}))}


def shape_key_counts(records: list[dict]) -> dict:
    """Both shape counts, so a run's log can never mean only one of them."""
    return {
        "n_shape_files": len({_shape_key(r, "path") for r in records}),
        "n_contours": len({_shape_key(r, "contour") for r in records}),
    }


def _stratified_shape_order(records: list[dict], seed: int,
                            shape_key_mode: str = "path") -> list[str]:
    """Shape keys ordered so that any PREFIX is size-balanced.

    A plain random subset of 10 shapes out of 70 can land entirely in the small
    half of a dataset that spans a 12.7x plan-area range, which would confound
    "fewer shapes" with "smaller bodies".  Shapes are binned into area quartiles
    and drawn round-robin, so the first N of the returned list always covers the
    size range.
    """
    area_index = len(GEOMETRY_COLS) + SIZE_COLS.index("area_mm2")
    area = {}
    for record in records:
        area.setdefault(_shape_key(record, shape_key_mode),
                        float(record["geometry"][area_index]))
    keys = sorted(area)
    rng = np.random.default_rng(int(seed))
    ranked = sorted(keys, key=lambda k: area[k])
    quartiles = [ranked[i::4] for i in range(4)]
    for bucket in quartiles:
        rng.shuffle(bucket)
    order = []
    for position in range(max(len(b) for b in quartiles) if quartiles else 0):
        for bucket in quartiles:
            if position < len(bucket):
                order.append(bucket[position])
    return order


def select_train_subset(records: list[dict], *, max_shapes=None,
                        cases_per_shape=None, max_cases=None, seed: int = 0,
                        shape_key_mode: str = "path"):
    """Restrict the TRAINING records for a shape/case learning-curve sweep.

    Two independent axes, so the two experiments the sweep needs are expressible:
      * ``max_shapes`` with a fixed ``cases_per_shape`` -> more shapes at a
        proportionally larger FEM budget.
      * ``max_cases`` held constant while ``max_shapes`` varies -> the same FEM
        budget spent on few shapes x many materials versus many shapes x few
        materials.

    Solid and hollow are kept balanced within every shape, and shapes are taken
    in size-stratified order, so a smaller subset is not also a differently
    distributed one.  Returns the kept records; validation is never touched.
    """
    if max_shapes is None and cases_per_shape is None and max_cases is None:
        return records
    rng = np.random.default_rng(int(seed))
    order = _stratified_shape_order(records, seed, shape_key_mode)
    if max_shapes is not None:
        order = order[:int(max_shapes)]
    keep_shapes = set(order)

    by_shape: dict[str, dict[str, list[str]]] = {}
    for record in records:
        shape = _shape_key(record, shape_key_mode)
        if shape not in keep_shapes:
            continue
        by_shape.setdefault(shape, {}).setdefault(
            record["body_type"], []).append(record["case_key"])

    # Per shape, an ORDER in which cases should be taken: body types interleaved,
    # so that any prefix of the list is solid/hollow balanced.  Both the per-shape
    # cap and the global budget trim consume this same order — re-sorting it later
    # would silently reintroduce the imbalance it exists to prevent.
    ordered_cases: dict[str, list[str]] = {}
    for shape in order:
        buckets = by_shape.get(shape)
        if not buckets:
            continue
        pools = []
        for body_type in sorted(buckets):
            cases = sorted(set(buckets[body_type]))
            rng.shuffle(cases)
            pools.append(cases)
        interleaved = []
        for position in range(max(len(p) for p in pools)):
            for pool in pools:
                if position < len(pool):
                    interleaved.append(pool[position])
        if cases_per_shape is not None:
            interleaved = interleaved[:int(cases_per_shape)]
        ordered_cases[shape] = interleaved

    if max_cases is None:
        keep_cases = {c for cases in ordered_cases.values() for c in cases}
    else:
        # Round-robin across shapes so a constant-budget point keeps every
        # selected shape represented, and take from each shape's balanced order
        # so the budget is not spent on one body type.
        keep_list: list[str] = []
        position = 0
        budget = int(max_cases)
        while len(keep_list) < budget:
            progressed = False
            for shape in order:
                cases = ordered_cases.get(shape, ())
                if position < len(cases):
                    keep_list.append(cases[position])
                    progressed = True
                    if len(keep_list) >= budget:
                        break
            if not progressed:
                break
            position += 1
        keep_cases = set(keep_list)

    kept = [r for r in records
            if _shape_key(r, shape_key_mode) in keep_shapes
            and r["case_key"] in keep_cases]
    counts = shape_key_counts(kept)
    print(f"[nn_dataset] train subset (shape_key={shape_key_mode}): "
          f"{counts['n_shape_files']} shape files / {counts['n_contours']} distinct "
          f"contours, {len(keep_cases)} cases, {len(kept)} samples "
          f"(from {len(records)})")
    if not kept:
        raise RuntimeError("train subset selection removed every sample")
    return kept


def build_datasets(dataset_dir=None, qc_csv=None,
                   val_frac: float = 0.15, seed: int = 42,
                   grid_size: int = DEFAULT_GRID_SIZE,
                   include_test: bool = False,
                   require_certified: bool = False,
                   max_missing_fraction: float = 0.5,
                   split_mode: str = "shape",
                   max_train_shapes=None, cases_per_shape=None,
                   max_train_cases=None, shape_subset_seed: int = 0,
                   shape_key_mode: str = "path",
                   shape_channels: str = "occupancy",
                   shape_pairing: str = "correct",
                   target_norm: str = "global",
                   target_norm_floor: float = 0.1,
                   relational_set: str = "basic",
                   augment: str = "none",
                   event_heatmaps: str = "none",
                   heatmap_sigma_cents: float = 50.0,
                   heatmap_prominence_db: float = 1.5):
    """Load one or more certified mixed datasets and honor their shape splits.

    ``dataset_dir`` takes a single path or a sequence of paths, so v4/v5/v6 runs
    can be trained on together; ``qc_csv`` likewise takes one path applied to
    every dataset, or one per dataset.  There is no default directory — the
    caller always says where its data lives.

    Combining is guarded: every dataset must share the frequency grid and
    peak-slot count, and no geometry may appear under two different split labels.
    Sample IDs are namespaced by dataset so identical IDs from different runs
    cannot collide or be silently de-duplicated.

    A dataset does NOT have to be finished.  When ``dataset_ready.json`` is
    present it is used (the strongest guarantee: the certificate seals the
    manifest hash and the planned sample count); otherwise the finished cases are
    read straight from ``case_status``, which is how a run that was stopped early
    or that defers cases is trained on.  Either way the load is announced, so a
    partial set is never silent.

    A sample whose artifacts are missing is skipped and reported, not fatal —
    for training it is no different from a case that was never run.
    ``max_missing_fraction`` (default 0.5) still stops a half-copied or unmounted
    dataset from loading as if it were fine; it is a catastrophe guard, not a
    quality gate.

    Integrity that always holds: plan-body hash, per-case plan hash, one shared
    frequency grid, all-finite arrays.  Deliberately NOT the ~350 ms/case artifact
    re-audit that merge runs once before sealing the certificate — repeating it on
    every training start would cost ~35 min over v4+v5+v6 and re-derives what the
    certificate already attests.

    ``require_certified=True`` refuses an unfinished dataset.

    Returns ``(train, val, stats)``, or ``(train, val, test, stats)`` when
    ``include_test`` is set.  The test split is built with the SAME
    train-fitted ``NormStats`` — normalization is never refitted on held-out
    data — and is off by default so training runs cannot touch it by accident.
    """
    # The plan's split is immutable; ``val_frac``/``seed`` are consulted only when
    # ``split_mode`` asks for a different, diagnostic grouping (see
    # ``_reassign_splits``).
    import pandas as pd

    if split_mode not in SPLIT_MODES:
        raise ValueError(
            f"split_mode must be one of {list(SPLIT_MODES)}, got {split_mode!r}")

    if dataset_dir is None:
        raise ValueError("dataset_dir is required: pass one or more dataset paths")
    dirs = ([Path(dataset_dir)] if isinstance(dataset_dir, (str, Path))
            else [Path(d) for d in dataset_dir])
    if not dirs:
        raise ValueError("dataset_dir is required: pass one or more dataset paths")
    if len({d.resolve() for d in dirs}) != len(dirs):
        raise ValueError("the same dataset directory was passed more than once")

    if qc_csv is None or isinstance(qc_csv, (str, Path)):
        qc_list = [qc_csv] * len(dirs)
    else:
        qc_list = list(qc_csv)
        if not qc_list:                       # `--qc-csv` with no values = off
            qc_list = [None] * len(dirs)
        elif len(qc_list) == 1:
            qc_list = qc_list * len(dirs)
        elif len(qc_list) != len(dirs):
            raise ValueError(
                f"qc_csv must be a single path or one per dataset "
                f"({len(qc_list)} given for {len(dirs)} datasets)")

    plans = {}
    for d in dirs:
        version = _dataset_generator_version(d)
        if version not in SUPPORTED_GENERATOR_VERSIONS:
            raise RuntimeError(
                f"{d}: generator version {version!r} is not supported "
                f"(expected one of {list(SUPPORTED_GENERATOR_VERSIONS)})")
        plans[d] = json.loads((d / "dataset_plan.json").read_text(encoding="utf-8"))
    _assert_splits_agree_across_datasets(plans)

    records = []
    for d, qc_for_dir in zip(dirs, qc_list):
        version = str(plans[d].get("generator_version", ""))
        certified = (d / "dataset_ready.json").is_file() and (d / "manifest.csv").is_file()
        if certified:
            ready = _validate_certificate(d, version)
            frame = pd.read_csv(d / "manifest.csv")
            expected_samples = None if ready is None else int(ready["n_samples"])
            n_planned = None if ready is None else ready.get("n_planned_cases")
            print(f"[nn_dataset] {d.name}: certified complete — "
                  f"{expected_samples} samples"
                  + (f" over {n_planned} planned cases" if n_planned else ""))
        elif require_certified:
            raise RuntimeError(
                f"{d}: not certified complete (dataset_ready.json / manifest.csv "
                f"missing).  A dataset is certified only once every planned case is "
                f"terminal, so a run that was stopped early or that defers cases has "
                f"none.  Drop require_certified to train on the finished subset.")
        else:
            rows, n_done = _rows_from_case_status(d, plans[d])
            frame = pd.DataFrame(rows)
            expected_samples = len(rows)
            n_planned = len(plans[d].get("cases", []))
            print(f"[nn_dataset] {d.name}: partial ({version}) — "
                  f"{n_done}/{n_planned} cases done, {expected_samples} samples")
            ready = None
        manifest = _harmonize_manifest(frame, version, d)
        done = manifest[manifest["status"] == "done"].copy()

        loaded = []
        unreadable = []
        contour_cache: dict = {}
        for _, row in done.iterrows():
            record = _load_sample_record(
                d, row, contour_cache,
                extended_relational=RELATIONAL_RAY_COUNTS.get(
                    relational_set, 0))
            if record is not None:
                loaded.append(record)
            else:
                unreadable.append(str(row.get("sample_id", "?")))
        if not loaded:
            raise RuntimeError(f"{d}: no valid samples found")
        # A sample whose artifacts are gone is, for training purposes, simply a
        # sample that is not there — the same as a case that was never run — so it
        # is skipped rather than fatal.  But it is reported: unlike a deferred
        # case, a missing file means something went wrong AFTER generation
        # (typically a truncated copy), and silently training on less data than
        # you think you have is how results stop being reproducible.
        if unreadable:
            shown = ", ".join(unreadable[:5])
            more = f" (+{len(unreadable) - 5} more)" if len(unreadable) > 5 else ""
            print(f"[nn_dataset] {d.name}: WARNING skipped {len(unreadable)} of "
                  f"{len(unreadable) + len(loaded)} samples — artifacts missing or "
                  f"unreadable: {shown}{more}")
        total = len(loaded) + len(unreadable)
        if total and len(unreadable) / total > float(max_missing_fraction):
            raise RuntimeError(
                f"{d}: {len(unreadable)}/{total} samples could not be read "
                f"({len(unreadable) / total:.0%} > "
                f"{float(max_missing_fraction):.0%}).  This is a broken or "
                f"half-copied dataset, not a few lost files; raise "
                f"max_missing_fraction to load it anyway.")
        if expected_samples is not None and total != expected_samples:
            # The manifest/certificate and the rows disagree about how many
            # samples should even be attempted — a different failure from files
            # being unreadable, and never expected.
            raise RuntimeError(
                f"{d}: manifest lists {total} done samples but the "
                f"{'certificate' if ready is not None else 'case status'} says "
                f"{expected_samples}")

        if qc_for_dir:
            qc_path = Path(qc_for_dir)
            if not qc_path.is_file() and not qc_path.is_absolute():
                qc_path = d / qc_path
            if qc_path.is_file():
                qc = pd.read_csv(qc_path)
                bad_ids = set(qc.loc[qc["flagged"] == True, "sample_id"].astype(str))
                loaded = [r for r in loaded if r["sample_id"] not in bad_ids]

        # Namespace the IDs: v4/v5/v6 case IDs can repeat across runs, and a
        # collision here would silently merge two different physical samples.
        # QC filtering above matches the RAW ids, which is what a per-dataset QC
        # CSV contains; keep them so a sample can still be traced back to its
        # source manifest after namespacing.
        tag = d.resolve().name
        for record in loaded:
            record["dataset"] = tag
            record["source_sample_id"] = record["sample_id"]
            record["sample_id"] = f"{tag}/{record['sample_id']}"
        records.extend(loaded)

    if not records:
        raise RuntimeError("No samples remain after QC filtering")
    if len({r["sample_id"] for r in records}) != len(records):
        raise RuntimeError("duplicate sample IDs across the given datasets")

    reference_frequencies = np.asarray(records[0]["freqs"], float)
    reference_peak_shape = records[0]["peak_frequency_hz"].shape
    for record in records:
        arrays = [record["material"], record["bridge"], record["geometry"],
                  record["log_magnitude_db"], record["peak_frequency_hz"],
                  record["peak_amplitude_db"], record["peak_q"]]
        if (not np.array_equal(record["freqs"], reference_frequencies)
                or record["peak_frequency_hz"].shape != reference_peak_shape
                or not all(np.all(np.isfinite(value)) for value in arrays)):
            raise RuntimeError(f"invalid or inconsistent sample: {record['sample_id']}")

    _reassign_splits(records, split_mode, seed, val_frac, test_frac=val_frac)
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    test_records = [record for record in records if record["split"] == "test"]
    if not train_records or not val_records:
        raise RuntimeError(
            "training selection must contain non-empty train and val splits")
    # Subsetting happens BEFORE the statistics are fitted: a learning-curve point
    # must be normalized by its own training data, or the smaller runs quietly
    # borrow information from the shapes they are supposed to be missing.
    train_records = select_train_subset(
        train_records, max_shapes=max_train_shapes,
        cases_per_shape=cases_per_shape, max_cases=max_train_cases,
        seed=shape_subset_seed, shape_key_mode=shape_key_mode)
    # Reflections are added to TRAIN only, and before the statistics are fitted
    # so the normalisation describes the data the model actually sees.  Adding
    # them to validation would inflate it with copies of bodies already in it.
    if augment == "mirror":
        before = len(train_records)
        train_records = augment_records(train_records, MIRROR_TRANSFORMS)
        print(f"[nn_dataset] mirror augmentation: {before} -> "
              f"{len(train_records)} train samples "
              f"({len({r['contour_digest'] for r in train_records})} distinct "
              f"reflected contours). Exact symmetry: the material axes are "
              f"coordinate-aligned and orthotropic, so a reflected body has the "
              f"identical response.")
    elif augment != "none":
        raise ValueError("augment must be 'none' or 'mirror'")
    stats = NormStats()
    stats.fit(train_records)
    # Fitted on the TRAIN records only, exactly like every other statistic here.
    stats.fit_target_mode(train_records, target_norm, target_norm_floor)
    # Shape indices are assigned from the TRAIN split alone.  A val shape has no
    # index by construction — that is the point of the diagnostic, not a gap —
    # and is mapped to the reserved unknown slot by AdmittanceDataset.
    shape_vocab = _shape_vocabulary(train_records, shape_key_mode)
    dataset_kw = dict(shape_vocab=shape_vocab, shape_key_mode=shape_key_mode,
                      shape_channels=shape_channels,
                      shape_pairing=shape_pairing, pairing_seed=seed)
    train_ds = AdmittanceDataset(train_records, stats, grid_size, **dataset_kw)
    val_ds = AdmittanceDataset(val_records, stats, grid_size, **dataset_kw)
    if event_heatmaps != "none":
        for dataset in (train_ds, val_ds):
            attach_event_heatmaps(dataset.samples, kinds=event_heatmaps,
                                  sigma_cents=heatmap_sigma_cents,
                                  prominence_db=heatmap_prominence_db)
    if shape_pairing == "constant":
        # The TRAIN mean, handed to both splits.
        mean_image = np.mean(np.stack(
            [train_ds._shape_grid(p) for p in
             sorted({s["contour_path"] for s in train_records})]), axis=0)
        for dataset in (train_ds, val_ds):
            dataset.set_constant_image(mean_image)
    if not include_test:
        return train_ds, val_ds, stats
    if not test_records:
        raise RuntimeError("include_test=True but the dataset has no test split")
    return (train_ds, val_ds,
            AdmittanceDataset(test_records, stats, grid_size, **dataset_kw),
            stats)
