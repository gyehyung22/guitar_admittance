"""placement_utils.py - Bridge point and soundhole placement inside a guitar contour."""

from __future__ import annotations
import numpy as np
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def random_bridge_points(
    contour_mm: np.ndarray,
    n: int = 3,
    min_edge_dist: float = 25.0,
    rng=None,
    max_tries: int = 8000,
) -> np.ndarray:
    """Sample n random bridge points strictly inside the body contour.

    Each point is at least min_edge_dist mm from the contour boundary.

    Parameters
    ----------
    contour_mm : (N, 2) body contour in mm, centred at origin
    n : number of bridge points to return
    min_edge_dist : minimum distance from the contour edge (mm)
    rng : np.random.Generator or int seed or None
    max_tries : size of the candidate pool (oversample factor)

    Returns
    -------
    pts : (n, 2) float64, mm coordinates
    """
    rng = _as_rng(rng)
    path = MplPath(contour_mm)

    x_lo, y_lo = contour_mm.min(axis=0)
    x_hi, y_hi = contour_mm.max(axis=0)

    # Oversample uniform candidates in bounding box
    cands = rng.uniform([x_lo, y_lo], [x_hi, y_hi], (max_tries, 2))

    # Keep only those inside the contour
    cands = cands[path.contains_points(cands)]
    if len(cands) == 0:
        raise RuntimeError("No candidates found inside contour — check contour validity")

    # Filter by minimum distance to the boundary
    edge_dists = _batch_dist_to_polygon(cands, contour_mm)
    valid = cands[edge_dists >= min_edge_dist]

    if len(valid) < n:
        raise RuntimeError(
            f"Only {len(valid)} valid locations found (need {n}). "
            f"Try reducing min_edge_dist (currently {min_edge_dist} mm)."
        )

    idx = rng.choice(len(valid), size=n, replace=False)
    return valid[idx]


def random_soundhole(
    contour_mm: np.ndarray,
    bridge_pts: np.ndarray,
    diameter_mm: float,
    min_bridge_dist: float = 50.0,
    min_edge_dist: float = 30.0,
    rng=None,
    max_tries: int = 8000,
) -> np.ndarray | None:
    """Find a random centre position for a round soundhole inside the contour.

    Constraints:
      - The hole rim (centre ± radius) is at least min_edge_dist mm inside the
        contour boundary.
      - The centre is at least min_bridge_dist mm from every bridge point.

    Parameters
    ----------
    contour_mm : (N, 2) body contour in mm
    bridge_pts : (B, 2) bridge point coordinates in mm
    diameter_mm : soundhole diameter (mm)
    min_bridge_dist : minimum distance from any bridge point (mm)
    min_edge_dist : clearance between hole rim and body edge (mm)
    rng : np.random.Generator or int seed or None
    max_tries : candidate pool size

    Returns
    -------
    centre : (2,) mm coordinates, or None if no valid location exists
    """
    rng = _as_rng(rng)
    path = MplPath(contour_mm)

    x_lo, y_lo = contour_mm.min(axis=0)
    x_hi, y_hi = contour_mm.max(axis=0)

    cands = rng.uniform([x_lo, y_lo], [x_hi, y_hi], (max_tries, 2))
    cands = cands[path.contains_points(cands)]
    if len(cands) == 0:
        return None

    # Hole rim must be at least min_edge_dist from the contour boundary
    radius = diameter_mm / 2.0
    edge_dists = _batch_dist_to_polygon(cands, contour_mm)
    cands = cands[edge_dists >= radius + min_edge_dist]
    if len(cands) == 0:
        return None

    # Each candidate must be far enough from every bridge point
    bridge_pts = np.atleast_2d(bridge_pts)
    for bp in bridge_pts:
        d = np.hypot(cands[:, 0] - bp[0], cands[:, 1] - bp[1])
        cands = cands[d >= min_bridge_dist]
        if len(cands) == 0:
            return None

    return cands[rng.integers(len(cands))]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _batch_dist_to_polygon(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Vectorised minimum distance from each point to the polygon boundary.

    Parameters
    ----------
    points  : (N, 2)
    contour : (M, 2)

    Returns
    -------
    dists : (N,) min distance to the nearest boundary segment
    """
    x0 = contour[:, 0]          # (M,)
    y0 = contour[:, 1]
    x1 = np.roll(x0, -1)        # (M,) — next vertex (wraps)
    y1 = np.roll(y0, -1)
    dx = x1 - x0
    dy = y1 - y0
    len_sq = dx * dx + dy * dy  # (M,)

    px = points[:, 0:1]         # (N, 1)
    py = points[:, 1:2]

    # Projection parameter t ∈ [0, 1] onto each segment
    t = np.clip(
        ((px - x0) * dx + (py - y0) * dy) / (len_sq + 1e-12),
        0.0, 1.0,
    )  # (N, M)

    proj_x = x0 + t * dx        # (N, M) nearest point on segment
    proj_y = y0 + t * dy

    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)  # (N, M)
    return dist.min(axis=1)     # (N,)


# ---------------------------------------------------------------------------
# CLI: visualise placements on an example shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from guitar_shapes import get_guitar_contour, GUITAR_SHAPES

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    seed = 42

    for ax, name in zip(axes.flat, GUITAR_SHAPES):
        contour = get_guitar_contour(name)
        closed = np.vstack([contour, contour[[0]]])
        ax.plot(closed[:, 0], closed[:, 1], "b-", linewidth=1.2)
        ax.fill(contour[:, 0], contour[:, 1], alpha=0.08)

        # 2 bridge points
        try:
            bpts = random_bridge_points(contour, n=2, min_edge_dist=20,
                                        rng=seed)
            ax.scatter(bpts[:, 0], bpts[:, 1], c="red", s=40, zorder=5,
                       label="bridge")
        except RuntimeError as e:
            bpts = np.zeros((0, 2))
            print(f"  {name}: bridge placement failed — {e}")

        # Soundhole (only for hollow-body shapes)
        if name in ("acoustic_dread", "acoustic_oo", "telecaster"):
            diameter = 80.0 if "acoustic" in name else 50.0
            centre = random_soundhole(contour, bpts, diameter_mm=diameter,
                                      min_bridge_dist=40, min_edge_dist=20,
                                      rng=seed)
            if centre is not None:
                circle = Circle(centre, diameter / 2, fill=False,
                                edgecolor="green", linewidth=1.5, zorder=5)
                ax.add_patch(circle)
                ax.scatter(*centre, c="green", s=25, zorder=6)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=9)

    plt.suptitle("Bridge points (red) & soundhole (green)", fontsize=12)
    plt.tight_layout()
    out = "placement_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
