"""shape_gen.py - Randomised 2-D body contour generator."""

from __future__ import annotations
import numpy as np
from scipy.interpolate import splprep, splev


def random_shape(seed=None, n_points: int = 120) -> np.ndarray:
    """Generate a random closed contour in mm units, centred at origin.

    Parameters
    ----------
    seed : int or None — RNG seed for reproducibility
    n_points : number of evenly-spaced points in the output contour

    Returns
    -------
    contour : (n_points, 2) float64, mm units, centred at origin

    Shape variety by corner_radius (sampled uniformly in [0, 0.7]):
      0.00–0.15  → angular / polygonal  (linear spline)
      0.15–0.35  → mixed               (cubic interpolating, no rounding)
      0.35–0.70  → smooth / rounded    (cubic smoothing spline)
    """
    rng = np.random.default_rng(seed)

    n_vertices    = int(rng.integers(5, 16))        # 5-15 control vertices
    irregularity  = float(rng.uniform(0.0, 0.6))    # 0=uniform radii, 0.6=very uneven
    corner_radius = float(rng.uniform(0.0, 0.7))    # 0=angular, 0.7=very round
    aspect_ratio  = float(rng.uniform(0.5, 1.5))    # x/y stretch
    scale_mm      = float(rng.uniform(250, 450))    # target bounding-box height (mm)

    # 1. Evenly-spaced base angles + bounded jitter
    #    (pure uniform sort can cluster all angles → degenerate polygon)
    base_angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    jitter_max  = np.pi / n_vertices * 0.7
    angles      = base_angles + rng.uniform(-jitter_max, jitter_max, n_vertices)

    # 2. Variable radii (r_lo keeps vertices from collapsing to near-zero)
    r_lo  = max(1.0 - irregularity, 0.35)
    radii = rng.uniform(r_lo, 1.0, n_vertices) * scale_mm / 2
    pts   = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # 3. Pull a subset of vertices inward for non-convex indentations
    #    Pull factor limited to [0.55, 0.80] to prevent near-zero widths
    if irregularity > 0.3:
        n_pull = max(1, n_vertices // 3)
        for i in rng.choice(n_vertices, size=n_pull, replace=False):
            pts[i] *= rng.uniform(0.55, 0.80)

    # 4. Apply aspect ratio (x-axis stretch)
    pts[:, 0] *= aspect_ratio

    # 5. Spline smoothing — style varies with corner_radius
    contour = _smooth_polygon(pts, smoothness=corner_radius, n_points=n_points)

    # 6. Normalise: height → scale_mm, centre at origin
    contour = _normalize_contour(contour, scale_mm)

    # 7. Clamp width/height ratio to [0.40, 2.00]
    contour = _clamp_aspect(contour, min_ratio=0.40, max_ratio=2.00)

    # 8. Degenerate check: if area < 8% of bounding box, fall back to ellipse
    w = contour[:, 0].max() - contour[:, 0].min()
    h = contour[:, 1].max() - contour[:, 1].min()
    if _polygon_area(contour) / (w * h + 1e-9) < 0.08:
        t  = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        rx = scale_mm * min(aspect_ratio, 2.0) * 0.40
        ry = scale_mm * 0.40
        contour = np.column_stack([rx * np.cos(t), ry * np.sin(t)])
        contour = _normalize_contour(contour, scale_mm)
        contour = _clamp_aspect(contour, 0.40, 2.00)

    return contour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth_polygon(pts: np.ndarray, smoothness: float,
                    n_points: int = 120) -> np.ndarray:
    """Fit a periodic spline through pts with style controlled by smoothness.

    smoothness < 0.15  → linear (k=1), sharp corners
    smoothness < 0.35  → cubic interpolating (k=3, s=0), rounded but exact
    smoothness >= 0.35 → cubic smoothing (k=3, s>0), ellipse-like
    """
    closed = np.vstack([pts, pts[[0]]])

    if smoothness < 0.15:
        k, s = 1, 0                             # linear — angular
    elif smoothness < 0.35:
        k, s = 3, 0                             # cubic interpolating — mixed
    else:
        k = 3
        diffs     = np.diff(closed, axis=0)
        mean_dist = np.mean(np.hypot(diffs[:, 0], diffs[:, 1]))
        factor    = (smoothness - 0.35) / 0.65  # 0→1 as smoothness 0.35→0.70
        s         = factor * len(closed) * (mean_dist ** 2) * 3.0

    tck, _ = splprep([closed[:, 0], closed[:, 1]], s=s, per=True, k=k)
    u_new  = np.linspace(0, 1, n_points, endpoint=False)
    x_sm, y_sm = splev(u_new, tck)
    return np.column_stack([x_sm, y_sm])


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for signed polygon area (returns absolute value)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clamp_aspect(contour: np.ndarray, min_ratio: float, max_ratio: float) -> np.ndarray:
    """Clamp the width/height ratio by scaling x independently."""
    h = contour[:, 1].max() - contour[:, 1].min()
    w = contour[:, 0].max() - contour[:, 0].min()
    if h < 1e-9:
        return contour
    ratio = w / h
    if ratio < min_ratio:
        contour = contour.copy()
        contour[:, 0] *= min_ratio / ratio
    elif ratio > max_ratio:
        contour = contour.copy()
        contour[:, 0] *= max_ratio / ratio
    return contour


def _normalize_contour(contour: np.ndarray, target_height_mm: float) -> np.ndarray:
    """Scale so bounding-box height == target_height_mm and centre at origin."""
    h = contour[:, 1].max() - contour[:, 1].min()
    if h < 1e-9:
        return contour
    c = contour * (target_height_mm / h)
    c -= c.mean(axis=0)
    return c


# ---------------------------------------------------------------------------
# CLI: visualise 10 random shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 10
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        c = random_shape(seed=i)
        closed = np.vstack([c, c[[0]]])
        ax.plot(closed[:, 0], closed[:, 1], "b-", linewidth=1.5)
        ax.fill(c[:, 0], c[:, 1], alpha=0.12)
        h = c[:, 1].max() - c[:, 1].min()
        w = c[:, 0].max() - c[:, 0].min()
        ax.set_title(f"seed={i}  {w:.0f}x{h:.0f}mm", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("random_shape() — 10 samples", fontsize=12, y=1.01)
    plt.tight_layout()
    out = "shape_gen_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()
