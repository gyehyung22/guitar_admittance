"""guitar_shapes.py - Hardcoded keypoints for 8 standard guitar body outlines."""

from __future__ import annotations
import numpy as np
from scipy.interpolate import splprep, splev

# Convention
#   x+ = treble side (right when facing the guitar)
#   y+ = toward neck  (up in front-view photos)
#   Body centred at origin.
#
# Keypoints are listed clockwise from the neck-pocket centre
# (top → treble side → bottom → bass side → top).
#
# "k" sets the B-spline degree:
#   k=3  cubic   → smooth / rounded (Strat, LP, acoustic, …)
#   k=1  linear  → angular / straight-edged (Flying V, Explorer)

_MODELS: dict[str, dict] = {

    # ------------------------------------------------------------------
    # Fender Stratocaster  ~324 mm wide × 343 mm tall
    # Asymmetric double cutaway:
    #   bass side (x–) has a longer upper horn
    #   treble side (x+) has a shallower cutaway
    # ------------------------------------------------------------------
    "stratocaster": {
        "k": 3,
        "pts": [
            (   0,  171),  # neck pocket centre
            (  20,  162),  # treble neck pocket edge
            (  55,  140),  # treble cutaway horn tip
            ( 108,   90),  # treble upper bout outer
            ( 110,   52),  # treble waist upper
            (  84,   18),  # treble waist
            (  97,  -18),  # lower treble entry
            ( 160,  -55),  # lower treble bout max
            ( 130, -125),  # lower treble lower
            (  68, -171),  # bottom treble corner
            (   0, -171),  # bottom centre
            ( -68, -171),  # bottom bass corner
            (-130, -125),  # lower bass lower
            (-160,  -55),  # lower bass bout max
            ( -97,  -18),  # lower bass entry
            ( -84,   18),  # bass waist
            (-115,   52),  # bass waist upper
            (-138,   88),  # bass upper bout outer
            (-128,  140),  # bass horn outer shoulder
            ( -95,  158),  # bass horn tip  (extends further than treble)
            ( -18,  168),  # bass neck pocket edge
        ],
    },

    # ------------------------------------------------------------------
    # Fender Telecaster  ~324 mm wide × 356 mm tall
    # Single cutaway on treble side, semi-rectangular lower bout,
    # gentle rounded bass shoulder
    # ------------------------------------------------------------------
    "telecaster": {
        "k": 3,
        "pts": [
            (   0,  178),  # neck pocket centre
            (  22,  172),  # treble neck pocket edge
            (  58,  158),  # treble cutaway
            ( 142,   90),  # treble upper bout outer
            ( 148,   20),  # treble mid outer
            ( 150,  -60),  # lower treble bout outer
            ( 138, -140),  # lower treble lower
            (  85, -178),  # bottom treble corner
            (   0, -178),  # bottom centre
            ( -85, -178),  # bottom bass corner
            (-138, -140),  # lower bass lower
            (-150,  -60),  # lower bass bout outer
            (-148,   20),  # bass mid outer
            (-148,   80),  # bass upper bout outer (no cutaway, rectangular)
            (-130,  145),  # bass upper shoulder
            ( -20,  175),  # bass neck pocket edge
        ],
    },

    # ------------------------------------------------------------------
    # Gibson Les Paul  ~330 mm wide × 352 mm tall
    # Single cutaway on treble side, full rounded bass shoulder
    # ------------------------------------------------------------------
    "les_paul": {
        "k": 3,
        "pts": [
            (   0,  176),  # neck pocket centre
            (  22,  170),  # treble neck pocket edge
            (  65,  148),  # treble cutaway
            ( 118,   50),  # treble upper bout
            ( 120,   -5),  # treble waist
            ( 130,  -70),  # lower treble entry
            ( 162,  -95),  # lower treble bout max
            ( 145, -140),  # lower treble lower
            (  88, -176),  # bottom treble corner
            (   0, -176),  # bottom centre
            ( -88, -176),  # bottom bass corner
            (-145, -140),  # lower bass lower
            (-162,  -95),  # lower bass bout max
            (-130,  -70),  # lower bass entry
            (-120,   -5),  # bass waist
            (-140,   40),  # bass waist upper
            (-145,   90),  # bass upper bout outer
            (-138,  135),  # bass upper shoulder  (full, no cutaway)
            ( -90,  162),  # bass upper shoulder top
            ( -20,  174),  # bass neck pocket edge
        ],
    },

    # ------------------------------------------------------------------
    # Gibson SG  ~310 mm wide × 360 mm tall
    # Double cutaway with "devil's horns" projecting above neck level
    # ------------------------------------------------------------------
    "sg": {
        "k": 3,
        "pts": [
            (   0,  171),  # neck pocket centre
            (  18,  170),  # treble neck pocket edge
            (  72,  168),  # treble horn base
            ( 105,  172),  # treble horn tip (above neck level)
            ( 130,  152),  # treble horn outer
            ( 132,  112),  # treble upper bout outer
            ( 118,   72),  # treble waist upper
            (  88,   35),  # treble waist
            ( 100,  -12),  # lower treble entry
            ( 148,  -55),  # lower treble bout max
            ( 132, -120),  # lower treble lower
            (  75, -171),  # bottom treble corner
            (   0, -171),  # bottom centre
            ( -75, -171),  # bottom bass corner
            (-132, -120),  # lower bass lower
            (-148,  -55),  # lower bass bout max
            (-100,  -12),  # lower bass entry
            ( -88,   35),  # bass waist
            (-118,   72),  # bass waist upper
            (-132,  112),  # bass upper bout outer
            (-110,  172),  # bass horn tip (slightly longer than treble)
            ( -72,  168),  # bass horn base
            ( -18,  170),  # bass neck pocket edge
        ],
    },

    # ------------------------------------------------------------------
    # Gibson Flying V  ~390 mm wide × 400 mm tall
    # V-shaped angular body — uses k=1 (linear) to preserve straight edges.
    # Dense keypoints along each wing segment force straight-line character.
    # ------------------------------------------------------------------
    "flying_v": {
        "k": 1,
        "pts": [
            # Neck area
            (   0,  200),  # neck pocket centre
            (  18,  195),  # treble neck pocket edge
            # Treble wing: straight upper arm neck→tip
            (  62,  175),
            ( 118,  118),
            ( 172,   42),
            ( 195,  -90),  # treble wing tip
            # Treble wing: bottom edge tip→inner
            ( 175, -148),
            ( 148, -178),  # treble wing bottom outer
            (  98, -178),  # treble wing flat bottom
            (  58, -178),  # treble wing inner bottom
            # V notch
            (  25, -135),  # V inner upper treble
            (   0, -108),  # V bottom point
            ( -25, -135),  # V inner upper bass
            # Bass wing: inner→tip
            ( -58, -178),  # bass wing inner bottom
            ( -98, -178),  # bass wing flat bottom
            (-148, -178),  # bass wing bottom outer
            (-175, -148),
            (-195,  -90),  # bass wing tip
            # Bass wing: straight upper arm tip→neck
            (-172,   42),
            (-118,  118),
            ( -62,  175),
            ( -18,  195),  # bass neck pocket edge
        ],
    },

    # ------------------------------------------------------------------
    # Gibson Explorer  ~400 mm wide × 432 mm tall
    # Angular asymmetric body — uses k=1 (linear) for angular edges.
    # Large bass wing (extends far left at mid-body);
    # treble side has a pointed upper-right projection.
    # ------------------------------------------------------------------
    "explorer": {
        "k": 1,
        "pts": [
            # Neck + treble upper arm
            (   0,  216),  # neck pocket centre
            (  20,  210),  # treble neck pocket edge
            (  88,  185),  # treble upper shoulder
            ( 165,  128),  # treble upper angular corner
            # Treble side: straight downward
            ( 178,   65),
            ( 178,  -10),
            ( 155,  -90),
            ( 118, -178),  # lower treble body
            # Angular bottom
            (  62, -216),  # bottom treble corner
            (   0, -216),  # bottom centre
            ( -60, -216),  # bottom bass corner
            # Bass lower body
            (-128, -178),
            (-178,  -58),  # bass outer lower
            # Bass wing: large horizontal extension
            (-178,   12),  # bass wing outer max
            (-178,   75),  # bass wing upper (straight left edge)
            # Transition from wing back to neck
            ( -98,  152),
            ( -20,  212),  # bass neck pocket edge
        ],
    },

    # ------------------------------------------------------------------
    # Acoustic Dreadnought  ~394 mm wide × 508 mm tall
    # Classic large hourglass with pronounced waist
    # ------------------------------------------------------------------
    "acoustic_dread": {
        "k": 3,
        "pts": [
            (   0,  254),  # neck joint centre
            (  20,  252),  # treble neck joint edge
            ( 128,  222),  # treble upper bout outer
            ( 146,  155),  # treble upper bout lower outer
            ( 132,   82),  # treble waist upper
            ( 100,   15),  # treble waist (narrow)
            ( 112,  -48),  # lower treble entry
            ( 192, -115),  # lower treble bout max
            ( 178, -188),  # lower treble lower
            ( 100, -254),  # bottom treble corner
            (   0, -254),  # bottom centre
            (-100, -254),  # bottom bass corner
            (-178, -188),  # lower bass lower
            (-192, -115),  # lower bass bout max
            (-112,  -48),  # lower bass entry
            (-100,   15),  # bass waist (narrow)
            (-132,   82),  # bass waist upper
            (-146,  155),  # bass upper bout lower outer
            (-128,  222),  # bass upper bout outer
            ( -20,  252),  # bass neck joint edge
        ],
    },

    # ------------------------------------------------------------------
    # Acoustic OO (double-oh)  ~356 mm wide × 483 mm tall
    # Smaller hourglass; more even upper/lower bout proportions
    # ------------------------------------------------------------------
    "acoustic_oo": {
        "k": 3,
        "pts": [
            (   0,  241),  # neck joint centre
            (  20,  239),  # treble neck joint edge
            ( 115,  212),  # treble upper bout outer
            ( 136,  148),  # treble upper bout lower outer
            ( 122,   78),  # treble waist upper
            (  92,   12),  # treble waist (narrow)
            ( 105,  -42),  # lower treble entry
            ( 175, -108),  # lower treble bout max
            ( 162, -178),  # lower treble lower
            (  92, -241),  # bottom treble corner
            (   0, -241),  # bottom centre
            ( -92, -241),  # bottom bass corner
            (-162, -178),  # lower bass lower
            (-175, -108),  # lower bass bout max
            (-105,  -42),  # lower bass entry
            ( -92,   12),  # bass waist (narrow)
            (-122,   78),  # bass waist upper
            (-136,  148),  # bass upper bout lower outer
            (-115,  212),  # bass upper bout outer
            ( -20,  239),  # bass neck joint edge
        ],
    },
}

GUITAR_SHAPES: list[str] = list(_MODELS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_guitar_contour(model_name: str, scale: float = 1.0,
                       n_points: int = 120) -> np.ndarray:
    """Return a smooth closed contour for the named guitar body model.

    Parameters
    ----------
    model_name : str — one of GUITAR_SHAPES
    scale : float — uniform scale factor (1.0 = real-world mm)
    n_points : int — number of evenly-spaced points on the output contour

    Returns
    -------
    contour : (n_points, 2) float64, mm units, centred at origin
    """
    if model_name not in _MODELS:
        raise ValueError(
            f"Unknown guitar model '{model_name}'. "
            f"Valid names: {GUITAR_SHAPES}"
        )
    m = _MODELS[model_name]
    pts = np.array(m["pts"], dtype=float)
    k   = m.get("k", 3)
    contour = _spline_through_points(pts, n_points, k=k)
    return contour * scale


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _spline_through_points(pts: np.ndarray, n_points: int = 120,
                            k: int = 3) -> np.ndarray:
    """Fit a periodic B-spline of degree k through pts."""
    closed = np.vstack([pts, pts[[0]]])
    tck, _ = splprep([closed[:, 0], closed[:, 1]], s=0, per=True, k=k)
    u_new = np.linspace(0, 1, n_points, endpoint=False)
    x_sm, y_sm = splev(u_new, tck)
    return np.column_stack([x_sm, y_sm])


# ---------------------------------------------------------------------------
# CLI: visualise all 8 models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    for ax, name in zip(axes.flat, GUITAR_SHAPES):
        c = get_guitar_contour(name)
        closed = np.vstack([c, c[[0]]])
        ax.plot(closed[:, 0], closed[:, 1], "b-", linewidth=1.5)
        ax.fill(c[:, 0], c[:, 1], alpha=0.12)
        kp = np.array(_MODELS[name]["pts"])
        ax.scatter(kp[:, 0], kp[:, 1], s=12, c="r", zorder=4)
        h = c[:, 1].max() - c[:, 1].min()
        w = c[:, 0].max() - c[:, 0].min()
        ax.set_title(f"{name}\n{w:.0f} × {h:.0f} mm", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Guitar body shapes — 8 models", fontsize=13, y=1.01)
    plt.tight_layout()
    out = "guitar_shapes_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
