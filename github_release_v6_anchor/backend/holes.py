"""
holes.py
--------
Predefined sound-hole shapes as CadQuery 2D wires on the XY plane.
Each function returns a cq.Workplane whose active wire can be extruded.
"""

import math
import cadquery as cq


def create_round_hole(diameter_mm: float) -> cq.Workplane:
    """Circular sound hole centered at origin."""
    return cq.Workplane("XY").circle(diameter_mm / 2)


def create_oval_hole(length_mm: float, width_mm: float) -> cq.Workplane:
    """Elliptical sound hole (length along X, width along Y) centered at origin."""
    return cq.Workplane("XY").ellipse(length_mm / 2, width_mm / 2)


def create_f_hole(
    length_mm: float,
    slot_width_mm: float = 8.0,
    offset_x_mm: float = 30.0,
    position_y_mm: float = 0.0,
) -> cq.Workplane:
    """
    F-hole pair: two narrow oval slots, mirrored left/right about the body center.

    Parameters
    ----------
    length_mm      : total slot height [mm]
    slot_width_mm  : slot width [mm] (default 8 mm)
    offset_x_mm    : distance of each slot center from body X-center [mm]
    position_y_mm  : Y offset for the pair (positive = toward neck) [mm]

    Returns
    -------
    cq.Workplane containing the combined (left + right) f-hole solid wire.
    The caller should extrude to body_thickness and cut from the body.
    """
    a = slot_width_mm / 2   # semi-axis along X
    b = length_mm / 2       # semi-axis along Y

    left  = (cq.Workplane("XY")
             .transformed(offset=cq.Vector(-offset_x_mm, position_y_mm, 0))
             .ellipse(a, b))

    right = (cq.Workplane("XY")
             .transformed(offset=cq.Vector( offset_x_mm, position_y_mm, 0))
             .ellipse(a, b))

    # Combine into a single compound so the caller can .extrude() once
    left_solid  = left.extrude(1)   # thin placeholder for boolean
    right_solid = right.extrude(1)
    return left_solid.union(right_solid)


def create_user_hole(contour_mm: list) -> cq.Workplane:
    """
    Sound hole from a user-drawn pixel contour.

    Parameters
    ----------
    contour_mm : list of (x_mm, y_mm) points forming a closed polygon
                 (already smoothed / downsampled by model_builder.py)

    Returns
    -------
    cq.Workplane with the closed spline wire ready for extrusion.
    """
    if len(contour_mm) < 3:
        raise ValueError("User hole contour must have at least 3 points.")

    pts = [cq.Vector(x, y, 0) for x, y in contour_mm]
    # Close the spline
    pts_closed = pts + [pts[0]]

    return (cq.Workplane("XY")
            .spline([p.toTuple() for p in pts_closed], includeCurrent=False)
            .close())


# ── convenience: build a solid (extruded) hole ready for .cut() ──────────────

def hole_solid(
    hole_type: str,
    params: dict,
    hole_center_mm: tuple,
    body_thickness_mm: float,
    user_contour_mm: list = None,
    z_start: float = None,
    z_height: float = None,
) -> cq.Workplane | None:
    """
    Return an extruded solid for the given hole type, positioned at hole_center_mm.

    Parameters
    ----------
    hole_type        : "none" | "round" | "oval" | "f-hole" | "user-defined"
    params           : hole_params dict from guitar_model.json
    hole_center_mm   : (x_mm, y_mm) center on the XY plane
    body_thickness_mm: full body thickness [mm] (default through-cut height)
    user_contour_mm  : list of (x, y) points for "user-defined" (already in body coords)
    z_start          : lower z of the cut solid [mm].  Default 0 = through-cut.
                       For a hollow body, pass the cavity-top z so the hole only
                       removes the TOP plate and leaves the back plate intact.
    z_height         : height of the cut solid [mm].  Default = body_thickness_mm.

    Returns
    -------
    cq.Workplane solid, or None for hole_type "none".
    """
    if hole_type == "none":
        return None

    cx, cy = hole_center_mm
    z0 = 0.0 if z_start is None else float(z_start)
    h  = float(body_thickness_mm) if z_height is None else float(z_height)

    if hole_type == "round":
        d = params["round"]["diameter"]
        wp = (cq.Workplane("XY")
              .transformed(offset=cq.Vector(cx, cy, 0))
              .circle(d / 2)
              .extrude(h))

    elif hole_type == "oval":
        hp = params["oval"]
        wp = (cq.Workplane("XY")
              .transformed(offset=cq.Vector(cx, cy, 0))
              .ellipse(hp["length"] / 2, hp["width"] / 2)
              .extrude(h))

    elif hole_type == "f-hole":
        hp = params["f-hole"]
        length  = hp["length"]
        pos_y   = hp.get("position_y", 0.0)
        slot_w  = hp.get("slot_width", 8.0)
        off_x   = hp.get("offset_x",  30.0)
        a, b    = slot_w / 2, length / 2

        left_solid = (cq.Workplane("XY")
                      .transformed(offset=cq.Vector(cx - off_x, cy + pos_y, 0))
                      .ellipse(a, b)
                      .extrude(h))
        right_solid = (cq.Workplane("XY")
                       .transformed(offset=cq.Vector(cx + off_x, cy + pos_y, 0))
                       .ellipse(a, b)
                       .extrude(h))
        wp = left_solid.union(right_solid)

    elif hole_type == "user-defined":
        if not user_contour_mm or len(user_contour_mm) < 3:
            raise ValueError("user_contour_mm required for user-defined hole.")
        wp = create_user_hole(user_contour_mm).extrude(h)

    else:
        raise ValueError(f"Unknown hole_type: {hole_type!r}")

    # Lift the cut solid so it spans z ∈ [z0, z0 + h].  For a hollow body this
    # restricts the cut to the top plate (back plate left intact).
    if z0:
        wp = wp.translate((0.0, 0.0, z0))

    return wp
