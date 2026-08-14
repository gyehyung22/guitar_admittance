"""
mesh_gen.py
-----------
STEP → Gmsh tetrahedral mesh (.msh 2.2).

Bridge saddle point is embedded into the top face so the mesh has a
guaranteed node at that coordinate. Physical Point tag=100 is used by
fenics_admittance.py.

ElmerGrid conversion removed — FEniCSx reads .msh directly.
"""

import sys
from pathlib import Path

import gmsh
import numpy as np

# ---------------------------------------------------------------------------
# Default configuration (used by main() only)
# ---------------------------------------------------------------------------
STEP_FILE      = Path("guitar_test.step")
MESH_DIR       = Path("mesh")
BRIDGE_COORDS  = (0.0, -225.0, 100.0)
MESH_SIZE_MIN  = 4.0
MESH_SIZE_MAX  = 6.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_bridge_to_surface(bridge_coords: tuple, face_tag: int) -> tuple:
    """Project bridge (x, y, z) onto the top surface.

    Works for both flat-top and archtop guitars.  The bridge point MUST lie
    on the surface for Gmsh embed() to work — if the user-specified z is wrong
    (e.g. archtop where the surface z varies), this snaps it correctly.
    """
    bx, by, bz = bridge_coords
    try:
        result = gmsh.model.getClosestPoint(2, face_tag, (bx, by, bz))
        bx_s, by_s, bz_s = result[0][0], result[0][1], result[0][2]
        dist = ((bx_s - bx)**2 + (by_s - by)**2 + (bz_s - bz)**2) ** 0.5
        if dist > 0.1:
            print(f"Bridge projected onto top surface: "
                  f"({bx:.2f}, {by:.2f}, {bz:.2f}) → "
                  f"({bx_s:.2f}, {by_s:.2f}, {bz_s:.2f})  [{dist:.2f} mm snap]")
        return (bx_s, by_s, bz_s)
    except Exception as exc:
        # Fallback: snap z to face bounding-box midpoint (reliable for flat tops)
        print(f"Closest-point projection unavailable ({exc}); using bbox-z snap")
        _, _, zmin, _, _, zmax = gmsh.model.getBoundingBox(2, face_tag)
        bz_snap = (zmin + zmax) / 2.0
        if abs(bz_snap - bz) > 0.5:
            print(f"Bridge z snapped: {bz:.2f} → {bz_snap:.2f} mm")
        return (bx, by, bz_snap)


def _find_top_face(bridge_coords: tuple) -> int:
    """Return the surface tag of the top face containing the bridge (x, y) position.

    Works for flat-top and archtop (curved) guitars.
    Priority:
      1. Flat face where zmin ≈ zmax ≈ bz  (original flat-top path)
      2. Topmost face whose (x, y) bounding box contains the bridge point
    """
    bx, by, bz = bridge_coords
    MARGIN = 2.0  # mm tolerance

    flat_candidates = []   # (area, tag) — flat face at z=bz
    arch_candidates = []   # (zmax, tag) — curved face containing (bx, by)

    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        xy_hit = (xmin - MARGIN <= bx <= xmax + MARGIN) and \
                 (ymin - MARGIN <= by <= ymax + MARGIN)

        if abs(zmin - bz) < 1.0 and abs(zmax - bz) < 1.0:
            area = (xmax - xmin) * (ymax - ymin)
            flat_candidates.append((area, tag))
        elif xy_hit and zmax >= bz - MARGIN:
            arch_candidates.append((zmax, tag))

    if flat_candidates:
        flat_candidates.sort(reverse=True)
        chosen = flat_candidates[0][1]
        print(f"Top face (flat): tag={chosen}  ({len(flat_candidates)} candidate(s) at z={bz})")
        return chosen

    if arch_candidates:
        arch_candidates.sort(reverse=True)
        chosen = arch_candidates[0][1]
        print(f"Top face (archtop/curved): tag={chosen}  zmax={arch_candidates[0][0]:.2f} mm")
        return chosen

    print("\n=== All surfaces bounding boxes ===")
    print(f"{'tag':>5}  {'zmin':>8}  {'zmax':>8}  {'xmin':>10}  {'xmax':>10}  {'ymin':>10}  {'ymax':>10}")
    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        print(f"{tag:>5}  {zmin:>8.3f}  {zmax:>8.3f}  {xmin:>10.3f}  {xmax:>10.3f}  {ymin:>10.3f}  {ymax:>10.3f}")
    raise RuntimeError(
        f"No top surface found containing bridge (x={bx}, y={by}, z={bz}) mm. "
        "Check BRIDGE_COORDS or model orientation."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plate_element_size(t_mm: float, plate_min_size: float = 1.5,
                       mesh_size_max: float = 6.0) -> float:
    """Target element edge length [mm] for a plate zone of real thickness `t_mm`.

    Aims for ~2 elements through the thickness while respecting a lower floor and
    the global maximum:  clamp(t/2, plate_min_size, mesh_size_max).

    Examples (production plate_min_size=1.5, mesh_size_max=6.0):
        plate_element_size(3.0) == 1.5
        plate_element_size(6.5) == 3.25
    """
    return min(mesh_size_max, max(plate_min_size, t_mm / 2.0))


def generate_mesh(
    step_file,
    bridge_coords: tuple = (0.0, -225.0, 100.0),
    mesh_size_min: float = 4.0,
    mesh_size_max: float = 6.0,
    output_dir="mesh/",
    top_plate_thickness: float = 0.0,
    back_plate_thickness: float = 0.0,
    plate_min_size: float = 1.5,
    plate_thickness: float = 0.0,
) -> Path:
    """
    Import STEP, embed bridge point, generate tet mesh, write MSH 2.2.

    Parameters
    ----------
    step_file            : path to .step file
    bridge_coords        : (x, y, z) in mm — point embedded into top face
    mesh_size_min        : minimum element edge length [mm]
    mesh_size_max        : maximum element edge length [mm]
    output_dir           : directory for output mesh.msh
    top_plate_thickness  : real top-plate thickness [mm].  If > 0, the top plate
                           zone  z ∈ [z_max - top_plate_thickness, z_max]  is
                           refined.  For a flat hollow body this equals
                           body_thickness·(1 - cavity_ratio).
    back_plate_thickness : real back-plate thickness [mm] (fixed BACK_PLATE_MM in
                           the CAD model).  If > 0, the back plate zone
                           z ∈ [z_min, z_min + back_plate_thickness] is refined.
    plate_min_size       : floor on plate element edge length [mm].
    plate_thickness      : DEPRECATED symmetric fallback — if > 0 and the two
                           explicit thicknesses are unset, both plates use this
                           value (old behaviour; mismatched real geometry).

    The top and back plates are refined INDEPENDENTLY because the CAD geometry is
    asymmetric (variable top, fixed-2 mm back).  Element size per plate targets
    ~2 elements through thickness: size = clamp(t/2, plate_min_size, mesh_size_max).

    Returns
    -------
    Path to the written mesh.msh file.
    """
    # Back-compat: a single symmetric plate_thickness sets both plates.
    if plate_thickness > 0 and top_plate_thickness <= 0 and back_plate_thickness <= 0:
        top_plate_thickness = back_plate_thickness = plate_thickness
    step_file  = Path(step_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / "mesh.msh"

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("guitar")

        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()

        vol_tags = [tag for _, tag in gmsh.model.getEntities(3)]
        if not vol_tags:
            raise RuntimeError("No 3-D volumes found in STEP file.")
        print(f"Volumes: {vol_tags}")

        # `bridge_coords` accepts a single (x,y,z) OR a list of points [mm]; with
        # bridge batching EVERY driven point is embedded as an exact mesh vertex so
        # the modal solver reads it with ~zero snap distance.  The FIRST point keeps
        # the tag=100 "BridgePoint" group (load_mesh's nominal node) for backward
        # compatibility; the rest are embedded (dim-0) but only tagged in the group.
        pts = np.atleast_2d(np.asarray(bridge_coords, dtype=float))

        # Find top face first (from the first point), then project each bridge onto
        # it so embed() works even when the user-specified z is wrong.
        top_face = _find_top_face(tuple(pts[0]))
        bridge_tags = []
        proj0 = None
        for k, p in enumerate(pts):
            pp = _project_bridge_to_surface(tuple(p), top_face)
            if k == 0:
                proj0 = pp
            bridge_tags.append(gmsh.model.occ.addPoint(pp[0], pp[1], pp[2]))
        gmsh.model.occ.synchronize()
        print(f"Bridge OCC point tags={bridge_tags}  (first coords={proj0})")

        gmsh.model.mesh.embed(0, bridge_tags, 2, top_face)

        gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name="Body")
        surf_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="OuterSurface")
        gmsh.model.addPhysicalGroup(0, [bridge_tags[0]], tag=100, name="BridgePoint")
        if len(bridge_tags) > 1:
            gmsh.model.addPhysicalGroup(0, bridge_tags[1:], tag=101, name="BridgePointsExtra")

        if top_plate_thickness > 0 or back_plate_thickness > 0:
            # Adaptive mesh: refine each real plate independently (top != back).
            # size = clamp(t/2, plate_min_size, mesh_size_max) -> ~2 elems/plate.
            _, _, z_min, _, _, z_max = gmsh.model.getBoundingBox(-1, -1)
            BIG = 1e6  # effectively infinite bounding box in x, y

            def _elem_size(t):
                return plate_element_size(t, plate_min_size, mesh_size_max)

            field_ids = []
            fid = 0

            # Box field: back plate zone  z ∈ [z_min, z_min + back_plate_thickness]
            if back_plate_thickness > 0:
                fid += 1
                bsize = _elem_size(back_plate_thickness)
                gmsh.model.mesh.field.add("Box", fid)
                gmsh.model.mesh.field.setNumber(fid, "VIn",  bsize)
                gmsh.model.mesh.field.setNumber(fid, "VOut", mesh_size_max)
                gmsh.model.mesh.field.setNumber(fid, "XMin", -BIG)
                gmsh.model.mesh.field.setNumber(fid, "XMax",  BIG)
                gmsh.model.mesh.field.setNumber(fid, "YMin", -BIG)
                gmsh.model.mesh.field.setNumber(fid, "YMax",  BIG)
                gmsh.model.mesh.field.setNumber(fid, "ZMin",  z_min)
                gmsh.model.mesh.field.setNumber(fid, "ZMax",  z_min + back_plate_thickness)
                field_ids.append(fid)
                print(f"Back plate:  t={back_plate_thickness:.2f}mm size={bsize:.2f}mm "
                      f"z∈[{z_min:.1f},{z_min + back_plate_thickness:.1f}]")

            # Box field: top plate zone  z ∈ [z_max - top_plate_thickness, z_max]
            if top_plate_thickness > 0:
                fid += 1
                tsize = _elem_size(top_plate_thickness)
                gmsh.model.mesh.field.add("Box", fid)
                gmsh.model.mesh.field.setNumber(fid, "VIn",  tsize)
                gmsh.model.mesh.field.setNumber(fid, "VOut", mesh_size_max)
                gmsh.model.mesh.field.setNumber(fid, "XMin", -BIG)
                gmsh.model.mesh.field.setNumber(fid, "XMax",  BIG)
                gmsh.model.mesh.field.setNumber(fid, "YMin", -BIG)
                gmsh.model.mesh.field.setNumber(fid, "YMax",  BIG)
                gmsh.model.mesh.field.setNumber(fid, "ZMin",  z_max - top_plate_thickness)
                gmsh.model.mesh.field.setNumber(fid, "ZMax",  z_max)
                field_ids.append(fid)
                print(f"Top plate:   t={top_plate_thickness:.2f}mm size={tsize:.2f}mm "
                      f"z∈[{z_max - top_plate_thickness:.1f},{z_max:.1f}]")

            # Min field: take finest size from all plate boxes
            min_fid = fid + 1
            gmsh.model.mesh.field.add("Min", min_fid)
            gmsh.model.mesh.field.setNumbers(min_fid, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_fid)

            # Disable size-from-points/curvature so background field dominates
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            finest = min(_elem_size(t) for t in
                         (back_plate_thickness, top_plate_thickness) if t > 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", finest)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)

        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))
        print(f"Mesh written: {msh_path}")

    finally:
        gmsh.finalize()

    return msh_path


# ===========================================================================
# Conformal multi-domain (3A) air-coupling mesh  —  ADDITIVE, structural path
# above is unchanged.  See docs/air_coupling_theory.md.
# ===========================================================================

# Physical group ids (stable across runs)
PHYS_SOLID_WOOD   = 1     # 3D
PHYS_AIR_INTERNAL = 2     # 3D
PHYS_FSI_TOP      = 11    # 2D
PHYS_FSI_BACK     = 12    # 2D
PHYS_FSI_SIDE     = 13    # 2D
PHYS_SOUNDHOLE    = 14    # 2D
PHYS_BRIDGE_POINT = 100   # 0D (load point, same as structural pipeline)


def _classify_fsi_surfaces(wood_vols, air_vols, back_plate_z, cavity_floor_z,
                           z_tol=1.5):
    """Split surfaces into FSI(top/back/side), soundhole, exterior by volume
    adjacency + geometry.  Returns dict of lists of surface tags + diagnostics.

    FSI      = surface shared by a wood volume AND an air volume (interface).
    soundhole= surface on the air boundary with NO wood neighbour (open hole).
    exterior = wood-only outer surfaces (untagged for coupling).
    """
    wood_set, air_set = set(wood_vols), set(air_vols)
    fsi_top, fsi_back, fsi_side, soundhole, exterior = [], [], [], [], []
    nonconformal = []

    for _, s in gmsh.model.getEntities(2):
        up, _down = gmsh.model.getAdjacencies(2, s)
        up = set(int(v) for v in up)
        has_w, has_a = bool(up & wood_set), bool(up & air_set)
        if has_w and has_a:
            # interface facet — must separate EXACTLY one wood + one air volume,
            # with no third volume neighbour.  Anything else is nonconformal.
            nw, na = len(up & wood_set), len(up & air_set)
            if not (nw == 1 and na == 1 and len(up) == 2):
                nonconformal.append((int(s), nw, na, len(up)))
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, s)
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s)
            dz = zmax - zmin
            if dz < z_tol and abs(cz - back_plate_z) < z_tol:
                fsi_back.append(s)
            elif dz < z_tol and abs(cz - cavity_floor_z) < z_tol:
                fsi_top.append(s)
            else:
                fsi_side.append(s)
        elif has_a and not has_w:
            soundhole.append(s)
        else:
            exterior.append(s)

    return {
        "fsi_top": fsi_top, "fsi_back": fsi_back, "fsi_side": fsi_side,
        "soundhole": soundhole, "exterior": exterior,
        "nonconformal": nonconformal,
    }


def _as_point_list(bridge_coords):
    """Accept a single (x,y,z) or a list/array of (x,y,z) -> list of 3-tuples [mm].

    Pure python (mesh_gen deliberately has no numpy dependency)."""
    seq = list(bridge_coords)
    if len(seq) == 3 and all(isinstance(v, (int, float)) for v in seq):
        return [(float(seq[0]), float(seq[1]), float(seq[2]))]
    out = []
    for p in seq:
        p = list(p)
        if len(p) != 3:
            raise ValueError(f"bridge point must be (x,y,z); got {p}")
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def _conformal_core(wood_vols, air_vols, output_dir, *, bridge_coords,
                    t_hole_mm, soundhole_bc="impedance",
                    mesh_size_min=1.5, mesh_size_max=10.0, air_size=10.0,
                    soundhole_size=3.0, top_plate_thickness=0.0,
                    back_plate_z=2.0, cavity_floor_z=0.0,
                    embed_bridge=True, label="model",
                    expected_soundhole_area_mm2=None,
                    expected_cavity_volume_mm3=None,
                    expected_soundhole_center_xy_mm=None,
                    expected_soundhole_radius_mm=None,
                    geometry_crosscheck_rtol=0.01):
    """Fragment wood+air OCC volumes into a conformal multi-domain mesh, tag
    physical groups, run sanity checks, write mesh + metadata.

    Assumes an ACTIVE gmsh session with the wood/air OCC solids already created
    (NOT yet fragmented).  Used by both the guitar wrapper and the rigid-box
    benchmark.  Returns the metadata dict.
    """
    import json as _json
    import numpy as np
    sys_msg = lambda m: print(f"[air-mesh] {m}")
    sizes = np.asarray([mesh_size_min, mesh_size_max, air_size,
                        soundhole_size, geometry_crosscheck_rtol], float)
    if (not np.all(np.isfinite(sizes)) or np.any(sizes[:4] <= 0.0)
            or mesh_size_min > mesh_size_max
            or geometry_crosscheck_rtol < 0.0
            or geometry_crosscheck_rtol >= 0.25):
        raise ValueError("invalid conformal mesh sizes or geometry_crosscheck_rtol")
    if ((expected_soundhole_center_xy_mm is None)
            != (expected_soundhole_radius_mm is None)):
        raise ValueError("soundhole center and radius expectations must be supplied together")

    # --- OCC BooleanFragments: guarantees shared (conformal) interface faces ---
    obj = [(3, t) for t in wood_vols]
    tool = [(3, t) for t in air_vols]
    out, out_map = gmsh.model.occ.fragment(obj, tool)
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # out_map aligns to (obj + tool); wood inputs first, then air inputs.
    n_wood_in = len(obj)
    wood_out, air_out = [], []
    for i, dimtags in enumerate(out_map):
        vols = [t for (d, t) in dimtags if d == 3]
        (wood_out if i < n_wood_in else air_out).extend(vols)
    wood_out = sorted(set(wood_out)); air_out = sorted(set(air_out))
    sys_msg(f"fragment -> wood vols {wood_out}, air vols {air_out}")
    if not wood_out or not air_out:
        raise RuntimeError("Conformal fragment produced no wood or no air volume "
                           f"(wood={wood_out}, air={air_out}). Check input solids.")

    # --- Classify surfaces ----------------------------------------------------
    cls = _classify_fsi_surfaces(wood_out, air_out, back_plate_z, cavity_floor_z)
    n_fsi = len(cls["fsi_top"]) + len(cls["fsi_back"]) + len(cls["fsi_side"])
    fsi_surfaces = cls["fsi_top"] + cls["fsi_back"] + cls["fsi_side"]
    cavity_volume_pre_mesh_mm3 = sum(
        gmsh.model.occ.getMass(3, t) for t in air_out)
    soundhole_area_pre_mesh_mm2 = sum(
        gmsh.model.occ.getMass(2, s) for s in cls["soundhole"])
    fsi_area_pre_mesh_mm2 = sum(
        gmsh.model.occ.getMass(2, s) for s in fsi_surfaces)
    soundhole_details = []
    for s in cls["soundhole"]:
        mass = float(gmsh.model.occ.getMass(2, s))
        center = [float(v) for v in gmsh.model.occ.getCenterOfMass(2, s)]
        bbox = [float(v) for v in gmsh.model.getBoundingBox(2, s)]
        soundhole_details.append(
            {"surface_tag": int(s), "area_mm2": mass,
             "center_of_mass_mm": center, "bounding_box_mm": bbox})
    if soundhole_area_pre_mesh_mm2 > 0.0:
        soundhole_centroid = (
            np.sum([d["area_mm2"] * np.asarray(d["center_of_mass_mm"])
                    for d in soundhole_details], axis=0)
            / soundhole_area_pre_mesh_mm2)
    else:
        soundhole_centroid = np.full(3, np.nan)

    # Count connected components through shared boundary curves.  OCC can split
    # one circular opening into several faces, but they must still form one opening.
    curve_to_faces = {}
    for s in cls["soundhole"]:
        for dim, tag in gmsh.model.getBoundary(
                [(2, s)], combined=False, oriented=False, recursive=False):
            if dim == 1:
                curve_to_faces.setdefault(int(tag), set()).add(int(s))
    adjacency = {int(s): set() for s in cls["soundhole"]}
    for faces in curve_to_faces.values():
        for face in faces:
            adjacency[face].update(faces - {face})
    components = 0
    unseen = set(adjacency)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            neighbours = adjacency[stack.pop()] & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
    sys_msg(f"surfaces: FSI top/back/side = "
            f"{len(cls['fsi_top'])}/{len(cls['fsi_back'])}/{len(cls['fsi_side'])}, "
            f"soundhole = {len(cls['soundhole'])}, exterior = {len(cls['exterior'])}")

    # --- Conformal sanity gate (fail loud) ------------------------------------
    conformal_ok = True
    all_vols = set(t for (_d, t) in gmsh.model.getEntities(3))
    overlap = set(wood_out) & set(air_out)
    unassigned = all_vols - set(wood_out) - set(air_out)
    if overlap:
        conformal_ok = False
        sys_msg(f"WOOD/AIR volume tags OVERLAP (not disjoint): {sorted(overlap)}")
    if unassigned:
        conformal_ok = False
        sys_msg(f"UNASSIGNED 3D volume(s) after fragment: {sorted(unassigned)} "
                f"(every volume must be SOLID_WOOD or AIR_INTERNAL).")
    if cls["nonconformal"]:
        conformal_ok = False
        sys_msg("NONCONFORMAL interface facets (surf, n_wood, n_air, n_adj) — each "
                f"must be (s,1,1,2): {cls['nonconformal'][:20]}")
    if n_fsi == 0:
        conformal_ok = False
        sys_msg("NO FSI interface facets found — wood and air do not share a surface.")
    # Each air-bounding facet must be either a shared FSI facet or an open
    # soundhole facet; a large air-only area that is NOT the soundhole signals an
    # unmerged (nonconformal) interface.
    if soundhole_bc != "closed" and not cls["soundhole"]:
        conformal_ok = False
        sys_msg("NO open soundhole facet found (expected for an open hole).")

    soundhole_center_error_mm = None
    soundhole_bbox_excess_mm = None
    soundhole_planarity_span_mm = None
    soundhole_opening_z_error_mm = None
    if expected_soundhole_center_xy_mm is not None:
        expected_xy = np.asarray(expected_soundhole_center_xy_mm, float)
        expected_r = float(expected_soundhole_radius_mm)
        if (expected_xy.shape != (2,) or not np.all(np.isfinite(expected_xy))
                or not np.isfinite(expected_r) or expected_r <= 0.0):
            raise ValueError("invalid expected soundhole center/radius")
        soundhole_center_error_mm = float(np.linalg.norm(
            soundhole_centroid[:2] - expected_xy))
        if soundhole_details:
            xmin = min(d["bounding_box_mm"][0] for d in soundhole_details)
            ymin = min(d["bounding_box_mm"][1] for d in soundhole_details)
            zmin = min(d["bounding_box_mm"][2] for d in soundhole_details)
            xmax = max(d["bounding_box_mm"][3] for d in soundhole_details)
            ymax = max(d["bounding_box_mm"][4] for d in soundhole_details)
            zmax_h = max(d["bounding_box_mm"][5] for d in soundhole_details)
            soundhole_bbox_excess_mm = float(max(
                expected_xy[0] - expected_r - xmin,
                xmin - (expected_xy[0] - expected_r),
                expected_xy[1] - expected_r - ymin,
                ymin - (expected_xy[1] - expected_r),
                xmax - (expected_xy[0] + expected_r),
                (expected_xy[0] + expected_r) - xmax,
                ymax - (expected_xy[1] + expected_r),
                (expected_xy[1] + expected_r) - ymax,
            ))
            # BBox equality for the intended round opening is stricter and more
            # useful than merely checking that a leak lies somewhere in the disk.
            bbox_expected = np.array([expected_xy[0] - expected_r,
                                      expected_xy[1] - expected_r,
                                      expected_xy[0] + expected_r,
                                      expected_xy[1] + expected_r])
            bbox_actual = np.array([xmin, ymin, xmax, ymax])
            soundhole_bbox_excess_mm = float(np.max(np.abs(
                bbox_actual - bbox_expected)))
            soundhole_planarity_span_mm = float(zmax_h - zmin)
            # AIR_INTERNAL intentionally excludes the soundhole neck: its
            # inertance is represented by the lumped port.  The acoustic opening
            # is therefore the cavity-side throat plane (the underside of the top
            # plate), not the outer wood top surface.  ``cavity_floor_z`` is the
            # historical API name for that cavity ceiling coordinate.
            soundhole_opening_z_error_mm = abs(
                float(soundhole_centroid[2]) - float(cavity_floor_z))
        position_tol = max(0.1, expected_r * float(geometry_crosscheck_rtol))
        if (components != 1
                or soundhole_center_error_mm > position_tol
                or soundhole_bbox_excess_mm is None
                or soundhole_bbox_excess_mm > position_tol
                or soundhole_planarity_span_mm is None
                or soundhole_planarity_span_mm > 0.1
                or soundhole_opening_z_error_mm is None
                or soundhole_opening_z_error_mm > 0.1):
            conformal_ok = False
            sys_msg("soundhole position/planarity/connectivity cross-check FAILED: "
                    f"center={soundhole_center_error_mm}, bbox={soundhole_bbox_excess_mm}, "
                    f"zspan={soundhole_planarity_span_mm}, "
                    f"zerr={soundhole_opening_z_error_mm}, components={components}")

    def _crosscheck(name, actual, expected):
        nonlocal conformal_ok
        if expected is None:
            return None
        actual = float(actual); expected = float(expected)
        if (not np.isfinite(actual) or not np.isfinite(expected)
                or actual <= 0.0 or expected <= 0.0):
            conformal_ok = False
            sys_msg(f"{name} cross-check has invalid values: actual={actual}, "
                    f"expected={expected}")
            return float("inf")
        rel = abs(actual - expected) / expected
        if rel > float(geometry_crosscheck_rtol):
            conformal_ok = False
            sys_msg(f"{name} cross-check FAILED: actual={actual:.9g}, "
                    f"expected={expected:.9g}, rel={rel:.3%} > "
                    f"{float(geometry_crosscheck_rtol):.3%}")
        return float(rel)

    soundhole_area_rel_error = _crosscheck(
        "soundhole area", soundhole_area_pre_mesh_mm2,
        expected_soundhole_area_mm2)
    cavity_volume_rel_error = _crosscheck(
        "cavity volume", cavity_volume_pre_mesh_mm3,
        expected_cavity_volume_mm3)
    if not conformal_ok:
        raise RuntimeError("Conformal interface verification FAILED — refusing to "
                           "continue (see [air-mesh] messages above).")
    sys_msg(f"conformal gate OK: {len(wood_out)} wood + {len(air_out)} air vols, "
            f"0 unassigned, {n_fsi} FSI facets all (1 wood,1 air).")

    # --- Physical groups ------------------------------------------------------
    gmsh.model.addPhysicalGroup(3, wood_out, tag=PHYS_SOLID_WOOD, name="SOLID_WOOD")
    gmsh.model.addPhysicalGroup(3, air_out, tag=PHYS_AIR_INTERNAL, name="AIR_INTERNAL")
    if cls["fsi_top"]:
        gmsh.model.addPhysicalGroup(2, cls["fsi_top"], tag=PHYS_FSI_TOP, name="FSI_TOP_INNER")
    if cls["fsi_back"]:
        gmsh.model.addPhysicalGroup(2, cls["fsi_back"], tag=PHYS_FSI_BACK, name="FSI_BACK_INNER")
    if cls["fsi_side"]:
        gmsh.model.addPhysicalGroup(2, cls["fsi_side"], tag=PHYS_FSI_SIDE, name="FSI_SIDE_INNER")
    if cls["soundhole"]:
        gmsh.model.addPhysicalGroup(2, cls["soundhole"], tag=PHYS_SOUNDHOLE, name="SOUNDHOLE")

    # --- Bridge load point(s) (embed on wood top face) -------------------------
    # `bridge_coords` may be a single (x,y,z) or a LIST of points: with bridge
    # batching every driven point must be an EXACT mesh node, otherwise the
    # non-first bridges would only be nearest-node approximations.  All embedded
    # points share the BRIDGE_LOAD physical group; the solvers still resolve each
    # bridge by nearest node, which now lands exactly on the embedded vertex.
    embedded_bridges = []
    if embed_bridge and bridge_coords is not None:
        pts = _as_point_list(bridge_coords)
        btags = []
        for p in pts:
            top_face = _find_top_face(p)
            bcs = _project_bridge_to_surface(p, top_face)
            btag = gmsh.model.occ.addPoint(*bcs)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.embed(0, [btag], 2, top_face)
            btags.append(btag)
            shift = sum((float(a) - float(b)) ** 2 for a, b in zip(bcs, p)) ** 0.5
            embedded_bridges.append({
                "requested_mm": [float(v) for v in p],
                "embedded_mm": [float(v) for v in bcs],
                "projection_shift_mm": float(shift),
            })
        gmsh.model.addPhysicalGroup(0, btags, tag=PHYS_BRIDGE_POINT, name="BRIDGE_LOAD")
        sys_msg(f"embedded {len(btags)} bridge point(s) into the wood top face")

    # --- Mesh sizing: coarse air, fine plates + soundhole ---------------------
    _, _, z_min, _, _, z_max = gmsh.model.getBoundingBox(-1, -1)
    BIG = 1e6
    fields = []
    fid = 0

    def _constant_on_volumes(size, volumes):
        nonlocal fid
        fid += 1
        gmsh.model.mesh.field.add("Constant", fid)
        gmsh.model.mesh.field.setNumber(fid, "VIn", float(size))
        gmsh.model.mesh.field.setNumbers(fid, "VolumesList", list(volumes))
        fields.append(fid)

    # Domain-specific background sizes.  Without these, using air_size as the
    # global maximum also coarsens wood side walls beyond mesh_size_max.
    _constant_on_volumes(mesh_size_max, wood_out)
    _constant_on_volumes(air_size, air_out)

    def _box(zlo, zhi, vin, xy_lo=-BIG, xy_hi=BIG, x0=-BIG, x1=BIG,
             y0=-BIG, y1=BIG, restrict_volumes=None):
        nonlocal fid
        fid += 1
        box_fid = fid
        gmsh.model.mesh.field.add("Box", box_fid)
        gmsh.model.mesh.field.setNumber(box_fid, "VIn", vin)
        gmsh.model.mesh.field.setNumber(box_fid, "VOut", max(mesh_size_max, air_size))
        gmsh.model.mesh.field.setNumber(box_fid, "XMin", x0); gmsh.model.mesh.field.setNumber(box_fid, "XMax", x1)
        gmsh.model.mesh.field.setNumber(box_fid, "YMin", y0); gmsh.model.mesh.field.setNumber(box_fid, "YMax", y1)
        gmsh.model.mesh.field.setNumber(box_fid, "ZMin", zlo); gmsh.model.mesh.field.setNumber(box_fid, "ZMax", zhi)
        if restrict_volumes:
            fid += 1
            gmsh.model.mesh.field.add("Restrict", fid)
            gmsh.model.mesh.field.setNumber(fid, "InField", box_fid)
            gmsh.model.mesh.field.setNumbers(fid, "VolumesList", list(restrict_volumes))
            fields.append(fid)
        else:
            fields.append(box_fid)

    # back plate band, top plate band (fine for thin wood)
    if back_plate_z > 0:
        _box(z_min, z_min + back_plate_z,
             plate_element_size(back_plate_z, mesh_size_min, mesh_size_max),
             restrict_volumes=wood_out)
    if top_plate_thickness > 0:
        _box(z_max - top_plate_thickness, z_max,
             plate_element_size(top_plate_thickness, mesh_size_min, mesh_size_max),
             restrict_volumes=wood_out)
    # soundhole refinement near cavity_floor_z
    if cls["soundhole"]:
        sxmin, symin, szmin, sxmax, symax, szmax = (1e9, 1e9, 1e9, -1e9, -1e9, -1e9)
        for s in cls["soundhole"]:
            bb = gmsh.model.getBoundingBox(2, s)
            sxmin, symin, szmin = min(sxmin, bb[0]), min(symin, bb[1]), min(szmin, bb[2])
            sxmax, symax, szmax = max(sxmax, bb[3]), max(symax, bb[4]), max(szmax, bb[5])
        pad = 3.0
        _box(szmin - pad, szmax + pad, soundhole_size,
             x0=sxmin - pad, x1=sxmax + pad, y0=symin - pad, y1=symax + pad)

    if fields:
        fid += 1
        gmsh.model.mesh.field.add("Min", fid)
        gmsh.model.mesh.field.setNumbers(fid, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fid)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max(mesh_size_max, air_size))
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.mesh.generate(3)
    try:
        gmsh.model.mesh.optimize("Netgen")
    except Exception:
        pass

    def _domain_element_stats(volumes):
        tags = []
        for volume in volumes:
            _types, element_tags, _nodes = gmsh.model.mesh.getElements(3, volume)
            for block in element_tags:
                tags.extend(int(v) for v in block)
        if not tags:
            return {"n_elements": 0}
        min_edges = np.asarray(
            gmsh.model.mesh.getElementQualities(tags, "minEdge"), float)
        max_edges = np.asarray(
            gmsh.model.mesh.getElementQualities(tags, "maxEdge"), float)
        return {
            "n_elements": int(len(tags)),
            "min_edge_mm": float(np.min(min_edges)),
            "median_min_edge_mm": float(np.median(min_edges)),
            "p95_max_edge_mm": float(np.percentile(max_edges, 95.0)),
            "max_edge_mm": float(np.max(max_edges)),
        }

    wood_element_stats = _domain_element_stats(wood_out)
    air_element_stats = _domain_element_stats(air_out)

    # --- Geometry quantities (mm -> SI) + Helmholtz estimate ------------------
    cavity_volume_mm3 = sum(gmsh.model.occ.getMass(3, t) for t in air_out)
    soundhole_area_mm2 = sum(gmsh.model.occ.getMass(2, s) for s in cls["soundhole"]) \
        if cls["soundhole"] else 0.0
    fsi_area_mm2 = sum(gmsh.model.occ.getMass(2, s)
                       for s in cls["fsi_top"] + cls["fsi_back"] + cls["fsi_side"])

    V = cavity_volume_mm3 * 1e-9
    S = soundhole_area_mm2 * 1e-6
    t_hole = t_hole_mm * 1e-3
    helm = {}
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from acoustic_helmholtz import helmholtz_estimate
        helm = helmholtz_estimate(V, S, t_hole).to_dict() if (V > 0 and S > 0) else {}
    except Exception as exc:
        sys_msg(f"Helmholtz estimate skipped ({exc})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / "mesh_air.msh"
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(msh_path))
    sys_msg(f"conformal mesh written: {msh_path}")

    meta = {
        "has_internal_air": True,
        "label": label,
        "msh_file": str(msh_path),
        "n_wood_volumes": len(wood_out),
        "n_air_volumes": len(air_out),
        # Soundhole neck length t_hole (= top plate thickness) — REQUIRED by
        # air_acoustics for L_eff/M_h; do not let it fall back to a default.
        "top_plate_thickness_mm": float(t_hole_mm),
        "soundhole_thickness_mm": float(t_hole_mm),
        "cavity_volume_mm3": cavity_volume_mm3,
        "cavity_volume_m3": V,
        "soundhole_area_mm2": soundhole_area_mm2,
        "soundhole_area_m2": S,
        "fsi_surface_area_mm2": fsi_area_mm2,
        "n_fsi_surfaces": int(n_fsi),
        "n_soundhole_surfaces": int(len(cls["soundhole"])),
        "n_soundhole_connected_components": int(components),
        "n_air_boundary_surfaces": int(n_fsi + len(cls["soundhole"])),
        "expected_soundhole_area_mm2": (
            None if expected_soundhole_area_mm2 is None
            else float(expected_soundhole_area_mm2)),
        "expected_cavity_volume_mm3": (
            None if expected_cavity_volume_mm3 is None
            else float(expected_cavity_volume_mm3)),
        "soundhole_area_rel_error": soundhole_area_rel_error,
        "cavity_volume_rel_error": cavity_volume_rel_error,
        "geometry_crosscheck_rtol": float(geometry_crosscheck_rtol),
        "expected_soundhole_center_xy_mm": (
            None if expected_soundhole_center_xy_mm is None
            else [float(v) for v in expected_soundhole_center_xy_mm]),
        "expected_soundhole_radius_mm": (
            None if expected_soundhole_radius_mm is None
            else float(expected_soundhole_radius_mm)),
        "soundhole_centroid_mm": [float(v) for v in soundhole_centroid],
        "soundhole_center_error_mm": soundhole_center_error_mm,
        "soundhole_bbox_error_mm": soundhole_bbox_excess_mm,
        "soundhole_planarity_span_mm": soundhole_planarity_span_mm,
        "expected_soundhole_opening_z_mm": float(cavity_floor_z),
        "soundhole_opening_z_error_mm": soundhole_opening_z_error_mm,
        "soundhole_surfaces": soundhole_details,
        "effective_soundhole_radius": helm.get("effective_soundhole_radius"),
        "effective_neck_length": helm.get("effective_neck_length"),
        "estimated_helmholtz_hz": helm.get("estimated_helmholtz_hz"),
        "air_mesh_size_target": air_size,
        "wood_mesh_size_target_min_mm": float(mesh_size_min),
        "wood_mesh_size_target_max_mm": float(mesh_size_max),
        # The element-size target actually applied to the top-plate refinement zone
        # (clamp(top/2, plate_min_size, mesh_size_max)) — lets a sentinel verify that
        # 3.0->1.5, 6.5->3.25 mm was really used, not just intended.
        "top_plate_mesh_size_target_mm": (
            plate_element_size(float(top_plate_thickness), mesh_size_min, mesh_size_max)
            if top_plate_thickness > 0 else None),
        "air_mesh_size_target_mm": float(air_size),
        "soundhole_mesh_size_target_mm": float(soundhole_size),
        "wood_element_stats": wood_element_stats,
        "air_element_stats": air_element_stats,
        "soundhole_bc": soundhole_bc,
        "conformal_interface_verified": bool(conformal_ok),
        # Every driven bridge point is an EXACT embedded mesh vertex (see above).
        # `projection_shift_mm` is the distance the requested point was moved onto
        # the top face; the solvers' own snap distance to these nodes should be ~0.
        "embedded_bridge_points": embedded_bridges,
        "bridge_response_semantics": (
            "bridge points are embedded vertices; solver picks the nearest wood node, "
            "which is the embedded vertex itself (snap ~0)"),
        "physical_groups": {
            "SOLID_WOOD": PHYS_SOLID_WOOD, "AIR_INTERNAL": PHYS_AIR_INTERNAL,
            "FSI_TOP_INNER": PHYS_FSI_TOP, "FSI_BACK_INNER": PHYS_FSI_BACK,
            "FSI_SIDE_INNER": PHYS_FSI_SIDE, "SOUNDHOLE": PHYS_SOUNDHOLE,
            "BRIDGE_LOAD": PHYS_BRIDGE_POINT,
        },
    }
    (output_dir / "air_mesh_meta.json").write_text(_json.dumps(meta, indent=2))
    sys_msg(f"metadata: V={V*1e3:.3f} L, S={S*1e4:.2f} cm^2, "
            f"f_H~={meta['estimated_helmholtz_hz']} Hz")
    return meta


def generate_conformal_air_mesh(
    wood_step,
    air_step,
    output_dir,
    bridge_coords=(0.0, -225.0, 100.0),
    t_hole_mm: float = 3.0,
    soundhole_bc: str = "impedance",
    mesh_size_min: float = 1.5,
    mesh_size_max: float = 10.0,
    air_size: float = 10.0,
    soundhole_size: float = 3.0,
    top_plate_thickness: float = 0.0,
    back_plate_z: float = 2.0,
    cavity_floor_z: float = 0.0,
    expected_soundhole_area_mm2: float | None = None,
    expected_cavity_volume_mm3: float | None = None,
    expected_soundhole_center_xy_mm=None,
    expected_soundhole_radius_mm: float | None = None,
    geometry_crosscheck_rtol: float = 0.01,
) -> dict:
    """Conformal multi-domain mesh from a wood STEP + an internal-air STEP.

    Imports both solids into ONE OCC model and uses BooleanFragments to enforce a
    shared (conformal) wood-air interface, then tags SOLID_WOOD / AIR_INTERNAL and
    FSI / SOUNDHOLE surfaces and writes mesh_air.msh + air_mesh_meta.json.

    `bridge_coords` accepts a single (x,y,z) OR a list of points [mm]: with bridge
    batching every driven point is embedded as an exact mesh vertex.

    Returns the metadata dict.  Raises RuntimeError if the interface is not
    conformal.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("guitar_air")
        wood_in = gmsh.model.occ.importShapes(str(wood_step))
        air_in = gmsh.model.occ.importShapes(str(air_step))
        gmsh.model.occ.synchronize()
        wood_vols = [t for (d, t) in wood_in if d == 3]
        air_vols = [t for (d, t) in air_in if d == 3]
        if not wood_vols or not air_vols:
            raise RuntimeError(f"Expected 3D solids in both STEPs "
                               f"(wood={wood_vols}, air={air_vols}).")
        return _conformal_core(
            wood_vols, air_vols, output_dir,
            bridge_coords=bridge_coords, t_hole_mm=t_hole_mm,
            soundhole_bc=soundhole_bc, mesh_size_min=mesh_size_min,
            mesh_size_max=mesh_size_max, air_size=air_size,
            soundhole_size=soundhole_size, top_plate_thickness=top_plate_thickness,
            back_plate_z=back_plate_z, cavity_floor_z=cavity_floor_z,
            expected_soundhole_area_mm2=expected_soundhole_area_mm2,
            expected_cavity_volume_mm3=expected_cavity_volume_mm3,
            expected_soundhole_center_xy_mm=expected_soundhole_center_xy_mm,
            expected_soundhole_radius_mm=expected_soundhole_radius_mm,
            geometry_crosscheck_rtol=geometry_crosscheck_rtol,
            label=Path(wood_step).stem,
        )
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# CLI entry point (backward-compatible)
# ---------------------------------------------------------------------------

def main():
    MESH_DIR.mkdir(exist_ok=True)
    msh_path = generate_mesh(
        step_file=STEP_FILE,
        bridge_coords=BRIDGE_COORDS,
        mesh_size_min=MESH_SIZE_MIN,
        mesh_size_max=MESH_SIZE_MAX,
        output_dir=MESH_DIR,
    )

if __name__ == "__main__":
    main()
