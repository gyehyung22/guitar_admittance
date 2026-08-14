"""
fsi_coupling.py
---------------
Fluid-structure interface coupling matrix G for the internal air-cavity coupling.

    G_{j i} = ∫_{Γ_sa} N_j^p (N_i^u · n_a) dS          (shape: n_p x n_u)

with n_a the AIR outward normal on the wood-air interface Γ_sa.  Signs / the block
placement (Everstine unsymmetric u-p form) are pinned in
docs/air_coupling_theory.md — this module must match it:

    ⎡ Z_s        -Gᵀ      ⎤ ⎡u⎤ = ⎡F⎤
    ⎣ -ρ0 ω² G    Z_a'    ⎦ ⎣p⎦   ⎣0⎦

Time convention exp(+iωt).

Primary path: multiphenicsx (restricted subdomain spaces + block assembly on the
FULL conformal mesh).  Fallback: custom facet-quadrature assembly.  EITHER WAY the
result is passed through verify_coupling(), which checks:
  * assembled interface area  vs  geometric FSI facet area,
  * air-normal orientation (outward from air),
  * a rigid-translation test integral  Gᵀ·1_p  ==  ∫_Γ n_a dS  (net area vector),
  * matrix shape / nnz / Frobenius norm.

The dolfinx / multiphenicsx imports are deferred into functions so this module
imports for static checks where they are absent (Windows).  The FEM assembly runs
on the server; the geometry helpers below are pure-numpy and unit-tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Physical ids — must match mesh_gen.py
PHYS_SOLID_WOOD = 1
PHYS_AIR_INTERNAL = 2
FSI_TAGS = (11, 12, 13)      # FSI_TOP_INNER, FSI_BACK_INNER, FSI_SIDE_INNER
PHYS_SOUNDHOLE = 14


# ===========================================================================
# Pure-numpy geometry helpers (unit-tested locally)
# ===========================================================================

def triangle_area_normal(v0, v1, v2):
    """Area and unit normal of a triangle (arbitrary orientation)."""
    v0, v1, v2 = map(lambda a: np.asarray(a, float), (v0, v1, v2))
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm == 0:
        return 0.0, np.zeros(3)
    return 0.5 * norm, n / norm


def orient_outward_from_air(normal, facet_centroid, air_cell_centroid):
    """Flip `normal` so it points OUT of the air cell (n_a): away from the air
    cell centroid, through the facet."""
    normal = np.asarray(normal, float)
    if np.dot(normal, np.asarray(facet_centroid, float)
              - np.asarray(air_cell_centroid, float)) < 0.0:
        normal = -normal
    return normal


def triangle_mass_matrix(area):
    """P1 surface mass matrix M_{ij} = ∫_T N_i N_j dS on a triangle of `area`.

        M = area/12 * [[2,1,1],[1,2,1],[1,1,2]]
    """
    return (area / 12.0) * np.array([[2., 1., 1.],
                                     [1., 2., 1.],
                                     [1., 1., 2.]])


def _coord_key(x, micron=1.0):
    """Canonical integer key for a coordinate [mm] at `micron`-level rounding.
    Exact identity for conformal (shared) nodes; NOT a proximity tolerance."""
    return tuple(np.round(np.asarray(x, float) * (1000.0 / micron)).astype(np.int64))


# ===========================================================================
# Mesh + restricted spaces
# ===========================================================================

def load_coupled_mesh(msh_path):
    """Load the FULL conformal mesh with cell + facet tags.

    Uses meshio -> XDMF -> dolfinx read_meshtags (NOT gmshio, which needs the
    gmsh python bindings that this FEniCSx build lacks).  Cell tags carry
    SOLID_WOOD / AIR_INTERNAL; facet tags carry FSI_* / SOUNDHOLE.

    Returns (mesh, cell_tags, facet_tags).
    """
    import tempfile
    import meshio
    from mpi4py import MPI
    from dolfinx.io import XDMFFile

    mio = meshio.read(str(msh_path))
    tet = mio.get_cells_type("tetra")
    tet_tags = mio.get_cell_data("gmsh:physical", "tetra")
    tri = mio.get_cells_type("triangle")
    tri_tags = mio.get_cell_data("gmsh:physical", "triangle")
    pts = mio.points

    comm = MPI.COMM_WORLD
    # Unique temp dir per call (avoids collisions across concurrent runs/sessions).
    # dolfinx reads the meshes fully into memory inside the `with` blocks, so the
    # files are no longer needed once we exit the TemporaryDirectory.
    with tempfile.TemporaryDirectory(prefix="fsi_coupled_") as td:
        dom_xdmf = Path(td) / "coupled_domain.xdmf"
        fac_xdmf = Path(td) / "coupled_facets.xdmf"
        meshio.write(str(dom_xdmf), meshio.Mesh(
            points=pts, cells=[("tetra", tet)],
            cell_data={"name_to_read": [tet_tags.astype(np.int32)]}))
        meshio.write(str(fac_xdmf), meshio.Mesh(
            points=pts, cells=[("triangle", tri)],
            cell_data={"name_to_read": [tri_tags.astype(np.int32)]}))
        with XDMFFile(comm, str(dom_xdmf), "r") as f:
            mesh = f.read_mesh(name="Grid")
            cell_tags = f.read_meshtags(mesh, name="Grid")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with XDMFFile(comm, str(fac_xdmf), "r") as f:
            facet_tags = f.read_meshtags(mesh, name="Grid")
    return mesh, cell_tags, facet_tags


def build_spaces(mesh):
    """Structural vector P1 (V_u) and acoustic scalar P1 (V_p) on the full mesh."""
    import dolfinx.fem as fem
    V_u = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    V_p = fem.functionspace(mesh, ("Lagrange", 1))
    return V_u, V_p


def build_restrictions(mesh, cell_tags, V_u, V_p):
    """multiphenicsx DofMapRestriction: u -> wood dofs, p -> air dofs."""
    import dolfinx.fem as fem
    import multiphenicsx.fem
    tdim = mesh.topology.dim
    wood_cells = cell_tags.find(PHYS_SOLID_WOOD)
    air_cells = cell_tags.find(PHYS_AIR_INTERNAL)
    dofs_u = fem.locate_dofs_topological(V_u, tdim, wood_cells)
    dofs_p = fem.locate_dofs_topological(V_p, tdim, air_cells)
    r_u = multiphenicsx.fem.DofMapRestriction(V_u.dofmap, dofs_u)
    r_p = multiphenicsx.fem.DofMapRestriction(V_p.dofmap, dofs_p)
    return r_u, r_p, dofs_u, dofs_p


# ===========================================================================
# G assembly — multiphenicsx (primary)
# ===========================================================================

def assemble_G_multiphenicsx(mesh, cell_tags, facet_tags, V_u, V_p,
                             air_side_marker=PHYS_AIR_INTERNAL):
    """Assemble G via multiphenicsx restricted block assembly on Γ_sa.

    Interface term uses the AIR-side trace of both fields and the air outward
    normal.  The '+'/'-' interface side is selected so that the restriction of p
    (defined only on air) and the air-cell normal are consistent; this is the
    convention that verify_coupling() checks on the first server run.
    """
    import dolfinx.fem as fem
    import dolfinx.fem.petsc  # noqa: F401
    import multiphenicsx.fem.petsc
    import ufl

    r_u, r_p, _, _ = build_restrictions(mesh, cell_tags, V_u, V_p)

    u = ufl.TrialFunction(V_u)
    q = ufl.TestFunction(V_p)
    n = ufl.FacetNormal(mesh)

    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags)
    # Restrict to the AIR side: build a cell-wise indicator (1 on air, 0 on wood)
    # so ('+')/('-') picks the air trace deterministically.
    DG0 = fem.functionspace(mesh, ("DG", 0))
    air_ind = fem.Function(DG0)
    air_ind.x.array[:] = 0.0
    # Set the indicator via each cell's DG0 dof (cell index != dof index in
    # general — never index x.array by cell id directly).
    dg_dofmap = DG0.dofmap
    air_cells = cell_tags.find(air_side_marker)
    air_dofs = np.array([dg_dofmap.cell_dofs(int(c))[0] for c in air_cells], dtype=np.int64)
    air_ind.x.array[air_dofs] = 1.0

    # air side selector on an interior facet: side whose indicator == 1
    def air(f):
        return f("+") * air_ind("+") + f("-") * air_ind("-")

    # Sign/normal convention (see docs/air_coupling_theory.md):
    #   n("+"), n("-") are the outward normals of the '+' and '-' cells.
    #   Selecting the AIR side gives n_a = n(air side) = outward-from-air normal
    #   (points from air into wood).  NO extra minus sign is applied here.
    n_a = n("+") * air_ind("+") + n("-") * air_ind("-")
    # G_{ji} = ∫ N_j^p (N_i^u · n_a) dS : acoustic-row (q) x structural-col (u).
    # Sum tags explicitly instead of relying on dS((11, 12, 13)) support across
    # UFL/dolfinx versions.
    a_expr = None
    for tag in FSI_TAGS:
        term = air(q) * ufl.dot(air(u), n_a) * dS(tag)
        a_expr = term if a_expr is None else a_expr + term
    a_form = fem.form(a_expr)

    G_petsc = multiphenicsx.fem.petsc.assemble_matrix(
        a_form, restriction=(r_p, r_u))
    G_petsc.assemble()
    return _petsc_to_scipy(G_petsc)


# ===========================================================================
# G assembly — custom facet quadrature (fallback)
# ===========================================================================

def assemble_G_custom(mesh, cell_tags, facet_tags, V_u, V_p):
    """Explicit per-facet assembly of G (P1).  Full control of the air normal.

    For a conformal P1 interface, pressure and displacement share the facet's 3
    vertices, so the facet-local coupling is  G_local[j, (i,c)] = n_a[c]·M_tri[j,i]
    with M_tri the triangle surface mass matrix.  Global dofs are found by exact
    coordinate-key lookup into the (blocked) dof coordinate tables.
    """
    import dolfinx.mesh as dmesh

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    f2c = mesh.topology.connectivity(fdim, tdim)

    # cell -> tag lookup
    cell_tag = np.zeros(mesh.topology.index_map(tdim).size_local
                        + mesh.topology.index_map(tdim).num_ghosts, dtype=np.int32)
    cell_tag[cell_tags.indices] = cell_tags.values

    # dof coordinate -> dof index maps (exact keys).  The mesh here is in METRES
    # (assemble_coupling scaled it), while _coord_key expects mm (it rounds to
    # micron via *1000).  Convert m->mm before keying so the resolution is 1
    # micron, NOT 1 mm — a 1 mm key would collide distinct nodes and silently drop
    # ~8.7% of the interface coupling (observed as G-net < geometric net).
    def _k(p):
        return _coord_key(np.asarray(p, float) * 1e3)

    p_coords = V_p.tabulate_dof_coordinates()[:, :3]
    p_by_key = {_k(x): i for i, x in enumerate(p_coords)}
    bs = V_u.dofmap.index_map_bs                      # 3
    # tabulate_dof_coordinates may return one row per BLOCK (node) or one per
    # scalar dof depending on the dolfinx version; normalize to per-block so the
    # global dof for (block, comp) is block*bs + comp.
    n_blocks_u = (V_u.dofmap.index_map.size_local
                  + V_u.dofmap.index_map.num_ghosts)
    u_coords_raw = V_u.tabulate_dof_coordinates()[:, :3]
    if len(u_coords_raw) == n_blocks_u * bs:
        u_coords = u_coords_raw[::bs]                 # per-scalar-dof -> per-block
    else:
        u_coords = u_coords_raw                       # already per-block
    u_by_key = {_k(x): i for i, x in enumerate(u_coords)}

    x = mesh.geometry.x

    # FSI facets
    fsi_facets = np.concatenate([facet_tags.find(t) for t in FSI_TAGS
                                 if len(facet_tags.find(t))]) \
        if any(len(facet_tags.find(t)) for t in FSI_TAGS) else np.array([], int)

    rows, cols, data = [], [], []
    area_assembled = 0.0
    n_missing = 0                      # facet-node -> dof key lookups that failed
    from air_acoustics import _entities_to_geometry  # reuse helper
    for f in fsi_facets:
        fverts = _entities_to_geometry(mesh, fdim, np.array([f]))[0]  # 3 geom nodes
        vc = x[fverts]                                                # (3,3)
        area, nrm = triangle_area_normal(vc[0], vc[1], vc[2])
        if area == 0:
            continue
        # air cell centroid for orientation
        cells = f2c.links(f)
        air_cell = None
        for c in cells:
            if cell_tag[c] == PHYS_AIR_INTERNAL:
                air_cell = c
                break
        if air_cell is None:
            continue
        acv = _entities_to_geometry(mesh, tdim, np.array([air_cell]))[0]
        air_centroid = x[acv].mean(axis=0)
        n_a = orient_outward_from_air(nrm, vc.mean(axis=0), air_centroid)
        Mt = triangle_mass_matrix(area)
        area_assembled += area
        for j in range(3):
            pj = p_by_key.get(_k(vc[j]))
            if pj is None:
                n_missing += 1
                continue
            for i in range(3):
                ublock = u_by_key.get(_k(vc[i]))
                if ublock is None:
                    n_missing += 1
                    continue
                for comp in range(3):
                    rows.append(pj)
                    cols.append(ublock * bs + comp)
                    data.append(n_a[comp] * Mt[j, i])
    if n_missing:
        print(f"[fsi] WARNING: {n_missing} interface facet-node dof lookups missed "
              f"(coord-key mismatch) — G is under-assembled; check units/key resolution.")

    n_p = V_p.dofmap.index_map.size_local * V_p.dofmap.index_map_bs
    n_u = V_u.dofmap.index_map.size_local * bs
    G = sp.csr_matrix((data, (rows, cols)), shape=(n_p, n_u))
    G._assembled_interface_area = area_assembled     # stash for verification
    return G


# ===========================================================================
# Verification
# ===========================================================================

def geometric_fsi_area(mesh, facet_tags):
    """Sum of physical FSI facet areas [m² if mesh in m], computed geometrically
    from the facet vertex coordinates."""
    from air_acoustics import _entities_to_geometry
    fdim = mesh.topology.dim - 1
    x = mesh.geometry.x
    total = 0.0
    for t in FSI_TAGS:
        facets = facet_tags.find(t)
        if not len(facets):
            continue
        geo = _entities_to_geometry(mesh, fdim, facets)
        for row in geo:
            vc = x[row]
            a, _ = triangle_area_normal(vc[0], vc[1], vc[2])
            total += a
    return total


def _cell_tag_array(mesh):
    """Dense cell -> physical tag lookup (0 where untagged)."""
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    return np.zeros(n, dtype=np.int32), tdim, n


def oriented_net_normal(mesh, cell_tags, facet_tags, tags):
    """Geometric ∫ n_a dS over the given facet `tags`, with n_a oriented OUTWARD
    from the adjacent AIR cell.  Returns (net_vector[3], area).

    For an interface (FSI) facet the two adjacent cells are wood+air; for an open
    soundhole facet only the air cell is adjacent.  Either way orient away from
    the air cell centroid.
    """
    from air_acoustics import _entities_to_geometry
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    f2c = mesh.topology.connectivity(fdim, tdim)
    ctag = np.zeros(mesh.topology.index_map(tdim).size_local
                    + mesh.topology.index_map(tdim).num_ghosts, dtype=np.int32)
    ctag[cell_tags.indices] = cell_tags.values
    x = mesh.geometry.x
    net = np.zeros(3)
    area = 0.0
    for t in tags:
        for f in facet_tags.find(t):
            vc = x[_entities_to_geometry(mesh, fdim, np.array([f]))[0]]
            a, nrm = triangle_area_normal(vc[0], vc[1], vc[2])
            if a == 0:
                continue
            cells = f2c.links(f)
            air_cell = next((c for c in cells if ctag[c] == PHYS_AIR_INTERNAL), None)
            if air_cell is None:                 # fall back to the sole adjacent cell
                air_cell = cells[0] if len(cells) else None
            if air_cell is None:
                continue
            acc = x[_entities_to_geometry(mesh, tdim, np.array([air_cell]))[0]].mean(axis=0)
            n_a = orient_outward_from_air(nrm, vc.mean(axis=0), acc)
            net += n_a * a
            area += a
    return net, area


def assembled_fsi_area(mesh, facet_tags):
    """FSI interface area as the FEM interior-facet measure sees it (works for
    ANY G assembler — independent of G).  Integrates 1 over dS(FSI), once per
    facet, via a DG0 unit function's '+' side."""
    import dolfinx.fem as fem
    import ufl
    from mpi4py import MPI
    DG0 = fem.functionspace(mesh, ("DG", 0))
    w = fem.Function(DG0)
    w.x.array[:] = 1.0
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags)
    expr = None
    for tag in FSI_TAGS:
        term = w("+") * dS(tag)
        expr = term if expr is None else expr + term
    form = fem.form(expr)
    local = fem.assemble_scalar(form)
    total = mesh.comm.allreduce(local, op=MPI.SUM)
    # complex FEniCSx build returns a complex scalar: use the real part, but the
    # imaginary part of a real geometric area MUST be ~0 (assert, don't skip).
    im = abs(np.imag(total))
    re = float(np.real(total))
    assert im <= 1e-9 * (abs(re) + 1e-30), \
        f"assembled FSI area has non-negligible imaginary part {im:.3e} (re={re:.3e})"
    return re


def verify_coupling(G, mesh, cell_tags, facet_tags, out_dir=None,
                    assembler=None) -> dict:
    """Diagnostics: shape / nnz / norm, interface area, air-normal net vector.

    - assembled interface area (FEM dS over FSI) vs geometric FSI area — catches
      wrong facet selection / double counting, for BOTH assemblers,
    - net area-normal vector ∫_Γ n_a dS = 1_pᵀ G applied to rigid translations;
      by the divergence theorem this equals -∫_soundhole n_a dS, so its magnitude
      should match the SOUNDHOLE area (0 only for a fully closed cavity).  This is
      the primary check of the interface side / normal-sign convention.
    """
    import numpy as _np
    geo_area = geometric_fsi_area(mesh, facet_tags)
    try:
        asm_area = assembled_fsi_area(mesh, facet_tags)
    except Exception as exc:
        print(f"[fsi] assembled-area FEM check skipped ({exc})")
        asm_area = float(getattr(G, "_assembled_interface_area", float("nan")))

    # 1_pᵀ G e_c = ∫_Γ (Σ_j N_j^p) n_a[c] dS = ∫_Γ n_a[c] dS  (partition of unity).
    n_u = G.shape[1]
    ones_p = _np.ones(G.shape[0])
    net = _np.zeros(3)
    for c in range(3):
        e = _np.zeros(n_u)
        e[c::3] = 1.0
        # complex build: G may be complex -> take the real part
        net[c] = float(_np.real(ones_p @ (G @ e)))

    def _json_number(x):
        x = float(x)
        return x if _np.isfinite(x) else None

    # Closure check: ∫_FSI n_a dS + ∫_SOUNDHOLE n_a dS should ≈ 0 (closed air
    # boundary).  This pins down the FSI-net vs soundhole-area discrepancy: if it
    # closes, |∫_FSI n_a dS| == soundhole projected-area vector exactly; a nonzero
    # residual means the air boundary is not fully tiled by FSI + SOUNDHOLE (gap /
    # overlap / a flipped facet), not a projected-vs-surface-area effect.
    fsi_net_geo, fsi_area_geo = oriented_net_normal(mesh, cell_tags, facet_tags, FSI_TAGS)
    sh_net_geo, sh_area_geo = oriented_net_normal(mesh, cell_tags, facet_tags, (PHYS_SOUNDHOLE,))
    closure = fsi_net_geo + sh_net_geo
    closure_mag = float(_np.linalg.norm(closure))
    # scale the closure residual by the soundhole area for interpretability
    closure_rel = closure_mag / (sh_area_geo + 1e-30)

    custom_area = getattr(G, "_assembled_interface_area", float("nan"))
    diag = {
        "assembler": assembler,
        "G_shape": list(G.shape),
        "G_nnz": int(G.nnz),
        # Frobenius norm from the CSR data directly (avoids relying on
        # scipy.sparse.linalg being imported by `import scipy.sparse as sp`).
        "G_fro_norm": float(np.linalg.norm(np.asarray(G.data))),
        "geometric_fsi_area": float(geo_area),
        "assembled_fsi_area": _json_number(asm_area),
        "custom_assembled_area": _json_number(custom_area),
        "net_area_normal_vector": net.tolist(),
        "net_area_normal_magnitude": float(_np.linalg.norm(net)),
        # soundhole projected-area vector + closure (#1)
        "soundhole_area": float(sh_area_geo),
        "soundhole_net_vector": sh_net_geo.tolist(),
        "soundhole_net_magnitude": float(_np.linalg.norm(sh_net_geo)),
        "fsi_net_vector_geometric": fsi_net_geo.tolist(),
        "closure_vector": closure.tolist(),
        "closure_magnitude": closure_mag,
        "closure_rel_to_soundhole": float(closure_rel),
    }
    if geo_area > 0 and _np.isfinite(asm_area):
        diag["area_rel_error"] = abs(asm_area - geo_area) / geo_area
    print(f"[fsi] assembler={assembler}  G {diag['G_shape']} nnz={diag['G_nnz']} "
          f"||G||_F={diag['G_fro_norm']:.3e}")
    print(f"[fsi] FSI area geometric={geo_area:.4e}  assembled(FEM)={asm_area:.4e}"
          + (f"  rel_err={diag['area_rel_error']:.2e}" if "area_rel_error" in diag else ""))
    print(f"[fsi] net area-normal |∫_FSI n_a dS|(G)={diag['net_area_normal_magnitude']:.3e}  "
          f"|∫_SH n_a dS|={diag['soundhole_net_magnitude']:.3e}  "
          f"soundhole_area={sh_area_geo:.3e}")
    print(f"[fsi] closure |∫_FSI n + ∫_SH n|={closure_mag:.3e} "
          f"({closure_rel*100:.2f}% of soundhole area; ~0 => boundary tiled correctly)")
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "fsi_coupling_diagnostics.json").write_text(
            json.dumps(diag, indent=2))
    return diag


# ===========================================================================
# Orchestration
# ===========================================================================

def _petsc_to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())


def _domain_scalar_dofs(V, cells):
    """Unique GLOBAL scalar dof indices of space V on `cells`, unrolled for
    blocked (vector) spaces so they index the same column/row space as the custom
    G (scalar dof = block*bs + comp)."""
    dofmap = V.dofmap
    bs = dofmap.index_map_bs
    blocks = set()
    for c in cells:
        blocks.update(int(b) for b in dofmap.cell_dofs(int(c)))
    blocks = np.array(sorted(blocks), dtype=np.int64)
    if bs == 1 or len(blocks) == 0:
        return blocks
    return (blocks[:, None] * bs + np.arange(bs)[None, :]).ravel()


def assemble_coupling(msh_path, output_dir="results/fsi", prefer="multiphenicsx"):
    """Load mesh, build spaces, assemble + verify G, and save diagnostics.

    `prefer='custom'` runs ONLY the custom assembler (never touches multiphenicsx);
    `prefer='multiphenicsx'` tries multiphenicsx and falls back to custom on error.

    Also computes the coupled-solver RESTRICTION: the active air-pressure dofs and
    wood-displacement dofs, and a restricted G (air_p x wood_u).  D must solve on
    these active dofs — the full-space G contains inactive air-u / wood-p rows and
    columns that would make the monolithic system singular.

    Returns (G_full, V_u, V_p, diagnostics).  The restricted G and dof index
    arrays are also saved to output_dir for D to load.
    """
    mesh, cell_tags, facet_tags = load_coupled_mesh(msh_path)
    # mesh comes in mm from the .msh; scale to m for physical areas.
    mesh.geometry.x[:] *= 1e-3
    V_u, V_p = build_spaces(mesh)

    if prefer == "custom":
        G = assemble_G_custom(mesh, cell_tags, facet_tags, V_u, V_p)
        used = "custom"
    else:
        try:
            G = assemble_G_multiphenicsx(mesh, cell_tags, facet_tags, V_u, V_p)
            used = "multiphenicsx"
        except Exception as exc:
            print(f"[fsi] multiphenicsx path failed ({type(exc).__name__}: {exc}); "
                  f"falling back to custom assembly.")
            G = assemble_G_custom(mesh, cell_tags, facet_tags, V_u, V_p)
            used = "custom (mpx fallback)"
    print(f"[fsi] G assembled via: {used}")

    diag = verify_coupling(G, mesh, cell_tags, facet_tags, out_dir=None,
                           assembler=used)

    # --- Restriction (#3): active domain dofs + restricted G for D -----------
    tdim = mesh.topology.dim
    wood_cells = cell_tags.find(PHYS_SOLID_WOOD)
    air_cells = cell_tags.find(PHYS_AIR_INTERNAL)
    wood_u_dofs = _domain_scalar_dofs(V_u, wood_cells)
    air_p_dofs = _domain_scalar_dofs(V_p, air_cells)
    # For the custom (full-space) G, restrict to (air_p rows, wood_u cols).  For
    # multiphenicsx, G is already restricted — leave as-is but still record dofs.
    if used.startswith("custom"):
        G_restricted = G.tocsr()[air_p_dofs][:, wood_u_dofs]
    else:
        G_restricted = G
    diag["restriction"] = {
        "n_wood_u_dofs": int(len(wood_u_dofs)),
        "n_air_p_dofs": int(len(air_p_dofs)),
        "G_full_shape": list(G.shape),
        "G_restricted_shape": list(G_restricted.shape),
        "G_restricted_nnz": int(G_restricted.nnz),
    }
    print(f"[fsi] restriction: wood_u_dofs={len(wood_u_dofs)}  "
          f"air_p_dofs={len(air_p_dofs)}  G_restricted={list(G_restricted.shape)} "
          f"nnz={G_restricted.nnz}")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fsi_coupling_diagnostics.json").write_text(json.dumps(diag, indent=2))
        np.savez(str(out / "fsi_restriction.npz"),
                 wood_u_dofs=wood_u_dofs, air_p_dofs=air_p_dofs)
        sp.save_npz(str(out / "G_restricted.npz"), G_restricted.tocsr())
        print(f"[fsi] saved: fsi_coupling_diagnostics.json, fsi_restriction.npz, "
              f"G_restricted.npz -> {out}")

    return G, V_u, V_p, diag


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Assemble + verify FSI coupling matrix G")
    ap.add_argument("--air-msh", type=str, required=True, help="conformal mesh_air.msh")
    ap.add_argument("--output-dir", type=str, default="results/fsi")
    ap.add_argument("--assembler", type=str, default="multiphenicsx",
                    choices=["multiphenicsx", "custom"])
    a = ap.parse_args()
    assemble_coupling(a.air_msh, a.output_dir, prefer=a.assembler)
