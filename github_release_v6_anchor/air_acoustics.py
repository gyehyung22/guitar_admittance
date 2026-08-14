"""
air_acoustics.py
----------------
Internal air-cavity acoustics for the guitar air-coupling branch (3A conformal
multi-domain).  Loads the AIR_INTERNAL sub-domain from the conformal mesh,
assembles the acoustic K_a / M_a, and provides:

  * air-only eigenmode diagnostics  (closed = rigid walls, pressure_release = p=0
    on the soundhole) — LINEAR generalized eigenproblems K_a φ = λ M_a φ,
    f_n = c√λ / 2π.  These are DIAGNOSTICS; neither reproduces A0.
  * the A0 / Helmholtz resonance via the impedance-PORT harmonic response
    (rank-1 nonlocal port, Sherman–Morrison) — this is what reproduces A0.

Time convention: exp(+iωt).  All coefficients/signs follow
docs/air_coupling_theory.md (Everstine u–p form; soundhole rank-1 port).

FEniCSx imports are deferred into functions so this module imports for static
checks even where dolfinx is absent (e.g. Windows).  The FEM itself runs on the
server (conda env fenicsx).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from acoustic_helmholtz import (
    AIR_C, AIR_RHO0, helmholtz_estimate, acoustic_inertance,
    port_impedance, radiation_resistance,
)

# Physical ids must match mesh_gen.py
PHYS_AIR_INTERNAL = 2
PHYS_SOUNDHOLE = 14


def _facet_vertex_keys(coords_mm):
    """Canonical, order-independent identity key per facet from its 3 vertex
    coordinates [mm].  Coordinates are rounded to 1 micron for float hashing;
    since distinct facets are >= ~0.5 mm apart this is EXACT facet identity, not a
    spatial proximity tolerance.  Two coincident (conformal) facets — the physical
    SOUNDHOLE triangle and its copy on the air sub-mesh boundary — hash to the
    same key, so this propagates the physical tag exactly.
    """
    keys = []
    for tri in np.round(np.asarray(coords_mm) * 1000.0).astype(np.int64):  # -> micron ints
        keys.append(tuple(sorted(tuple(v) for v in tri)))
    return keys


# ---------------------------------------------------------------------------
# Air sub-mesh extraction
# ---------------------------------------------------------------------------

def load_air_mesh(msh_path: Path):
    """Extract the AIR_INTERNAL tetrahedra as a standalone dolfinx mesh.

    Returns (dolfin_mesh, soundhole_coords [N,3] mm, soundhole_facet_keys, info).
    `soundhole_facet_keys` is the set of canonical vertex-coordinate keys of the
    physical SOUNDHOLE facets (used for EXACT facet-tag propagation onto the air
    sub-mesh boundary — no proximity tolerance).  `soundhole_coords` are the
    SOUNDHOLE node coordinates (used only for the pressure_release Dirichlet BC).
    """
    import meshio
    from mpi4py import MPI
    from dolfinx.io import XDMFFile

    mio = meshio.read(str(msh_path))
    phys = mio.cell_data.get("gmsh:physical", [])
    tet = tet_tags = tri = tri_tags = None
    for cb, tg in zip(mio.cells, phys):
        if cb.type == "tetra":
            tet, tet_tags = cb.data, np.asarray(tg)
        elif cb.type == "triangle":
            tri, tri_tags = cb.data, np.asarray(tg)
    if tet is None:
        raise RuntimeError("No tetrahedra in air mesh")

    air_mask = (tet_tags == PHYS_AIR_INTERNAL)
    if not air_mask.any():
        raise RuntimeError(f"No AIR_INTERNAL (physical={PHYS_AIR_INTERNAL}) cells found")
    air_tets = tet[air_mask]
    used = np.unique(air_tets)
    remap = -np.ones(int(used.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    air_points = mio.points[used]
    air_tets_c = remap[air_tets]

    # Propagate the physical SOUNDHOLE facets from the conformal mesh by EXACT
    # facet identity (vertex-coordinate keys), NOT by proximity tolerance.
    sh_coords = np.zeros((0, 3))
    sh_facet_keys: set = set()
    if tri is not None and tri_tags is not None:
        sh_tris = tri[tri_tags == PHYS_SOUNDHOLE]
        if len(sh_tris):
            sh_nodes = np.unique(sh_tris)
            sh_nodes = sh_nodes[np.isin(sh_nodes, used)]
            sh_coords = mio.points[sh_nodes]
            sh_facet_keys = set(_facet_vertex_keys(mio.points[sh_tris]))  # (n,3,3) mm

    import tempfile
    with tempfile.TemporaryDirectory(prefix="air_tmp_") as td:
        tmp = Path(td) / "air_tmp.xdmf"
        meshio.write(str(tmp), meshio.Mesh(points=air_points, cells=[("tetra", air_tets_c)]))
        with XDMFFile(MPI.COMM_WORLD, str(tmp), "r") as f:
            dolfin_mesh = f.read_mesh(name="Grid")

    info = {
        "n_air_nodes": int(len(air_points)),
        "n_air_tets": int(len(air_tets_c)),
        "n_soundhole_nodes": int(len(sh_coords)),
        "n_soundhole_facets_mesh": int(len(sh_facet_keys)),
    }
    print(f"[air] AIR_INTERNAL: {info['n_air_nodes']} nodes, "
          f"{info['n_air_tets']} tets, {info['n_soundhole_nodes']} soundhole nodes, "
          f"{info['n_soundhole_facets_mesh']} soundhole facets (physical tag)")
    return dolfin_mesh, sh_coords, sh_facet_keys, info


# ---------------------------------------------------------------------------
# Acoustic assembly
# ---------------------------------------------------------------------------

def assemble_acoustic(dolfin_mesh):
    """Assemble K_a = ∫∇p·∇q, M_a = ∫ p q over the air domain (mm -> m).

    Returns (K_csr, M_csr, V) with V the scalar P1 function space.
    """
    import dolfinx.fem as fem
    import dolfinx.fem.petsc as petsc_fem
    import ufl

    dolfin_mesh.geometry.x[:] *= 1e-3          # mm -> m

    V = fem.functionspace(dolfin_mesh, ("Lagrange", 1))
    p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    K_form = fem.form(ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx)
    M_form = fem.form(ufl.inner(p, q) * ufl.dx)

    K = petsc_fem.assemble_matrix(K_form); K.assemble()
    M = petsc_fem.assemble_matrix(M_form); M.assemble()

    def to_scipy(A):
        ai, aj, av = A.getValuesCSR()
        return sp.csr_matrix((av, aj, ai), shape=A.getSize())

    K_s, M_s = to_scipy(K), to_scipy(M)
    dolfin_mesh.geometry.x[:] *= 1e3           # restore mm
    print(f"[air] K_a nnz={K_s.nnz}, M_a nnz={M_s.nnz}, ndofs={K_s.shape[0]}")
    return K_s, M_s, V


def _entities_to_geometry(mesh, dim, entities):
    """dolfinx.mesh.entities_to_geometry across 0.10 signature variants."""
    import dolfinx.mesh as dmesh
    try:
        return dmesh.entities_to_geometry(mesh, dim, entities, False)
    except TypeError:
        return dmesh.entities_to_geometry(mesh, dim, entities)


def soundhole_port_vector(dolfin_mesh, V, soundhole_facet_keys):
    """Boundary integral vector b_j = ∫_{Γ_h} N_j dS  (so S = Σ b_j).

    Soundhole facets are selected by EXACT physical-tag identity: each air-submesh
    boundary facet's canonical vertex-coordinate key is looked up in the set of
    physical SOUNDHOLE facet keys.  No proximity tolerance is used, so coplanar
    rim FSI_TOP facets are never grabbed regardless of mesh fineness.

    Returns (b [ndofs], S, n_soundhole_facets).
    """
    import dolfinx.fem as fem
    import dolfinx.mesh as dmesh
    import ufl

    if not soundhole_facet_keys:
        ndofs = V.dofmap.index_map.size_global
        return np.zeros(ndofs), 0.0, 0

    n_expected = len(soundhole_facet_keys)
    fdim = dolfin_mesh.topology.dim - 1
    dolfin_mesh.topology.create_connectivity(fdim, dolfin_mesh.topology.dim)
    bfacets = dmesh.exterior_facet_indices(dolfin_mesh.topology)

    # Facet vertex coords in mm (mesh native) -> canonical keys -> exact match.
    geo = _entities_to_geometry(dolfin_mesh, fdim, bfacets)      # (n,3) geom node ids
    fcoords = dolfin_mesh.geometry.x[geo]                        # (n,3,3) mm
    keys = _facet_vertex_keys(fcoords)
    mask = np.array([k in soundhole_facet_keys for k in keys], dtype=bool)
    sel = np.sort(bfacets[mask]).astype(np.int32)
    print(f"[air] soundhole facet match (physical tag, exact): {len(sel)} selected "
          f"vs {n_expected} SOUNDHOLE facets")

    markers = np.ones(len(sel), dtype=np.int32)
    mt = dmesh.meshtags(dolfin_mesh, fdim, sel, markers)
    ds = ufl.Measure("ds", domain=dolfin_mesh, subdomain_data=mt)

    dolfin_mesh.geometry.x[:] *= 1e-3                       # mm -> m for the integral
    q = ufl.TestFunction(V)
    # Complex FEniCSx build requires the test function to be conjugated in a
    # linear form; ufl.conj is a no-op on the real shape functions so the
    # geometric integral b_j = ∫_{Γ_h} N_j dS is unchanged.
    b_form = fem.form(ufl.conj(q) * ds(1))
    b_vec = fem.petsc.assemble_vector(b_form)
    b_vec.assemble()
    b = b_vec.getArray().real.copy()
    S = float(b.sum())
    dolfin_mesh.geometry.x[:] *= 1e3                        # restore mm
    print(f"[air] soundhole port: {len(sel)} facets, area S={S*1e4:.3f} cm^2")
    return b, S, int(len(sel))


# ---------------------------------------------------------------------------
# Air-only eigenmodes (diagnostics)
# ---------------------------------------------------------------------------

def air_eigenmodes(K_csr, M_csr, n_modes=20, c=AIR_C,
                   dirichlet_dofs=None, f_target=120.0):
    """Generalized eigenproblem K_a φ = λ M_a φ, λ = (ω/c)², f_n = c√λ/2π.

    dirichlet_dofs: if given (pressure_release), those dofs are fixed p=0 by
    deleting their rows/cols.  closed BC passes dirichlet_dofs=None.
    Returns sorted eigenfrequencies [Hz] (drops the ~0 rigid/constant mode).
    """
    from slepc4py import SLEPc
    from petsc4py import PETSc

    K = K_csr.tocsr().astype(complex)
    M = M_csr.tocsr().astype(complex)
    n = K.shape[0]
    keep = np.ones(n, dtype=bool)
    if dirichlet_dofs is not None and len(dirichlet_dofs):
        keep[np.asarray(dirichlet_dofs, dtype=int)] = False
        K = K[keep][:, keep]
        M = M[keep][:, keep]

    def to_petsc(A):
        A = A.tocsr()
        return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))

    Kp, Mp = to_petsc(K), to_petsc(M)
    Kp.assemble(); Mp.assemble()

    eps = SLEPc.EPS().create()
    eps.setOperators(Kp, Mp)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(n_modes)
    sigma = (2.0 * np.pi * f_target / c) ** 2
    eps.setTarget(sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    st = eps.getST(); st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP(); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu")
    for s in ("mumps", "superlu_dist", "superlu"):
        try:
            pc.setFactorSolverType(s); break
        except Exception:
            pass
    eps.solve()
    nconv = eps.getConverged()
    freqs = []
    for i in range(nconv):
        lam = eps.getEigenvalue(i).real
        if lam > 1e-6:                         # drop the ~0 constant-pressure mode
            freqs.append(c * np.sqrt(lam) / (2.0 * np.pi))
    eps.destroy()
    return np.array(sorted(freqs))


# ---------------------------------------------------------------------------
# A0 via impedance-port harmonic response (Sherman–Morrison)
# ---------------------------------------------------------------------------

def a0_harmonic_port(K_csr, M_csr, b, S, cavity_volume, soundhole_area, t_hole,
                     freqs_hz, c=AIR_C, rho0=AIR_RHO0, include_radiation=True,
                     port_end_corrections=1):
    """Average cavity pressure response with the nonlocal rank-1 impedance port.

        A(ω) = K_a − (ω²/c²) M_a + κ(ω) b bᵀ ,  κ = iωρ0/(S² Z_h)
        Z_h  = R_rad + iω M_h ,  M_h = ρ0 L_eff_port / S

    Solved by Sherman–Morrison (never forms b bᵀ).  Source = uniform volume
    source f = M_a · 1.  Returns (p_bar(ω) complex array, f_A0_observed).

    `port_end_corrections`: number of 0.85·r_eff end corrections added to the
    NECK length for the LUMPED port.  Default 1 (exterior only): the INTERIOR
    near-hole added mass is already represented by the 3D FE cavity, so adding
    the interior correction here double-counts it and pushes A0 low.  (The
    analytic f_H still uses 2 corrections — that is the standalone lumped model.)
    """
    import scipy.sparse.linalg as spla
    from acoustic_helmholtz import effective_neck_length

    K = K_csr.tocsc().astype(complex)
    M = M_csr.tocsc().astype(complex)
    b = np.asarray(b, dtype=complex)
    f_src = (M @ np.ones(M.shape[0])).astype(complex)

    # Port neck length: physical thickness + EXTERIOR end correction only.
    L_eff = effective_neck_length(t_hole, soundhole_area, n_open_ends=port_end_corrections)
    M_h = acoustic_inertance(L_eff, soundhole_area, rho0)

    p_bar = np.zeros(len(freqs_hz), dtype=complex)
    for i, f in enumerate(freqs_hz):
        omega = 2.0 * np.pi * f
        A0 = (K - (omega ** 2 / c ** 2) * M).tocsc()
        R = radiation_resistance(omega, soundhole_area, c, rho0) if include_radiation else 0.0
        Z_h = R + 1j * omega * M_h
        kappa = 1j * omega * rho0 / (S ** 2 * Z_h)
        lu = spla.splu(A0)
        x = lu.solve(f_src)                    # A0^-1 f
        y = lu.solve(b)                        # A0^-1 b
        denom = 1.0 + kappa * (b @ y)
        p = x - kappa * y * (b @ x) / denom    # Sherman–Morrison
        p_bar[i] = (b @ p) / S
    f_a0 = float(freqs_hz[np.argmax(np.abs(p_bar))])
    return p_bar, f_a0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_air_diagnostics(air_msh_path=None, air_step=None, output_dir="results/air",
                        soundhole_bc="impedance", n_modes=20,
                        a0_freqs=None, c=AIR_C, rho0=AIR_RHO0,
                        t_hole_mm=None, conformal_mesh_builder=None):
    """Run air-cavity diagnostics for one hollow sample.

    Requires a conformal air mesh (mesh_air.msh).  If only `air_step` (+ a wood
    step) is available, the caller should build the mesh first
    (mesh_gen.generate_conformal_air_mesh); this function focuses on the acoustics.

    Saves: air_eigenfrequencies_hz / air_modes.npz / air_metadata.json /
           helmholtz_estimate.json (and A0 sweep if impedance).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if air_msh_path is None:
        raise ValueError("run_air_diagnostics needs a conformal air mesh (mesh_air.msh)")

    # Read mesh-level metadata (cavity volume / soundhole / t_hole) if present.
    meta_file = Path(air_msh_path).parent / "air_mesh_meta.json"
    mesh_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    if t_hole_mm is None:
        # neck length L_eff = t_hole + 2*0.85 r_eff → recover t_hole if available
        t_hole_mm = mesh_meta.get("soundhole_thickness_mm", None)

    dolfin_mesh, sh_coords, sh_facet_keys, info = load_air_mesh(Path(air_msh_path))
    K, M, V = assemble_acoustic(dolfin_mesh)
    b, S, n_sh = soundhole_port_vector(dolfin_mesh, V, sh_facet_keys)

    # Fail-loud sanity: the port area must match the mesh metadata soundhole area.
    S_meta = mesh_meta.get("soundhole_area_m2")
    if S_meta and S_meta > 0:
        rel = abs(S - S_meta) / S_meta
        print(f"[air] soundhole area check: port {S*1e4:.3f} cm^2 vs mesh "
              f"{S_meta*1e4:.3f} cm^2 ({rel*100:.2f}%)")
        if rel > 0.02:
            raise RuntimeError(
                f"Soundhole port area {S*1e4:.3f} cm^2 disagrees with mesh "
                f"metadata {S_meta*1e4:.3f} cm^2 by {rel*100:.1f}% (> 2%). "
                f"Exact facet-tag selection mis-matched — check SOUNDHOLE physical "
                f"tag / mesh conformality.")
    if n_sh != info.get("n_soundhole_facets_mesh", n_sh):
        print(f"[air] WARNING: matched {n_sh} facets but mesh has "
              f"{info.get('n_soundhole_facets_mesh')} SOUNDHOLE facets — "
              f"possible node-reindex or non-conformal boundary.")

    # Geometry quantities from the mesh (authoritative).
    cavity_volume = mesh_meta.get("cavity_volume_m3")
    if cavity_volume is None:
        # fall back: ∫ 1 dΩ = ones^T M_a ones
        cavity_volume = float(np.ones(M.shape[0]) @ (M @ np.ones(M.shape[0])))
    soundhole_area = S if S > 0 else mesh_meta.get("soundhole_area_m2", 0.0)
    t_hole = (t_hole_mm * 1e-3) if t_hole_mm else \
        (mesh_meta.get("top_plate_thickness_mm", 3.0) * 1e-3)

    est = helmholtz_estimate(cavity_volume, soundhole_area, t_hole, c, rho0)
    (output_dir / "helmholtz_estimate.json").write_text(json.dumps(est.to_dict(), indent=2))

    out = {"soundhole_bc": soundhole_bc, "info": info,
           "helmholtz": est.to_dict(), "n_soundhole_facets": n_sh}

    # --- Eigenmode diagnostics (closed / pressure_release) ------------------
    if soundhole_bc in ("closed", "pressure_release"):
        dofs = None
        if soundhole_bc == "pressure_release":
            # dofs near soundhole nodes (reuse the port marking coords)
            dofs = _soundhole_dofs(dolfin_mesh, V, sh_coords)
        eig = air_eigenmodes(K, M, n_modes=n_modes, c=c, dirichlet_dofs=dofs,
                             f_target=max(est.estimated_helmholtz_hz, 50.0))
        np.savez(str(output_dir / "air_modes.npz"), air_eigenfrequencies_hz=eig,
                 soundhole_bc=soundhole_bc)
        out["air_eigenfrequencies_hz"] = eig.tolist()
        print(f"[air] {soundhole_bc} eigenfreqs (first 8): {np.round(eig[:8], 1)}")
        print("[air] NOTE: neither closed nor pressure_release reproduces A0 — "
              "A0 needs the impedance port harmonic response.")

    # --- A0 via impedance port harmonic ------------------------------------
    if soundhole_bc in ("impedance", "throat"):
        if a0_freqs is None:
            f0 = est.estimated_helmholtz_hz or 150.0
            a0_freqs = np.linspace(max(20.0, 0.3 * f0), 2.5 * f0, 600)
        p_bar, f_a0 = a0_harmonic_port(K, M, b, S, cavity_volume, soundhole_area,
                                       t_hole, a0_freqs, c, rho0)
        np.savez(str(output_dir / "a0_port_response.npz"),
                 freqs_hz=a0_freqs, p_bar=p_bar, f_a0_observed=f_a0)
        rel = abs(f_a0 - est.estimated_helmholtz_hz) / max(est.estimated_helmholtz_hz, 1e-9)
        out["A0_observed_hz"] = f_a0
        out["A0_estimated_hz"] = est.estimated_helmholtz_hz
        out["A0_rel_error"] = rel
        (output_dir / "A0_estimated_vs_observed.json").write_text(json.dumps({
            "A0_estimated_hz": est.estimated_helmholtz_hz,
            "A0_observed_hz": f_a0, "rel_error": rel,
            "soundhole_bc": soundhole_bc,
            "port_end_corrections": 1,
            "note": ("Port uses EXTERIOR end correction only (1x0.85 r_eff); the "
                     "interior near-hole added mass is supplied by the 3D FE "
                     "cavity. The analytic A0_estimated uses the standalone 2-EC "
                     "lumped formula, which over-counts the interior EC for a "
                     "bounded cavity, so a few-percent FE-vs-analytic gap is "
                     "expected and shrinks for deeper-neck/larger cavities."),
        }, indent=2))
        print(f"[air] A0: estimated {est.estimated_helmholtz_hz:.1f} Hz, "
              f"observed {f_a0:.1f} Hz ({rel*100:.2f}% error)")

    (output_dir / "air_metadata.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def _soundhole_dofs(dolfin_mesh, V, soundhole_coords_mm, tol_mm=2.0):
    """Locate scalar dofs near the soundhole node cloud (for pressure_release BC)."""
    import dolfinx.fem as fem
    from scipy.spatial import cKDTree
    if len(soundhole_coords_mm) == 0:
        return np.array([], dtype=np.int32)
    dofs_coords = V.tabulate_dof_coordinates()       # in metres (mesh scaled in assemble)
    # mesh is in mm here (assemble restored it); compare in mm
    tree = cKDTree(np.asarray(soundhole_coords_mm))
    d, _ = tree.query(dofs_coords)
    return np.where(d < tol_mm)[0].astype(np.int32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Internal air-cavity acoustics diagnostics")
    ap.add_argument("--air-msh", type=str, required=True, help="conformal mesh_air.msh")
    ap.add_argument("--soundhole-bc", type=str, default="impedance",
                    choices=["impedance", "throat", "pressure_release", "closed"])
    ap.add_argument("--n-modes", type=int, default=20)
    ap.add_argument("--t-hole-mm", type=float, default=None)
    ap.add_argument("--output-dir", type=str, default="results/air")
    a = ap.parse_args()
    run_air_diagnostics(air_msh_path=a.air_msh, output_dir=a.output_dir,
                        soundhole_bc=a.soundhole_bc, n_modes=a.n_modes,
                        t_hole_mm=a.t_hole_mm)
