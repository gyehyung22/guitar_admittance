"""
fenics_admittance.py
--------------------
FEniCSx(dolfinx) guitar bridge admittance Y(omega) 계산.

Standalone use:
    python fenics_admittance.py                     # legacy: uses hardcoded config
    python fenics_admittance.py --test              # box mesh validation

Subprocess use (from run_pipeline.py):
    python fenics_admittance.py --msh /path/mesh.msh --material engelmann_spruce ...

Library use:
    from fenics_admittance import compute_admittance
"""
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception as _mpl_err:
    _MPL_OK = False
    print(f"[warn] matplotlib unavailable: {_mpl_err}")

import sys
import json
import tempfile
import time
import numpy as np
from pathlib import Path
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _load_json_arg(value):
    """Load a CLI JSON argument that may be either a JSON string or a file path."""
    text = str(value).strip()
    if text.startswith(("[", "{")):
        return json.loads(text)
    try:
        p = Path(text)
        if p.exists():
            return json.loads(p.read_text())
    except OSError:
        pass
    return json.loads(text)

# ---------------------------------------------------------------------------
# Legacy configuration (used by main() only)
# ---------------------------------------------------------------------------
_MESH_FILE    = Path(__file__).parent / "mesh" / "guitar.msh"
_RESULTS_DIR  = Path(__file__).parent / "results"
_FREQ_MIN     = 20.0
_FREQ_MAX     = 5000.0
_FREQ_POINTS  = 500
_BRIDGE_COORDS = np.array([0.0, -225.0, 100.0])
_FORCE_N      = 1.0

# Target rigid-body suspension frequency [Hz].  The diagonal "weak spring" used
# to remove the 6 free-free rigid-body modes is sized PER MESH so that the rigid
# modes land at ~this frequency, well below freq_min (20 Hz) and out of the band
# of interest.  Using a fixed absolute stiffness instead makes the rigid-mode
# frequency scale as sqrt(N_nodes) and pollutes the response (see code review).
_RIGID_FREQ_HZ = 1.0
SOLID_FULL_PEAK_SOLVER_REVISION = "structural-full-harmonic-peaks-v1"

_MATERIAL = {
    "E1": 9.79e9,  "E2": 1.25e9,  "E3": 0.58e9,
    "G12": 1.21e9, "G13": 1.17e9, "G23": 0.10e9,
    "nu12": 0.422, "nu13": 0.462, "nu23": 0.53,
    "density": 350.0,
    "alpha": 0.0,
    "beta":  5.0e-6,
}


# ---------------------------------------------------------------------------
# Orthotropic stiffness matrix
# ---------------------------------------------------------------------------

def engineering_to_C(mat: dict, orient_grain_along_y: bool = True) -> np.ndarray:
    """Engineering constants -> 6x6 Voigt stiffness matrix [Pa] in the GLOBAL frame.

    Material axes: 1=L (longitudinal/grain), 2=R (radial), 3=T (tangential),
    with Voigt ordering [LL, RR, TT, RT, LT, LR].

    Guitar geometry convention (guitar_shapes.py): global X = body width
    (treble–bass), Y = neck→tail (length), Z = thickness (out-of-plane).  Wood
    grain L runs along the neck→tail direction, so material axis 1 (L) must align
    with global Y — NOT global X.  Mapping the material frame straight onto the
    mesh axes (the previous behaviour) put the stiff grain across the width,
    rotating the in-plane orthotropy by 90°.

    With `orient_grain_along_y` we rotate the material frame 90° about Z so that
    L→Y, R→X, T→Z.  A 90° rotation about a principal axis keeps the material
    orthotropic, so this is exactly the Voigt index permutation [1,0,2,4,3,5]
    (swap normal X↔Y, swap shear yz↔xz; xy and zz unchanged).
    """
    E1, E2, E3 = mat["E1"], mat["E2"], mat["E3"]
    G12, G13, G23 = mat["G12"], mat["G13"], mat["G23"]
    nu12, nu13, nu23 = mat["nu12"], mat["nu13"], mat["nu23"]

    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    C = np.linalg.inv(S)

    if orient_grain_along_y:
        perm = [1, 0, 2, 4, 3, 5]          # rotate 90° about Z: L→Y, R→X, T→Z
        C = C[np.ix_(perm, perm)]
    return C


# ---------------------------------------------------------------------------
# Rigid-body weak spring (mesh-independent)
# ---------------------------------------------------------------------------

def _total_mass(M) -> float:
    """Total physical mass [kg] = vᵀ M v with v = 1 on z-DOFs (block size 3).

    Works for both a scipy CSR matrix and a PETSc Mat.
    """
    n = M.getSize()[0] if hasattr(M, "getSize") else M.shape[0]
    v = np.zeros(n)
    v[2::3] = 1.0
    if hasattr(M, "getSize"):          # PETSc Mat
        vp = M.createVecRight(); vp.setArray(v); vp.assemblyBegin(); vp.assemblyEnd()
        Mv = M.createVecLeft();  M.mult(vp, Mv)
        m  = float(vp.dot(Mv).real)
        vp.destroy(); Mv.destroy()
        return m
    return float(v @ (M @ v))          # scipy sparse


def weak_spring_for_rigid_freq(M, rigid_freq_hz: float = _RIGID_FREQ_HZ) -> float:
    """Per-DOF diagonal spring stiffness [N/m] placing the rigid-body translation
    modes at ~`rigid_freq_hz`, independent of mesh resolution.

    A spring k on every DOF grounds each node; for a rigid translation all
    N_nodes move together, giving generalized stiffness k·N_nodes against
    generalized mass M_total, so f = √(k·N_nodes / M_total) / 2π.  Inverting:

        k = (2π·f)² · M_total / N_nodes
    """
    n_nodes = (M.getSize()[0] if hasattr(M, "getSize") else M.shape[0]) // 3
    m_total = _total_mass(M)
    k = (2.0 * np.pi * rigid_freq_hz) ** 2 * m_total / max(n_nodes, 1)
    print(f"  Rigid-body spring: target {rigid_freq_hz:.3g} Hz -> "
          f"k={k:.4g} N/m  (M_total={m_total:.4f} kg, N_nodes={n_nodes})")
    return k


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(msh_file: Path, bridge_coords: np.ndarray):
    """
    MSH 2.2 -> dolfinx mesh.

    Parameters
    ----------
    msh_file      : path to .msh file (POSIX path, accessible from WSL)
    bridge_coords : [x, y, z] in mm — nominal position; actual position is
                    read from Physical Point tag=100 embedded by Gmsh.

    Returns
    -------
    (dolfin_mesh, bridge_node_index)
    """
    import meshio
    from mpi4py import MPI

    print(f"Loading mesh: {msh_file}")

    mio = meshio.read(str(msh_file))

    tet_cells = None
    for cell_block in mio.cells:
        if cell_block.type == "tetra":
            tet_cells = cell_block.data
            break
    if tet_cells is None:
        raise RuntimeError("No tetrahedral cells found in mesh")

    points = mio.points
    print(f"  Nodes: {len(points)}, Tets: {len(tet_cells)}")

    # Find bridge node.  Physical Point tag=100 (embedded by Gmsh at bridge_pts[0])
    # gives an exact mesh node, but only when bridge_coords matches that point.
    # For bridge_pts[1] (different coords), fall back to nearest-node coordinate
    # search so each bridge location is measured correctly.
    search_coords = bridge_coords.copy()
    phys_tags = mio.cell_data.get("gmsh:physical", [])
    for cb, tags in zip(mio.cells, phys_tags):
        if cb.type == "vertex":
            mask = (np.asarray(tags) == 100)
            if mask.any():
                msh_node_idx = int(cb.data[mask].flatten()[0])
                embedded_coords = mio.points[msh_node_idx]
                snap_dist = float(np.linalg.norm(embedded_coords - bridge_coords))
                print(f"  Bridge (tag=100): msh_idx={msh_node_idx}, "
                      f"coords={embedded_coords.round(2)}, "
                      f"snap={snap_dist:.2f} mm from nominal")
                if snap_dist < 5.0:
                    search_coords = embedded_coords
                break
    else:
        print("  WARNING: Physical Point tag=100 not found in .msh; "
              "falling back to coordinate search")

    # Each dataset shard runs an independent solver process.  A fixed /tmp XDMF
    # basename lets concurrent solid cases overwrite both the .xdmf and its .h5
    # sidecar, potentially loading another shape's mesh.  Keep the pair in a
    # process-unique temporary directory until dolfinx has read it.
    from dolfinx.io import XDMFFile
    with tempfile.TemporaryDirectory(prefix="guitar_solid_") as tmp_dir:
        tmp_xdmf = Path(tmp_dir) / "mesh.xdmf"
        meshio.write(
            str(tmp_xdmf),
            meshio.Mesh(points=points, cells=[("tetra", tet_cells)]),
        )
        with XDMFFile(MPI.COMM_WORLD, str(tmp_xdmf), "r") as f:
            dolfin_mesh = f.read_mesh(name="Grid")

    print(f"  dolfinx mesh: {dolfin_mesh.topology.index_map(0).size_global} nodes")

    coords = dolfin_mesh.geometry.x
    dist2  = np.sum((coords - search_coords) ** 2, axis=1)
    bridge_idx  = int(np.argmin(dist2))
    bridge_dist = dist2[bridge_idx] ** 0.5
    print(f"  Bridge node: idx={bridge_idx}, dist={bridge_dist:.4f} mm, "
          f"coords={coords[bridge_idx].round(3)}")
    if bridge_dist > 5.0:
        print(f"  WARNING: bridge snapped {bridge_dist:.1f} mm to nearest node "
              f"(nominal coords not on surface). CAD pipeline will supply exact coords.")

    return dolfin_mesh, bridge_idx


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------

def _build_KM_forms(dolfin_mesh, C_pa: np.ndarray, density: float):
    """Shared UFL form builder (mm->m conversion applied in-place)."""
    import dolfinx.fem as fem
    import ufl
    from petsc4py import PETSc

    dolfin_mesh.geometry.x[:] *= 1e-3   # mm -> m

    V = fem.functionspace(dolfin_mesh, ("Lagrange", 1, (3,)))
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"  DOFs: {n_dof}")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def eps_voigt(w):
        e = ufl.sym(ufl.grad(w))
        return ufl.as_vector([
            e[0, 0], e[1, 1], e[2, 2],
            2*e[1, 2], 2*e[0, 2], 2*e[0, 1],
        ])

    C_ufl = fem.Constant(dolfin_mesh, PETSc.ScalarType(C_pa))
    rho   = fem.Constant(dolfin_mesh, PETSc.ScalarType(density))

    K_form = fem.form(ufl.inner(ufl.dot(C_ufl, eps_voigt(u)), eps_voigt(v)) * ufl.dx)
    M_form = fem.form(rho * ufl.inner(u, v) * ufl.dx)
    return K_form, M_form, V


def assemble_KM(dolfin_mesh, C_pa: np.ndarray, density: float):
    """Assemble K and M as scipy CSR (legacy scipy solver path)."""
    import dolfinx.fem.petsc as petsc_fem

    print("Assembling K and M (scipy) ...")
    K_form, M_form, V = _build_KM_forms(dolfin_mesh, C_pa, density)

    K_petsc = petsc_fem.assemble_matrix(K_form); K_petsc.assemble()
    M_petsc = petsc_fem.assemble_matrix(M_form); M_petsc.assemble()

    def to_scipy(A):
        ai, aj, av = A.getValuesCSR()
        return sp.csr_matrix((av, aj, ai), shape=A.getSize())

    K = to_scipy(K_petsc)
    M = to_scipy(M_petsc)

    print(f"  K nnz={K.nnz}, M nnz={M.nnz}")
    print(f"  K diag: {K.diagonal().min():.3e} ~ {K.diagonal().max():.3e}")
    print(f"  M diag: {M.diagonal().min():.3e} ~ {M.diagonal().max():.3e}")

    dolfin_mesh.geometry.x[:] *= 1e3
    K_petsc.destroy(); M_petsc.destroy()
    return K, M, V


def assemble_KM_petsc(dolfin_mesh, C_pa: np.ndarray, density: float):
    """Assemble K and M as complex PETSc matrices (MUMPS solver path)."""
    import dolfinx.fem.petsc as petsc_fem

    print("Assembling K and M (PETSc complex) ...")
    K_form, M_form, V = _build_KM_forms(dolfin_mesh, C_pa, density)

    K_petsc = petsc_fem.assemble_matrix(K_form); K_petsc.assemble()
    M_petsc = petsc_fem.assemble_matrix(M_form); M_petsc.assemble()

    ai, aj, av = K_petsc.getValuesCSR()
    k_diag = K_petsc.getDiagonal().getArray().real
    m_diag = M_petsc.getDiagonal().getArray().real
    print(f"  K nnz={len(av)}, M nnz={len(av)}")
    print(f"  K diag: {k_diag.min():.3e} ~ {k_diag.max():.3e}")
    print(f"  M diag: {m_diag.min():.3e} ~ {m_diag.max():.3e}")

    dolfin_mesh.geometry.x[:] *= 1e3
    return K_petsc, M_petsc, V


# ---------------------------------------------------------------------------
# DOF utilities
# ---------------------------------------------------------------------------

def bridge_dof_z(V, bridge_node_idx: int) -> int:
    """Global DOF index for Z-displacement at bridge node."""
    bs = V.dofmap.index_map_bs   # block size = 3
    return bridge_node_idx * bs + 2


# ---------------------------------------------------------------------------
# Harmonic solve — scipy (legacy)
# ---------------------------------------------------------------------------

def solve_harmonic(K, M, alpha, beta, K_weak, force_vec, freq):
    """Solve one frequency with scipy spsolve (full LU per call)."""
    omega  = 2.0 * np.pi * freq
    C_damp = alpha * M + beta * K
    A = (K + K_weak).astype(complex) - omega**2 * M + 1j * omega * C_damp
    return spla.spsolve(A, force_vec)


# ---------------------------------------------------------------------------
# Harmonic solve — PETSc/MUMPS (fast path)
# ---------------------------------------------------------------------------

def solve_harmonic_petsc(K_petsc, M_petsc, dof_z: int, alpha: float, beta: float,
                          rigid_freq_hz: float, freqs: np.ndarray,
                          force_n: float = 1.0) -> np.ndarray:
    """
    Frequency sweep using PETSc complex matrices + MUMPS direct solver.

    MUMPS performs symbolic analysis only on the first frequency and reuses
    it for all subsequent frequencies (same sparsity pattern) — numeric
    re-factorization only, ~3–10× faster than scipy spsolve per frequency.

    A(ω) = K·(1 + jω·β) + K_weak + M·(−ω² + jω·α)

    `rigid_freq_hz` sets the target rigid-body suspension frequency; the diagonal
    spring is sized per mesh (mesh-independent) so the 6 rigid modes sit there.
    """
    from petsc4py import PETSc

    # Mesh-independent weak spring: place rigid-body modes at rigid_freq_hz
    weak_spring_k = weak_spring_for_rigid_freq(M_petsc, rigid_freq_hz)

    # Pre-compute K + K_weak (tiny diagonal; frequency-independent)
    K_ws = K_petsc.copy()
    kd = K_ws.getDiagonal()
    kd.array += weak_spring_k
    K_ws.setDiagonal(kd)
    kd.destroy()
    K_ws.assemble()

    # Force vector
    F = K_ws.createVecRight()
    F.zeroEntries()
    F.setValue(dof_z, complex(force_n))
    F.assemblyBegin(); F.assemblyEnd()

    # Working matrix pre-allocated once
    A = K_ws.copy()

    # KSP: MUMPS direct solver (symbolic factorization cached across calls)
    ksp = PETSc.KSP().create()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    for solver in ["mumps", "superlu_dist", "superlu"]:
        try:
            pc.setFactorSolverType(solver)
            break
        except Exception:
            pass
    ksp.setFromOptions()
    print(f"  PETSc direct solver: {pc.getFactorSolverType()}")

    Y = np.zeros(len(freqs), dtype=complex)

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f

        # A = (K + K_weak)·(1 + jω·β) + M·(−ω² + jω·α)
        K_ws.copy(result=A)
        A.scale(1.0 + 1j * omega * beta)
        A.axpy(-omega**2 + 1j * omega * alpha, M_petsc)
        A.assemble()

        ksp.setOperators(A)
        u = A.createVecRight()
        ksp.solve(F, u)

        Y[i] = 1j * omega * u.getValue(dof_z) / force_n
        u.destroy()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(freqs)}] {f:7.1f} Hz  |Y|={abs(Y[i]):.3e}")

    ksp.destroy()
    F.destroy()
    A.destroy()
    K_ws.destroy()
    return Y


def solve_harmonic_petsc_batch(K_petsc, M_petsc, dof_z_list, alpha: float,
                               beta: float, rigid_freq_hz: float,
                               freqs: np.ndarray, force_n: float = 1.0) -> np.ndarray:
    """Multi-bridge frequency sweep sharing ONE factorization per frequency.

    Same A(ω) as solve_harmonic_petsc (bit-identical formulation), but for a list
    of bridge DOFs.  A(ω) does NOT depend on the bridge point, so per frequency we
    factorize A once (first ksp.solve after setOperators) and REUSE that
    factorization for every bridge RHS (subsequent ksp.solve calls with the same
    operators only back-substitute).  This is the multi-RHS optimization: B bridge
    points cost 1 factorization + B back-substitutions per frequency, not B
    factorizations.

    Returns Y of shape (n_bridge, n_freq).  For a single bridge this is
    numerically identical to solve_harmonic_petsc (same operators, same solves).
    """
    from petsc4py import PETSc

    dof_z_list = [int(d) for d in dof_z_list]
    weak_spring_k = weak_spring_for_rigid_freq(M_petsc, rigid_freq_hz)

    K_ws = K_petsc.copy()
    kd = K_ws.getDiagonal(); kd.array += weak_spring_k
    K_ws.setDiagonal(kd); kd.destroy(); K_ws.assemble()

    # One reusable RHS vector, refilled per bridge.
    F = K_ws.createVecRight()
    A = K_ws.copy()
    u = A.createVecRight()

    ksp = PETSc.KSP().create()
    ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu")
    for solver in ["mumps", "superlu_dist", "superlu"]:
        try:
            pc.setFactorSolverType(solver); break
        except Exception:
            pass
    ksp.setFromOptions()
    print(f"  PETSc direct solver (batch, {len(dof_z_list)} bridges): "
          f"{pc.getFactorSolverType()}")

    Y = np.zeros((len(dof_z_list), len(freqs)), dtype=complex)

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        K_ws.copy(result=A)
        A.scale(1.0 + 1j * omega * beta)
        A.axpy(-omega**2 + 1j * omega * alpha, M_petsc)
        A.assemble()
        ksp.setOperators(A)          # first solve below factorizes; reused after
        for bi, dof in enumerate(dof_z_list):
            F.zeroEntries()
            F.setValue(dof, complex(force_n))
            F.assemblyBegin(); F.assemblyEnd()
            ksp.solve(F, u)
            Y[bi, i] = 1j * omega * u.getValue(dof) / force_n
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(freqs)}] {f:7.1f} Hz  "
                  f"|Y[0]|={abs(Y[0, i]):.3e}")

    ksp.destroy(); F.destroy(); A.destroy(); u.destroy(); K_ws.destroy()
    return Y


def compute_admittance_full_with_eigen_peaks(
    msh_path,
    material: dict,
    bridge_coords,
    *,
    bridge_points,
    freqs,
    output_dir="results/",
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 5e-6,
    force_n: float = 1.0,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
    eigen_fmax: float = 7500.0,
    n_modes: int = 400,
    top_k: int = 32,
    candidate_multiplier: int = 2,
    bulk: bool = False,
) -> dict:
    """Production solid backend: full harmonic main target + eigen peak labels.

    K and M are assembled once.  A weak-spring copy of K is used for the
    eigensolve, while the full harmonic path constructs its identical weak-spring
    operator internally.  The exact harmonic system is evaluated both on the
    canonical NN grid and at a bridge-specific union of eigenfrequency candidates.
    """
    from mpi4py import MPI
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "solid full/eigen dataset solve is serial; launch without mpirun")

    from fenics_modal_admittance import (
        find_bridge_nodes,
        rayleigh_modal_damping,
        solve_modal_eigen,
    )
    from peak_labels import (
        group_structural_modes,
        select_structural_candidate_union,
        structural_peak_labels,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs, float)
    if (freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs))
            or freqs[0] <= 0.0 or np.any(np.diff(freqs) <= 0.0)):
        raise ValueError("freqs must be finite, positive and strictly increasing")
    if (not np.isfinite(eigen_fmax) or eigen_fmax < freqs[-1]
            or int(n_modes) <= 0 or int(top_k) <= 0):
        raise ValueError("invalid solid eigen/peak configuration")

    pts = [np.asarray(p, float) for p in bridge_points]
    if not pts:
        raise ValueError("bridge_points must contain at least one point")
    timing: dict[str, float | dict] = {}
    alpha = float(material.get("alpha", rayleigh_alpha))
    beta = float(material.get("beta", rayleigh_beta))

    t0 = time.time()
    dolfin_mesh, bridge_idx0 = load_mesh(Path(msh_path), np.asarray(bridge_coords, float))
    bridge_meta = find_bridge_nodes(dolfin_mesh, pts)
    bridge_meta[0]["node_idx"] = int(bridge_idx0)
    bridge_meta[0]["bridge_snapped_xyz"] = (
        dolfin_mesh.geometry.x[bridge_idx0].astype(float).tolist())
    bridge_meta[0]["snap_distance_mm"] = float(np.linalg.norm(
        dolfin_mesh.geometry.x[bridge_idx0] - pts[0]))
    timing["mesh"] = time.time() - t0

    C_pa = engineering_to_C(material)
    t0 = time.time()
    K_petsc, M_petsc, V = assemble_KM_petsc(
        dolfin_mesh, C_pa, float(material["density"]))
    timing["assembly"] = time.time() - t0
    bs = V.dofmap.index_map_bs
    dof_list = [int(meta["node_idx"] * bs + 2) for meta in bridge_meta]

    K_eig = None
    try:
        K_eig = K_petsc.copy()
        weak_k = weak_spring_for_rigid_freq(M_petsc, rigid_freq_hz)
        diagonal = K_eig.getDiagonal()
        diagonal.array += weak_k
        K_eig.setDiagonal(diagonal)
        diagonal.destroy()
        K_eig.assemble()

        eig = solve_modal_eigen(
            K_eig, M_petsc, dof_list, n_modes=int(n_modes),
            sigma_hz=float(rigid_freq_hz))
        timing["eigenbasis"] = eig.get("timing", {})
        timing["eigensolve"] = float(eig.get("eig_time", 0.0))

        eig_freqs = np.asarray(eig["freqs_hz"], float)
        phi = np.asarray(eig["phi_at_dofs"], float)
        eig_max = float(np.max(eig_freqs)) if eig_freqs.size else 0.0
        coverage_boundary_rtol = 1.0e-6
        coverage_ok = bool(
            eig_max > float(eigen_fmax) * (1.0 + coverage_boundary_rtol))

        in_band = ((eig_freqs >= float(freqs[0]))
                   & (eig_freqs <= float(freqs[-1])))
        if not np.any(in_band):
            raise RuntimeError("solid eigensolve returned no modes in the analysis band")
        band_f = eig_freqs[in_band]
        band_zeta = rayleigh_modal_damping(
            2.0 * np.pi * band_f, alpha, beta)
        band_residues = np.square(phi[:, in_band])
        grouped_f, grouped_zeta, grouped_residues = group_structural_modes(
            band_f, band_zeta, band_residues)
        union, candidate_eligible = select_structural_candidate_union(
            grouped_f, grouped_zeta, grouped_residues,
            top_k=int(top_k), multiplier=int(candidate_multiplier))
        candidate_f = grouped_f[union]
        candidate_zeta = grouped_zeta[union]

        solve_freqs = np.unique(np.concatenate([freqs, candidate_f]))
        t0 = time.time()
        solved = solve_harmonic_petsc_batch(
            K_petsc, M_petsc, dof_list, alpha, beta, rigid_freq_hz,
            solve_freqs, force_n)
        timing["harmonic_sweep"] = time.time() - t0
        main_idx = np.searchsorted(solve_freqs, freqs)
        candidate_idx = np.searchsorted(solve_freqs, candidate_f)
        if (not np.array_equal(solve_freqs[main_idx], freqs)
                or not np.array_equal(solve_freqs[candidate_idx], candidate_f)):
            raise RuntimeError("failed to recover canonical/candidate frequencies")
        Y_main = solved[:, main_idx]
        Y_candidate = solved[:, candidate_idx]
        peak_batch = structural_peak_labels(
            candidate_f, candidate_zeta, Y_candidate, top_k=int(top_k),
            eligible=candidate_eligible)
        peak_payload = peak_batch.as_dict()
        total_visible = np.sum(grouped_residues > 0.0, axis=1, dtype=np.int64)
        peak_payload["peak_count_total"] = total_visible
        peak_payload["peak_truncated"] = total_visible > int(top_k)

        np.savez(
            str(output_dir / "admittance.npz"),
            frequencies=freqs, admittance=Y_main[0])
        np.savez(
            str(output_dir / "admittance_harmonic_multi.npz"),
            frequencies=freqs, admittance=Y_main,
            bridge_requested=np.asarray(
                [m["bridge_requested_xyz"] for m in bridge_meta], float),
            bridge_snapped=np.asarray(
                [m["bridge_snapped_xyz"] for m in bridge_meta], float),
            snap_distance_mm=np.asarray(
                [m["snap_distance_mm"] for m in bridge_meta], float))
        np.savez_compressed(str(output_dir / "peak_labels.npz"), **peak_payload)
        np.savez_compressed(
            str(output_dir / "solid_full_eigen.npz"),
            schema_version=np.array("solid-full-eigen-aux-v1"),
            eigenfrequency_hz=grouped_f,
            zeta=grouped_zeta,
            bridge_residue_inv_kg=grouped_residues,
            candidate_frequency_hz=candidate_f,
            candidate_zeta=candidate_zeta,
            candidate_eligible=candidate_eligible,
            candidate_admittance=Y_candidate,
            bridge_requested_xyz_mm=np.asarray(
                [m["bridge_requested_xyz"] for m in bridge_meta], float),
            bridge_snapped_xyz_mm=np.asarray(
                [m["bridge_snapped_xyz"] for m in bridge_meta], float),
            rayleigh_alpha=np.array(alpha), rayleigh_beta=np.array(beta),
            force_n=np.array(float(force_n)),
            time_convention=np.array("exp(+i omega t)"),
            response_units=np.array("m/s/N"))

        timing["total_s"] = float(
            timing["mesh"] + timing["assembly"]
            + float(eig.get("timing", {}).get("total_s", eig.get("eig_time", 0.0)))
            + timing["harmonic_sweep"])
        coverage = {
            "solver_revision": SOLID_FULL_PEAK_SOLVER_REVISION,
            "coverage_ok": coverage_ok,
            "eig_freq_max_hz": eig_max,
            "n_modes_retained": int(np.sum(eig_freqs <= float(eigen_fmax))),
            "n_eig_converged": int(eig["n_conv"]),
            "eigen_fmax": float(eigen_fmax),
            "coverage_boundary_rtol": coverage_boundary_rtol,
            "freq_min": float(freqs[0]), "freq_max": float(freqs[-1]),
            "freq_points": int(freqs.size),
            "damping": "rayleigh",
            "eigenbasis_diagnostics": eig.get("eigenbasis_diagnostics", {}),
            "n_grouped_in_band_modes": int(grouped_f.size),
            "n_exact_candidate_frequencies": int(candidate_f.size),
        }
        meta = {
            "backend": "full harmonic PETSc/MUMPS + auxiliary structural eigensolve",
            "solver_revision": SOLID_FULL_PEAK_SOLVER_REVISION,
            "coverage": coverage,
            "bridges": bridge_meta,
            "peak_label_source": "full harmonic evaluated at grouped eigenfrequencies",
            "peak_order": "select_amplitude_desc_store_frequency_asc",
            "top_k": int(top_k),
            "timing": timing,
        }
        (output_dir / "admittance_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

        if not bulk and _MPL_OK:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(Y_main[0]), 1e-30)))
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("|Y| [dB re 1 m/s/N]")
            ax.grid(True, which="both", alpha=0.4)
            fig.tight_layout()
            fig.savefig(str(output_dir / "admittance.png"), dpi=150)
            plt.close(fig)

        return {
            "freqs": freqs,
            "Y": Y_main[0],
            "Y_list": [Y_main[i] for i in range(Y_main.shape[0])],
            "bridge_meta": bridge_meta,
            "solver_revision": SOLID_FULL_PEAK_SOLVER_REVISION,
            "coverage_ok": coverage_ok,
            "eig_freq_max_hz": eig_max,
            "n_modes_retained": coverage["n_modes_retained"],
            "n_eig_converged": int(eig["n_conv"]),
            "eigenbasis_diagnostics": eig.get("eigenbasis_diagnostics", {}),
            "timing": timing,
            "peaks": peak_payload,
        }
    finally:
        if K_eig is not None:
            K_eig.destroy()
        K_petsc.destroy()
        M_petsc.destroy()


# ---------------------------------------------------------------------------
# Peak counting
# ---------------------------------------------------------------------------

def count_admittance_peaks(freqs: np.ndarray, Y: np.ndarray,
                            prominence_db: float = 3.0) -> int:
    """Count resonance peaks in |Y(ω)| with given prominence threshold [dB]."""
    from scipy.signal import find_peaks
    mag_db = 20.0 * np.log10(np.abs(Y) + 1e-30)
    peaks, _ = find_peaks(mag_db, prominence=prominence_db)
    return len(peaks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_admittance(
    msh_path,
    material: dict,
    bridge_coords,
    freq_min: float = 20.0,
    freq_max: float = 5000.0,
    freq_points: int = 500,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 5e-6,
    output_dir="results/",
    force_n: float = 1.0,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
    solver: str = "petsc",
    method: str = "harmonic",
    modal_fmax: float = 7500.0,
    modal_nmodes: int = 400,
    damping: str = "rayleigh",
    zeta_const: float = 0.01,
    bridge_points=None,
    save_modes: bool = False,
    include_rigid_modes: bool = True,
    residual_flexibility: bool = False,
    freqs=None,
) -> dict:
    """
    Compute bridge admittance Y(omega) = j*omega * u_z / F.

    `method` selects the solver backend:
      - "harmonic" (default): full frequency-by-frequency direct solve
        (solve_harmonic_petsc / scipy).  Unchanged legacy behaviour.
      - "modal": SLEPc eigensolve + modal superposition reconstruction
        (fenics_modal_admittance.compute_admittance_modal).  Supports
        bridge-point batching via `bridge_points`.

    Parameters
    ----------
    msh_path       : path to Gmsh .msh file
    material       : dict with E1..E3, G12..G23, nu12..nu23, density [SI]
    bridge_coords  : (x, y, z) in mm
    freq_min/max   : frequency sweep range [Hz]
    freq_points    : number of log-spaced frequency points
    rayleigh_alpha : mass-proportional damping coefficient
    rayleigh_beta  : stiffness-proportional damping coefficient
    output_dir     : where to write admittance.npz and admittance.png
    force_n        : applied force magnitude [N]
    rigid_freq_hz  : target rigid-body suspension frequency [Hz]; the diagonal
                     spring is sized per mesh so rigid modes land here (out of band)

    Returns
    -------
    dict with keys "freqs" [Hz] and "Y" [m/s/N] (complex array)
    """
    # --- Modal backend dispatch (additive; harmonic path below is untouched) --
    if method == "modal":
        sys.path.insert(0, str(Path(__file__).parent))
        from fenics_modal_admittance import compute_admittance_modal
        return compute_admittance_modal(
            msh_path=msh_path,
            material=material,
            bridge_coords=bridge_coords,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_points=freq_points,
            rayleigh_alpha=rayleigh_alpha,
            rayleigh_beta=rayleigh_beta,
            output_dir=output_dir,
            force_n=force_n,
            rigid_freq_hz=rigid_freq_hz,
            modal_fmax=modal_fmax,
            n_modes=modal_nmodes,
            damping=damping,
            zeta_const=zeta_const,
            bridge_points=bridge_points,
            save_modes=save_modes,
            include_rigid_modes=include_rigid_modes,
            residual_flexibility=residual_flexibility,
            freqs=freqs,
        )
    elif method != "harmonic":
        raise ValueError(f"Unknown method '{method}' (use 'harmonic' or 'modal')")

    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_arr   = np.asarray(bridge_coords, dtype=float)

    alpha = material.get("alpha", rayleigh_alpha)
    beta  = material.get("beta",  rayleigh_beta)

    # 1. Mesh
    dolfin_mesh, bridge_idx = load_mesh(Path(msh_path), bridge_arr)

    # Resolve every requested bridge while mesh coordinates are explicitly in
    # their public mm frame.  Matrix assembly temporarily scales coordinates to
    # metres and restores them, but node identity must not depend on that internal
    # implementation detail.
    bridge_meta_h = None
    if bridge_points:
        from fenics_modal_admittance import find_bridge_nodes
        pts = [np.asarray(p, dtype=float) for p in bridge_points]
        bridge_meta_h = find_bridge_nodes(dolfin_mesh, pts)
        bridge_meta_h[0]["node_idx"] = bridge_idx
        bridge_meta_h[0]["bridge_snapped_xyz"] = \
            dolfin_mesh.geometry.x[bridge_idx].astype(float).tolist()
        bridge_meta_h[0]["snap_distance_mm"] = float(np.linalg.norm(
            dolfin_mesh.geometry.x[bridge_idx] - pts[0]))

    # 2. Matrices + frequency sweep
    C_pa = engineering_to_C(material)
    print(f"\nC diagonal (GPa): {np.diag(C_pa)/1e9}")

    if freqs is not None:
        freqs = np.asarray(freqs, dtype=float)
        if (freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs))
                or freqs[0] <= 0.0 or np.any(np.diff(freqs) <= 0.0)):
            raise ValueError("explicit freqs must be finite, positive, and strictly increasing")
        freq_min, freq_max, freq_points = float(freqs[0]), float(freqs[-1]), int(freqs.size)
    else:
        freqs = np.geomspace(freq_min, freq_max, freq_points)
    print(f"\nFrequency sweep: {freq_min}-{freq_max} Hz, {freq_points} points")
    print("-" * 60)

    # Harmonic multi-bridge batching state (populated only in the petsc batch path)
    Y_list = None

    if solver == "petsc":
        K_petsc, M_petsc, V = assemble_KM_petsc(dolfin_mesh, C_pa, material["density"])
        n_dof = K_petsc.getSize()[0]
        if bridge_points:
            # --- Multi-bridge batch: ONE factorization/freq shared across bridges
            dof_list = [bridge_dof_z(V, m["node_idx"]) for m in bridge_meta_h]
            print(f"\nBridge DOFs (Z), {len(dof_list)} bridges: {dof_list[:5]}"
                  + (" ..." if len(dof_list) > 5 else ""))
            Y_multi = solve_harmonic_petsc_batch(
                K_petsc, M_petsc, dof_list, alpha, beta, rigid_freq_hz, freqs, force_n)
            Y_list = [Y_multi[b] for b in range(Y_multi.shape[0])]
            Y = Y_list[0]
        else:
            dof_z = bridge_dof_z(V, bridge_idx)
            print(f"\nBridge DOF (Z): {dof_z} / {n_dof}")
            Y = solve_harmonic_petsc(K_petsc, M_petsc, dof_z, alpha, beta,
                                      rigid_freq_hz, freqs, force_n)
        K_petsc.destroy(); M_petsc.destroy()
    else:
        K, M, V = assemble_KM(dolfin_mesh, C_pa, material["density"])
        n_dof = K.shape[0]
        dof_z = bridge_dof_z(V, bridge_idx)
        print(f"\nBridge DOF (Z): {dof_z} / {n_dof}")
        weak_spring_k = weak_spring_for_rigid_freq(M, rigid_freq_hz)
        K_weak = sp.diags(np.full(n_dof, weak_spring_k, dtype=float), format="csr")
        F = np.zeros(n_dof, dtype=complex)
        F[dof_z] = force_n
        Y = np.zeros(freq_points, dtype=complex)
        for i, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            u     = solve_harmonic(K, M, alpha, beta, K_weak, F, f)
            u_z   = u[dof_z]
            Y[i]  = 1j * omega * u_z / force_n
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1:3d}/{freq_points}] {f:7.1f} Hz  "
                      f"|Y|={abs(Y[i]):.3e}  Re(uz)={u_z.real:.3e}  Im(uz)={u_z.imag:.3e} m")

    # 5. Save (single-bridge admittance.npz keeps its legacy keys unchanged)
    npz_path = output_dir / "admittance.npz"
    np.savez(str(npz_path), frequencies=freqs, admittance=Y)
    print(f"\nSaved: {npz_path}")

    # Multi-bridge batch: also save the (n_bridge, n_freq) matrix + snap metadata.
    if Y_list is not None and bridge_meta_h is not None and len(Y_list) > 1:
        np.savez(str(output_dir / "admittance_harmonic_multi.npz"),
                 frequencies=freqs,
                 admittance=np.array(Y_list),
                 bridge_requested=np.array([m["bridge_requested_xyz"] for m in bridge_meta_h]),
                 bridge_snapped=np.array([m["bridge_snapped_xyz"] for m in bridge_meta_h]),
                 snap_distance_mm=np.array([m["snap_distance_mm"] for m in bridge_meta_h]))
        print(f"Saved multi-bridge: {output_dir / 'admittance_harmonic_multi.npz'}")

    # 6. Plot
    if _MPL_OK:
        mag_db = 20.0 * np.log10(np.abs(Y) + 1e-30)
        n_peaks = count_admittance_peaks(freqs, Y)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(freqs, mag_db, linewidth=1.2)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|Y(omega)| [dB re 1 m/s/N]")
        ax.set_title(f"Guitar Body Bridge Admittance  |  N_peaks={n_peaks}, β={beta:.0e}")
        ax.grid(True, which="both", alpha=0.4)
        ax.set_xlim(freq_min, freq_max)
        fig.tight_layout()
        plot_path = output_dir / "admittance.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"Plot saved: {plot_path}")
    else:
        print("Plot skipped (matplotlib unavailable at import time)")

    result = {"freqs": freqs, "Y": Y}
    if Y_list is not None:
        result["Y_list"] = Y_list
        result["bridge_meta"] = bridge_meta_h
    return result


# ---------------------------------------------------------------------------
# Modal analysis (SLEPc eigensolver)
# ---------------------------------------------------------------------------

def modal_analysis(
    msh_path,
    material: dict,
    bridge_coords,
    freq_min: float = 20.0,
    freq_max: float = 5000.0,
    n_modes: int = 200,
    rigid_freq_hz: float = _RIGID_FREQ_HZ,
) -> np.ndarray:
    """
    Compute eigenfrequencies using SLEPc shift-invert eigensolver.

    Returns sorted array of eigenfrequencies [Hz] within [freq_min, freq_max].
    """
    try:
        from slepc4py import SLEPc
    except ImportError:
        raise RuntimeError("SLEPc not available — install petsc4py+slepc4py in fenicsx env")

    bridge_arr = np.asarray(bridge_coords, dtype=float)
    C_pa       = engineering_to_C(material)
    dolfin_mesh, _ = load_mesh(Path(msh_path), bridge_arr)
    K_p, M_p, V   = assemble_KM_petsc(dolfin_mesh, C_pa, material["density"])

    # Add weak diagonal spring to K (same mesh-independent sizing as harmonic solve)
    from petsc4py import PETSc
    weak_spring_k = weak_spring_for_rigid_freq(M_p, rigid_freq_hz)
    kd = K_p.getDiagonal()
    kd.array += weak_spring_k
    K_p.setDiagonal(kd)
    kd.destroy()
    K_p.assemble()

    print(f"\nSLEPc modal analysis: {n_modes} modes requested ...")

    eps = SLEPc.EPS().create()
    eps.setOperators(K_p, M_p)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(n_modes)

    # Shift-invert around freq_min^2 to find lowest modes first
    sigma = (2.0 * np.pi * freq_min) ** 2
    eps.setTarget(sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc  = ksp.getPC()
    pc.setType("lu")
    for solver in ["mumps", "superlu_dist", "superlu"]:
        try:
            pc.setFactorSolverType(solver); break
        except Exception:
            pass

    eps.solve()
    n_conv = eps.getConverged()
    print(f"  Converged eigenpairs: {n_conv}")

    freqs = []
    for i in range(n_conv):
        lam = eps.getEigenvalue(i).real
        if lam > 0:
            f = np.sqrt(lam) / (2.0 * np.pi)
            if freq_min <= f <= freq_max:
                freqs.append(f)

    freqs = np.array(sorted(freqs))
    print(f"  Modes in [{freq_min:.0f}, {freq_max:.0f}] Hz: {len(freqs)}")
    if len(freqs):
        print(f"  First 10 eigenfreqs: {np.round(freqs[:10], 1)}")

    K_p.destroy(); M_p.destroy()
    return freqs


# ---------------------------------------------------------------------------
# Legacy main (backward compat)
# ---------------------------------------------------------------------------

def main():
    compute_admittance(
        msh_path=_MESH_FILE,
        material=_MATERIAL,
        bridge_coords=_BRIDGE_COORDS,
        freq_min=_FREQ_MIN,
        freq_max=_FREQ_MAX,
        freq_points=_FREQ_POINTS,
        rayleigh_alpha=_MATERIAL["alpha"],
        rayleigh_beta=_MATERIAL["beta"],
        output_dir=_RESULTS_DIR,
    )


# ---------------------------------------------------------------------------
# Box mesh validation test
# ---------------------------------------------------------------------------

def test_box():
    """Quick validation: box mesh resonance peaks (no guitar.msh needed)."""
    import dolfinx.mesh as dmesh
    import dolfinx.fem as fem
    import dolfinx.fem.petsc as petsc_fem
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    print("=" * 60)
    print("BOX MESH TEST (500x350x50 mm)")
    print("=" * 60)

    Lx, Ly, Lz = 0.500, 0.350, 0.050
    mesh = dmesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [20, 14, 4],
        cell_type=dmesh.CellType.tetrahedron,
    )

    coords = mesh.geometry.x
    target = np.array([Lx/2, Ly/4, Lz])
    bridge_idx = int(np.argmin(np.sum((coords - target)**2, axis=1)))
    print(f"Bridge node: {coords[bridge_idx]}")

    V   = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    u   = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    C_pa = engineering_to_C(_MATERIAL)

    def eps_voigt(w):
        e = ufl.sym(ufl.grad(w))
        return ufl.as_vector([e[0,0], e[1,1], e[2,2], 2*e[1,2], 2*e[0,2], 2*e[0,1]])

    C_ufl = fem.Constant(mesh, PETSc.ScalarType(C_pa))
    a_K   = ufl.inner(ufl.dot(C_ufl, eps_voigt(u)), eps_voigt(v)) * ufl.dx
    rho   = fem.Constant(mesh, PETSc.ScalarType(_MATERIAL["density"]))
    a_M   = rho * ufl.inner(u, v) * ufl.dx

    K_petsc = petsc_fem.assemble_matrix(fem.form(a_K)); K_petsc.assemble()
    M_petsc = petsc_fem.assemble_matrix(fem.form(a_M)); M_petsc.assemble()

    def to_scipy(A):
        ai, aj, av = A.getValuesCSR()
        return sp.csr_matrix((av, aj, ai), shape=A.getSize())

    K = to_scipy(K_petsc)
    M = to_scipy(M_petsc)

    print(f"M diag range: {M.diagonal().min():.3e} ~ {M.diagonal().max():.3e}")

    bs    = V.dofmap.index_map_bs
    dof_z = bridge_idx * bs + 2
    F_vec = np.zeros(n_dof, dtype=complex)
    F_vec[dof_z] = 1.0
    weak_spring_k = weak_spring_for_rigid_freq(M, _RIGID_FREQ_HZ)
    K_weak = sp.diags(np.full(n_dof, weak_spring_k), format="csr")

    freqs_test = np.geomspace(20, 3000, 80)
    Y_test = np.zeros(len(freqs_test), dtype=complex)
    for i, f in enumerate(freqs_test):
        omega = 2.0 * np.pi * f
        u_vec = solve_harmonic(K, M, _MATERIAL["alpha"], _MATERIAL["beta"], K_weak, F_vec, f)
        Y_test[i] = 1j * omega * u_vec[dof_z]

    _RESULTS_DIR.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    mag_db = 20.0 * np.log10(np.abs(Y_test) + 1e-30)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freqs_test, mag_db, linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|Y| [dB re 1 m/s/N]")
    ax.set_title("Box Mesh Test — Engelmann Spruce")
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    out = _RESULTS_DIR / "admittance_box_test.png"
    fig.savefig(str(out), dpi=150)
    print(f"Plot saved: {out}")

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(mag_db, prominence=3.0)
    if len(peaks):
        print(f"Resonance peaks: {freqs_test[peaks].round(1)} Hz  <- M != 0 confirmed")
    else:
        print("No peaks found — check M matrix")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="FEniCSx guitar bridge admittance")
    parser.add_argument("--test", action="store_true", help="Run box mesh validation")
    parser.add_argument("--msh",  type=str,   default=None, help="Path to .msh file")
    parser.add_argument("--bridge", type=float, nargs=3, default=[0.0, -225.0, 100.0],
                        metavar=("X", "Y", "Z"), help="Bridge coords [mm]")
    parser.add_argument("--material",      type=str,   default=None,
                        help="Material name (from materials.py)")
    parser.add_argument("--material-json", type=str,   default=None,
                        help="Material as JSON string")
    parser.add_argument("--freq-min",      type=float, default=20.0)
    parser.add_argument("--freq-max",      type=float, default=5000.0)
    parser.add_argument("--freq-points",   type=int,   default=300)
    parser.add_argument("--rayleigh-alpha", type=float, default=0.0)
    parser.add_argument("--rayleigh-beta",  type=float, default=5e-6)
    parser.add_argument("--output-dir",    type=str,   default="results/")
    parser.add_argument("--solver",        type=str,   default="petsc",
                        choices=["petsc", "scipy"], help="Harmonic solver backend")
    parser.add_argument("--method",        type=str,   default="harmonic",
                        choices=["harmonic", "modal"],
                        help="Admittance backend: full harmonic (default) or modal eigensolve")
    parser.add_argument("--modal-fmax",    type=float, default=7500.0,
                        help="Modal: compute/keep modes up to this frequency [Hz]")
    parser.add_argument("--modal-nmodes",  type=int,   default=400,
                        help="Modal: number of lowest eigenpairs to request")
    parser.add_argument("--damping",       type=str,   default="rayleigh",
                        choices=["rayleigh", "constant"],
                        help="Modal damping model (default: rayleigh-equivalent)")
    parser.add_argument("--zeta",          type=float, default=0.01,
                        help="Constant modal damping ratio (only if --damping constant)")
    parser.add_argument("--bridge-points-json", type=str, default=None,
                        help="Harmonic (batch) & modal: JSON list of [x,y,z] bridge points")
    parser.add_argument("--save-modes",    action="store_true",
                        help="Modal: also save modes.npz (eigenfreqs, participation)")
    parser.add_argument("--modal-no-rigid", action="store_true",
                        help="Modal ABLATION: exclude rigid/weak-spring modes "
                             "(drops the free-free low-frequency mass line - worse)")
    parser.add_argument("--modal-residual", action="store_true",
                        help="Modal EXPERIMENTAL: enable residual flexibility "
                             "(currently unstable for free-free - see docs; off by default)")
    args = parser.parse_args()

    if args.modal_residual:
        print("WARNING: --modal-residual is EXPERIMENTAL and known to blow up for "
              "free-free structures (rigid/nullspace not deflated). Diagnostics are "
              "saved to admittance_meta.json; the blowup guard may disable it per bridge.")

    if args.test:
        test_box()
    elif args.msh:
        if args.material_json:
            mat = json.loads(args.material_json)
        elif args.material:
            sys.path.insert(0, str(Path(__file__).parent))
            from materials import get_material
            mat = get_material(args.material)
        else:
            mat = _MATERIAL

        bridge_points = None
        if args.bridge_points_json:
            bridge_points = _load_json_arg(args.bridge_points_json)

        compute_admittance(
            msh_path=args.msh,
            material=mat,
            bridge_coords=tuple(args.bridge),
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            freq_points=args.freq_points,
            rayleigh_alpha=args.rayleigh_alpha,
            rayleigh_beta=args.rayleigh_beta,
            output_dir=args.output_dir,
            solver=args.solver,
            method=args.method,
            modal_fmax=args.modal_fmax,
            modal_nmodes=args.modal_nmodes,
            damping=args.damping,
            zeta_const=args.zeta,
            bridge_points=bridge_points,
            save_modes=args.save_modes,
            include_rigid_modes=not args.modal_no_rigid,
            residual_flexibility=args.modal_residual,
        )
    else:
        main()
