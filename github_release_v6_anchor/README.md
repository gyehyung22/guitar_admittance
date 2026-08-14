# Guitar Body Bridge-Admittance Surrogate

Research code for generating finite-element bridge-admittance data from guitar-body
geometry and orthotropic material properties, then training neural surrogates for
the frequency response.

This public snapshot is deliberately narrow. It contains the production `mixed-v6`
pipeline and the controlled anchor-dataset planner, together with the CAD, mesh,
FEM, model-reduction, data-loading, and neural-network code they require. It does
not contain generated meshes, FEM results, datasets, checkpoints, logs, exploratory
analysis, one-off launch scripts, or test programs.

## Status

- Dataset generation contract: `mixed-v6`.
- Main target: bridge driving-point admittance magnitude, `20 log10 |Y|`, at 500
  logarithmically spaced frequencies from 20 to 5000 Hz.
- Auxiliary target: up to 32 labelled resonances per bridge point.
- Solid bodies: full structural harmonic FEM for the canonical response.
- Hollow bodies: reduced, two-way structure-air coupled solver using the same
  operators as the full coupled reference formulation.
- Neural architecture: active research. The repository contains implemented
  baselines, ablations, spatial variants, and spectral decoders; it does not claim
  that one architecture is final.
- Audio synthesis is outside this snapshot. The generated admittance is the physical
  transfer function intended to support that later stage.

## Pipeline

```text
parametric contour + thickness + soundhole + bridge points + material
                              |
                              v
                    CadQuery STEP geometry
                              |
                              v
             Gmsh conformal tetrahedral mesh
                    /                     \
                   /                       \
         solid structural FEM       hollow wood-air FEM
          full harmonic solve        reduced coupled solve
                   \                       /
                    \                     /
                     v6 response contract
             500-bin log|Y| + 32 peak labels
                              |
                              v
            shape-disjoint PyTorch surrogate training
```

The complete physical and numerical formulation is in
[`docs/METHODS.md`](docs/METHODS.md). Dataset and tensor schemas are in
[`docs/DATA_CONTRACTS.md`](docs/DATA_CONTRACTS.md).
Packaging checks and deliberately unverified items are recorded in
[`docs/PACKAGING_REPORT.md`](docs/PACKAGING_REPORT.md).

## Production Dataset Design

The default v6 plan contains:

| Axis | Default |
|---|---:|
| Base contours | 100, including 8 canonical guitar outlines |
| Materials | 10 independent SPD orthotropic draws per contour |
| Body realizations | solid and hollow |
| Bridge points | 10 per case, solved in one batch |
| FEM cases | 2,000 |
| Supervised samples | 20,000 |
| Frequency grid | 500 log-spaced points, 20 to 5000 Hz |
| Peak slots | 32 per bridge |

The immutable plan is created before FEM work. Every case is keyed by shape, body
type, and material; all bridge points share the case solve. Results are staged,
checked, and atomically committed. Resume accepts only artifacts that still satisfy
the plan and QC contracts.

## Anchor Dataset

`plan_anchor_dataset.py` is included as the current work-in-progress controlled-plan
implementation. Its design is still being tested and is not certified by this source
packaging pass. It uses a completed or in-progress v6 plan as its source:

- Block C varies scalar geometry while reusing source contours.
- Block A scales source contours and their primary dimensions. Exact `1/s`
  eigenfrequency similarity applies only to the solid bodies; fixed-mm hollow-body
  wall and back dimensions break exact hollow similarity. Fixed Rayleigh damping
  also changes modal damping under scale.
- Block B changes contour while holding characteristic area, thickness, material,
  normalized bridge layout, and normalized soundhole layout fixed. Width, height,
  and other contour-derived quantities can still differ.

Anchor materials are two SPD vectors copied identically across shapes. Splits are
inherited from the source v6 contour to prevent geometry leakage when datasets are
combined.

## Environment

The FEM stack is Linux-only in this project and is normally run under WSL2:

```bash
conda create -n fenicsx -c conda-forge \
  python=3.11 fenics-dolfinx 'petsc=*=complex*' 'slepc=*=complex*' \
  petsc4py slepc4py mpi4py mpich mumps-mpi \
  numpy scipy pandas matplotlib meshio gmsh cadquery shapely opencv
conda activate fenicsx
pip install torch
```

Exact package availability and PETSc/MUMPS builds vary by platform. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before launching FEM work.

## Minimal Workflow

Create the immutable v6 plan without running FEM:

```bash
python dataset_gen_mixed.py \
  --output-dir results/mixed_v6_full \
  --n-base-shapes 100 --n-materials 10 --n-bridge-points 10 \
  --plan-only
```

Generate or resume it:

```bash
python dataset_gen_mixed.py \
  --output-dir results/mixed_v6_full \
  --n-base-shapes 100 --n-materials 10 --n-bridge-points 10 \
  --resume
```

Audit an anchor design without writing a plan:

```bash
python plan_anchor_dataset.py \
  --source results/mixed_v6_full \
  --out results/mixed_anchor_v1 \
  --dry-run
```

Write the anchor plan by removing `--dry-run`, then invoke
`dataset_gen_mixed.py` with the same configuration stored in that plan and with the
anchor output directory. See the reproducibility notes for the plan-immutability
requirements.

Train a baseline after the dataset has published its readiness certificate:

```bash
python nn_train.py \
  --dataset results/mixed_v6_full results/mixed_anchor_v1 \
  --model full --split-mode shape --require-certified \
  --out nn_runs/baseline
```

The NN CLI exposes diagnostic baselines and alternative architectures. Treat those
as experiments, not as production defaults.

## Source Map

| Area | Primary files |
|---|---|
| v6 planning, orchestration, QC | `dataset_gen_mixed.py`, `_dataset_solve_worker.py` |
| Anchor plan | `plan_anchor_dataset.py` |
| Shapes and placement | `shape_gen.py`, `guitar_shapes.py`, `placement_utils.py` |
| CAD | `backend/model_builder.py`, `backend/holes.py`, `dataset_gen.py` |
| Meshing | `mesh_gen.py` |
| Solid FEM | `fenics_admittance.py`, `fenics_modal_admittance.py` |
| Full coupled reference operators | `fenics_admittance_coupled.py`, `fsi_coupling.py`, `air_acoustics.py` |
| Reduced coupled production solver | `modal_coupled_admittance.py`, `modal_eigenbasis.py`, `reduced_model_io.py` |
| Soundhole port and labels | `acoustic_helmholtz.py`, `peak_labels.py` |
| NN data and models | `nn_dataset.py`, `nn_model.py`, `nn_spectrum_decoders.py`, `nn_train.py` |
| Spectral evaluation | `spectral_metrics.py`, `_peak_backend.py` |

## Publication Notes

No license has been selected in this snapshot. Add an explicit license and citation
metadata before making the repository public. Generated data and trained weights
need their own provenance, storage, and licensing decisions.
