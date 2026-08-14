# Reproducibility Guide

This repository is a source snapshot. It contains no generated geometry, mesh,
dataset, checkpoint, or experiment result. Reproducing numbers therefore requires a
Linux FEniCSx/PETSc environment and substantial computation.

## 1. Platform

The production FEM path was developed under Linux in WSL2. Required compiled
components include:

- DOLFINx and UFL;
- PETSc with complex scalars;
- SLEPc;
- PETSc/SLEPc Python bindings;
- MPI;
- a MUMPS-enabled PETSc factorization backend;
- Gmsh, CadQuery, OpenCASCADE, Shapely, NumPy, SciPy, and MeshIO.

PyTorch and pandas are required for NN training. W&B is optional.

Start with:

```bash
conda env create -f environment.yml
conda activate guitar-admittance
```

Conda solver availability differs by operating system and date. Verify the resulting
PETSc scalar type and MUMPS availability before starting a long run. The response
code expects complex harmonic matrices.

Install `wandb` only when experiment tracking is needed. `multiphenicsx` is optional
for the alternative coupling-assembly route; the production conformal-facet route
does not require it.

## 2. Fast Source Checks

From the repository root:

```bash
export PYTHONUTF8=1
python -m py_compile *.py backend/*.py
python dataset_gen_mixed.py --help
python plan_anchor_dataset.py --help
python nn_train.py --help
```

These checks do not validate FEniCS, Gmsh, or numerical accuracy. They only catch
source/import/CLI failures.

Run them inside the Linux/WSL environment. Native Windows is not a supported FEM
runtime; a Korean Windows code page can also fail to print Unicode characters present
in CLI help unless Python UTF-8 mode is enabled.

## 3. Create a v6 Plan

Create and inspect the immutable plan before solving:

```bash
python dataset_gen_mixed.py \
  --output-dir results/mixed_v6_full \
  --n-base-shapes 100 \
  --n-materials 10 \
  --n-bridge-points 10 \
  --plan-only
```

The default seed is `20260720`. Record the printed plan hash. Do not edit
`dataset_plan.json` by hand.

## 4. Generate or Resume v6

The simplest single-orchestrator run is:

```bash
python dataset_gen_mixed.py \
  --output-dir results/mixed_v6_full \
  --n-base-shapes 100 \
  --n-materials 10 \
  --n-bridge-points 10 \
  --resume
```

The code also supports shape-based shards and parallel geometry-cache warming. Those
settings are machine-dependent. Sparse factorization and eigensolution can be limited
by memory bandwidth and RAM rather than core count. Benchmark representative solid
and hollow cases before choosing worker/thread counts.

Do not run several unconstrained threaded workers merely to occupy all cores. Each
process can instantiate threaded BLAS and sparse-solver work, causing memory exhaustion
or oversubscription.

## 5. Verify a Published Dataset

The generator publishes `dataset_ready.json` and an artifact inventory after every
planned case is terminal. Inspect at least:

- plan hash and generator version;
- planned, successful, failed, and sample counts;
- completion policy;
- mode escalation and terminal failure records;
- `peak_truncated` distribution;
- mesh/geometry digest and source fingerprint consistency.

A readiness certificate can include terminally failed cases that were deliberately
dropped. It certifies coherent publication under the stated policy, not a perfect
factorial grid.

## 6. Build an Anchor Plan

The anchor planner is a work in progress in this snapshot. The commands below document
its current interface; they are not evidence that the design or FEM output is final.

The anchor planner requires the source v6 `dataset_plan.json`:

```bash
python plan_anchor_dataset.py \
  --source results/mixed_v6_full \
  --out results/mixed_anchor_v1 \
  --dry-run
```

The dry run checks contour simplicity, hole and bridge containment, cavity height,
SPD materials, shared material vectors, and inherited splits. It writes nothing.

Remove `--dry-run` to write the plan and `anchor_design.json`. The latter preserves
the base-shape-to-block map needed for per-block analysis. The command prints the
realized shape and case counts because invalid Block B candidates can be rejected
rather than silently given different bridge or soundhole placements. For an older
anchor plan whose block map was not persisted, rerun the identical deterministic
planner arguments with `--dry-run --emit-blocks recovered_blocks.json`.

Generate the written plan by invoking `dataset_gen_mixed.py` on the anchor directory
with configuration values matching the anchor plan. At minimum, pass the reported
shape count, `--n-materials 2`, `--n-bridge-points 10`, and the anchor seed
`20260805`; preserve the source plan's solver budgets and frequency contract. A
mismatch must abort rather than replace the plan.

## 7. Train a Neural Baseline

After the required datasets are ready:

```bash
python nn_train.py \
  --dataset results/mixed_v6_full results/mixed_anchor_v1 \
  --model full \
  --split-mode shape \
  --target bins \
  --target-norm global \
  --require-certified \
  --out nn_runs/full_baseline
```

The code saves the model configuration, best checkpoint, normalization statistics,
and metrics inside the run directory. Preserve those together. A checkpoint without
its exact normalization and model configuration is not a reproducible predictor.

For architecture research, change one controlled option at a time and retain the
same dataset selection and shape split. `physics_only`, shuffled/constant shape,
capacity-matched scalar, spatial-query, relational, decoder, and PCA experiments
answer different questions and should not be compared after silently changing the
target or normalization.

## 8. Numerical Validation Expected Before Publication

The lightweight tests and validation launchers used during development are excluded
from this minimal snapshot at the user's request. Before publishing scientific
results, regenerate and archive evidence for:

1. CAD requested-versus-realized dimensions and cavity volume.
2. Conformal FSI topology, physical tags, soundhole position, and mesh convergence.
3. Solid full-harmonic mesh convergence.
4. Hollow reduced E versus full coupled D over representative shape/material/plate
   extremes.
5. Eigen residuals, mass orthonormality, modal-band coverage, and reduced-model
   reconstruction.
6. Shape-disjoint data splits and cross-dataset contour-digest consistency.
7. Model results over several random seeds with the exact training configuration.

The source code contains fail-closed production checks, but those checks are not a
substitute for publishing convergence and reference-solver evidence.

## 9. Source Integrity

`SOURCE_SNAPSHOT_SHA256.json` records the copied implementation files and their hashes
at packaging time. Documentation and environment files are not part of the running
v6 plan's implementation fingerprint.

If generated data is published later, archive its original `dataset_plan.json`,
`dataset_ready.json`, artifact inventory, code commit, package environment, and
hardware/solver configuration with it.
