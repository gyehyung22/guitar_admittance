# Repository Scope and File Inventory

The folder is a minimal review/publication snapshot, not a copy of the working
research directory.

## Included

### Dataset plans and orchestration

- `dataset_gen_mixed.py`: v6 plan, geometry cache, case orchestration, retries, QC,
  atomic commit, resume, manifest, and readiness publication.
- `_dataset_solve_worker.py`: isolated solid/hollow case solver worker.
- `plan_anchor_dataset.py`: controlled v6-derived anchor plan.
- `plan_source_compatibility.json`: audited source-fingerprint transitions.

### Geometry and mesh

- `guitar_shapes.py`, `shape_gen.py`: canonical and random outer contours.
- `placement_utils.py`: bridge and soundhole placement.
- `dataset_gen.py`: direct CAD payload helper used by v6. Its older standalone
  dataset CLI is retained only because the helper lives in this module.
- `backend/model_builder.py`, `backend/holes.py`: CadQuery body/cavity/hole builder.
- `mesh_gen.py`: structural and conformal wood-air Gmsh meshes.

### FEM and reduction

- `materials.py`: named material records used by legacy/helper paths.
- `fenics_admittance.py`: structural assembly, weak spring, harmonic solve, solid
  canonical response, and peak candidate path.
- `fenics_modal_admittance.py`: structural modal implementation and shared helpers.
- `modal_eigenbasis.py`: real, degeneracy-safe generalized eigenbasis recovery.
- `air_acoustics.py`: acoustic operators and active pressure space.
- `fsi_coupling.py`: conformal interface coupling assembly.
- `fenics_admittance_coupled.py`: full two-way coupled reference and shared blocks.
- `acoustic_helmholtz.py`: soundhole impedance/radiation model.
- `modal_coupled_admittance.py`: reduced coupled production backend E.
- `reduced_model_io.py`: portable reduced-model loading and response reconstruction.
- `peak_labels.py`: common solid/hollow peak schema and extraction.

### Neural surrogate

- `nn_dataset.py`: manifest loader, raster/SDF preprocessing, scalar/relational
  features, splits, normalization, and PyTorch dataset.
- `nn_model.py`: baseline, diagnostic, residual, expert, spatial-query, relational,
  case/bridge, and PCA-output models.
- `nn_spectrum_decoders.py`: direct, envelope/detail, and salience decoders.
- `nn_train.py`: training/evaluation CLI.
- `spectral_metrics.py`, `_peak_backend.py`: event-aware spectral metrics.

### Documentation and dependencies

- `README.md`, `docs/*.md`, `environment.yml`, `requirements.txt`, `.gitignore`.
- `SOURCE_SNAPSHOT_SHA256.json`: package-time hashes of copied implementation files.

## Excluded Intentionally

- all `results/`, dataset, shape cache, CAD, mesh, response, and manifest artifacts;
- all NN runs, checkpoints, normalization files, W&B logs, plots, and tables;
- unit tests, FEM validators, smoke tests, regression gates, diagnostic notebooks,
  and one-off analysis scripts;
- server launch wrappers, shell run policies, monitoring scripts, and machine-specific
  commands;
- experimental MMG and anisotropic-mesh probes;
- legacy v4/v5 production launch paths;
- frontend/inference demo code;
- unfinished audio synthesis code.

The exclusions keep the review focused on the v6 and anchor implementation. They also
mean this snapshot alone is not the complete evidence package for a paper. Numerical
validation scripts and result artifacts should be archived separately when the study
is frozen.
