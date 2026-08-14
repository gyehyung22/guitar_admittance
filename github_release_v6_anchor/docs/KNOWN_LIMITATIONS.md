# Known Limitations and Scope Boundaries

## Physics

- The structure is linear, small-strain, and frequency independent except for
  Rayleigh damping. Joints, contact, nonlinear material behavior, strings, pickups,
  bridge hardware, and neck coupling are not represented.
- Material axes are globally aligned. Grain curvature, local rotations, laminate
  construction, and spatially varying material are not modelled.
- Bracing is disabled in the v6 production contract.
- Hollow-body air is an internal acoustic domain. Exterior radiation is reduced to
  a low-`ka` lumped soundhole resistance/end correction rather than a full exterior
  boundary-element or infinite-element acoustic field.
- The soundhole port assumes one collective piston-like volume-flow coordinate.
  Strongly nonuniform aperture flow and multiple interacting ports need a richer
  port model.
- The weak spring is a numerical free-free regularization. It intentionally moves
  rigid modes to about 1 Hz and should remain well below the analysis band.
- Semi-hollow CAD support is present, but semi-hollow cases are not in v6 and have no
  production dataset certification here.

## Numerical Method

- Production hollow responses use a reduced solver. It projects the same two-way
  coupled operators as the full reference, but its accuracy still depends on modal
  coverage and attachment enrichment. The built-in checks detect several failures,
  not every possible modelling error.
- The `craig-bampton` label denotes static/residual attachment enrichment. It is not
  classical fixed-interface Craig-Bampton component-mode synthesis.
- Top/back plate meshing targets about two linear tetrahedra through thickness. A
  published study still needs a mesh-convergence argument for the observables used.
- Rayleigh `beta` is fixed. Damping ratios therefore vary with mode frequency and do
  not obey exact geometric-similarity scaling.
- Peak extraction is an operational label definition. Overlapping, broad, weak, or
  grid-edge resonances can be missed or assigned uncertain Q. Truncation is recorded
  when more than 32 peaks are eligible.

## Dataset

- Marginal parameter ranges are synthetic design choices and do not represent a
  measured population of manufactured instruments.
- Per-shape material banks increase diversity but do not create a complete crossed
  shape-material factorial design.
- The completion policy can publish successful cases while recording terminal
  failures. Downstream analyses must inspect missingness and avoid assuming a perfect
  rectangular design.
- v6 contains only 100 base contours. Ten bridges and many materials increase response
  samples but not the number of statistically independent outer shapes.
- Shape-disjoint splitting prevents direct contour leakage, but hyperparameter search
  can still overfit the validation set.

## Anchor Interpretation

- The included anchor planner is under active testing. This packaging pass does not
  certify its plan design or generated FEM output.
- Block A gives exact geometric scaling only for solids, apart from fixed Rayleigh
  damping. Hollow walls, back plates, and centre-block settings remain fixed in mm,
  so hollow scale cases are coverage points rather than similarity identities.
- Block B fixes characteristic area and explicit controls, not width, height, aspect,
  moments, or every contour-derived geometry scalar. It isolates contour more tightly
  than v6 but is not a mathematically pure one-variable intervention.
- Invalid Block B contours are rejected if they cannot accept the shared normalized
  bridge and soundhole templates. This can introduce a feasibility selection effect.
- Reusing source contours requires inherited splits. Changing those splits would
  create train/validation geometry leakage when v6 and anchor are combined.

## Neural Models

- Architecture selection is ongoing. The presence of a class or CLI option does not
  mean it has been validated as the final surrogate.
- The default raster is scale normalized; absolute scale reaches the model through
  width, height, and area scalars. This can encourage scalar shortcuts and is an
  explicit subject of the included ablations.
- The output is magnitude only. Phase, complex admittance, causal reconstruction, and
  audio rendering are not NN targets in this snapshot.
- Predictions outside the geometry/material support of the training and anchor plans
  are extrapolations. Plausible-looking spectra are not evidence of physical accuracy.

## Repository Snapshot

- No datasets, checkpoints, benchmark tables, plots, tests, or launch wrappers are
  included.
- No license is included yet. The code should not be publicly released until the
  intended license and third-party obligations are reviewed.
- Exact environment lock files and container images are not included. The supplied
  environment is a readable dependency specification, not a bitwise lock.
