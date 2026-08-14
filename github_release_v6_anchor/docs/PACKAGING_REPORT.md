# Packaging Report

Date: 2026-08-10

## Scope

This folder was assembled from the working research tree without editing the original
implementation files. The current anchor planner was copied as a work-in-progress
snapshot and was not behaviorally reviewed or executed in this pass.

## Included Source Integrity

- 27 Python implementation files are included.
- `plan_source_compatibility.json` and the pinned air-coupling theory note are also
  included.
- All 29 copied source/theory entries matched their working-tree originals by SHA-256
  at packaging time.
- All 18 files named by `dataset_gen_mixed._implementation_fingerprint()` are present
  in the release snapshot.
- Against the in-progress `mixed_v6_full` plan, the only working-tree fingerprint
  difference was `dataset_gen_mixed.py`; its exact old-to-new hash pair is the audited
  runtime-only run-policy transition in `plan_source_compatibility.json`.
- Exact hashes are stored in `SOURCE_SNAPSHOT_SHA256.json`.
- Public-only edits are limited to README, documentation, dependency specifications,
  `.gitignore`, and the hash manifest inside this release folder.

## Checks Performed

- All 27 Python files passed `python -m py_compile` with bytecode redirected outside
  the release directory.
- `dataset_gen_mixed.py --help` loaded successfully with Python UTF-8 mode.
- `backend/model_builder.py --help` loaded successfully with Python UTF-8 mode.
- All relative links in the new Markdown documentation resolve.
- The release tree contains no test/validator/launcher files selected by the packaging
  audit and no `.npz`, `.npy`, `.msh`, `.step`, checkpoint, image, or log artifacts.
- Total packaged size after documentation was approximately 1.1 MiB.

## Checks Not Performed

- No FEM solve, CAD build, mesh generation, dataset generation, or previous regression
  test was rerun.
- The work-in-progress anchor planner was not dry-run or otherwise validated.
- Full NN runtime import was not certified on the packaging host. Its existing native
  Windows PyTorch installation failed to load `torch/lib/shm.dll` (`WinError 127`).
  This is an environment failure rather than a Python syntax failure; verify training
  inside the intended Linux/WSL environment before publication.
- No trained checkpoint or dataset was available or included for end-to-end prediction
  verification.

## Before Public Release

1. Select a license and add citation metadata.
2. Create a locked Linux/WSL environment or container and record PETSc/SLEPc/MUMPS
   configuration.
3. Run the excluded numerical validation suite and publish convergence/reference
   evidence separately.
4. Freeze the NN architecture and anchor design, or label their reported results as
   exploratory.
5. Review all documentation against the exact commit used for paper results.
