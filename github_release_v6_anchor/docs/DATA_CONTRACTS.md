# v6 and Anchor Data Contracts

This snapshot contains the `mixed-v6` production contract and the current
work-in-progress `plan_anchor_dataset.py`. Anchor behavior is still under testing;
its inclusion is for review, not a certification. Generated data is excluded.

## 1. Immutable Plan

`dataset_plan.json` is written before FEM starts. Its canonical body contains the
generator configuration, production contract, base shapes, per-shape materials,
frequency grid, shape-level splits, case definitions, shape digests, and source
fingerprints. `plan_hash` is the SHA-256 digest of the canonical JSON body.

Default stable identifiers are derived from semantic identities, not completion
order:

```text
shape_id       base contour and body realization
material_id    material_<base-shape>_<slot>
case_id        shape x body type x material
sample_id      case x bridge index
```

One case contains ten bridge points. One supervised sample selects one bridge row
from that shared case artifact.

An output directory is not a mutable configuration workspace. Reusing it with a
different plan is an error. Runtime scheduling, including shape order and deferred
material slots, is separate from the physical plan.

## 2. Default v6 Contract

```text
generator                  mixed-v6
body types                 solid, hollow
frequency grid             geomspace(20, 5000, 500) Hz
time convention            exp(+i omega t)
force                      1 N in global Z
admittance                 m/s/N
main target                dB re 1 m/s/N
peak slots                 32
material axes              L=+Y, R=+X, T=+Z
solid solver               structural-full-harmonic-peaks-v1
hollow solver              modal-coupled-real-basis-v2
output schema              magnitude-peaks-v1
```

Shape defaults are 100 base contours, 10 materials per contour, two body types, and
10 bridge points. This produces 2,000 FEM cases and 20,000 bridge samples before any
terminal failures.

## 3. Output Layout

A completed or partially generated root has the following conceptual layout:

```text
dataset_root/
  dataset_plan.json
  manifest.csv
  dataset_ready.json                 # published only at terminal completion
  dataset_artifact_inventory.json
  shapes/
    shape_NNNN/
      contour.npy
      model.step
      mesh.msh
      geometry.json
      ...
  cases/
    <case_id>/
      .committed
      case_response.npz              # canonical NN-facing artifact
      case_admittance.npz            # native complex case response
      peak_labels.npz
      case_meta.json
      timing.json
      solve_spec.json
      reduced_model.npz              # hollow
      solid_full_eigen.npz           # solid
      ... solver provenance ...
  samples/
    <sample_id>/
      .committed
      admittance.npz                 # one bridge, complex response
      sample_params.json
  case_status/
    <case_id>.json
  failed_attempts/
    ...
```

The marker `.committed` is an atomic-transaction marker, not an alternate JSON file
format. Staging directories do not become valid cases or samples until QC passes and
the marker is written.

## 4. Canonical Case Response

`case_response.npz` is the common solid/hollow NN artifact:

| Array | Shape | Meaning |
|---|---:|---|
| `schema_version` | scalar | `magnitude-peaks-v1` |
| `frequencies_hz` | `(500,)` | common log-frequency grid |
| `log_magnitude_db` | `(B, 500)` | `20 log10 |Y|` for B bridges |
| `peak_frequency_hz` | `(B, 32)` | labelled peak frequencies |
| `peak_amplitude_db` | `(B, 32)` | labelled peak amplitudes |
| `peak_q` | `(B, 32)` | quality factors |
| `peak_mask` | `(B, 32)` | valid packed slots |
| `peak_count_total` | `(B,)` | eligible count before truncation |
| `peak_truncated` | `(B,)` | eligible count exceeded 32 |

Peak slots are selected by descending amplitude and then stored in ascending
frequency. Invalid padded slots are zero and must be ignored through `peak_mask`.

`case_admittance.npz` retains the complex response. Hollow cases additionally store
mean soundhole pressure `p_bar` and port volume flow `U_h`.

## 5. Manifest Contract

`manifest.csv` has one row per bridge sample. It carries:

- sample, case, shape, material, split, and bridge identifiers;
- all ten material constants;
- requested and snapped bridge coordinates;
- paths to the shared case response, native sample response, parameters, contour,
  CAD, mesh, and case directory;
- thickness, top plate, cavity volume, and soundhole features;
- solver and source revisions, plan and artifact hashes;
- modal coverage, eigen residual, basis orthonormality/conditioning, timing, attempt,
  and QC state.

The manifest is regenerated from committed status records. It is never treated as an
append-only source of truth.

## 6. NN Input Contract

### 6.1 Shape tensor

Default shape input:

```text
shape: (1, 96, 96), binary occupancy
```

The isotropic frame is centred on the contour bounding box and has span
`1.1 * max(width, height)`. Therefore the image does not encode absolute scale.
Optional experiments replace occupancy with SDF or add a soundhole channel.

### 6.2 Standard scalar tensor

The 30-dimensional standard vector concatenates:

```text
material (10)
  E_L, E_R, E_T, G_LR, G_LT, G_RT,
  nu_LR, nu_LT, nu_RT, density

bridge (6)
  x_mm, y_mm, z_mm, x_norm, y_norm, z/thickness

geometry (12)
  thickness, top_plate_thickness_mm, cavity_volume_m3,
  soundhole_diameter, soundhole_area_m2,
  soundhole_center_x_mm, soundhole_center_y_mm,
  width_mm, height_mm, area_mm2,
  soundhole_center_x_norm, soundhole_center_y_norm

body type (2)
  is_solid, is_hollow
```

Solid rows use zeros for all cavity and soundhole quantities. They do not encode a
fictitious hole at `(0, 0)`.

Relational geometry is optional and separate. It does not alter this 30-value
checkpoint contract.

### 6.3 Targets

```text
spectrum:        (500,) float32 dB
peak frequency:  (32,)
peak amplitude:  (32,)
peak Q:          (32,)
peak mask:       (32,) bool
```

Normalization is fitted on training records only and saved beside the model run.
Validation and test records never contribute statistics.

## 7. Split Contract

The plan assigns a split to each base geometry. Every body type, material, and bridge
for that geometry inherits it. The default proportions are 70 percent train, 15
percent validation, and 15 percent test.

When multiple datasets are loaded, contour digests are checked across plans. The
loader refuses to combine the same contour under different split labels.

## 8. Anchor Compatibility

The anchor planner constructs a plan body compatible with the v6 generator. It does
not define another response schema or another NN tensor layout.

During planning, `_anchor` metadata records block, source geometry, scale, shared
placement, and source split. The generator's canonical plan serialization retains
only geometry-defining fields, so `plan_anchor_dataset.py` persists the block and
anchor maps in the companion `anchor_design.json`. `--emit-blocks PATH` can recover
the same deterministic map for a plan created before that companion metadata was
added. These labels support controlled analysis and do not change the FEM equations.

The source v6 plan is required because anchor contours, materials, and split
provenance are intentionally derived from it. No source plan or generated anchor
dataset is included in this code-only snapshot.

## 9. Readiness and Terminal Failures

`dataset_ready.json` certifies that every planned case has reached a terminal state
under the configured completion policy and that the manifest/inventory are coherent.
It does not assert that every case succeeded.

The v6 policy may drop terminally failed cases. Published metadata records planned,
successful, failed, and sampled counts. Consumers requiring a complete rectangular
factorial design must inspect those counts rather than relying only on the presence
of the certificate.
