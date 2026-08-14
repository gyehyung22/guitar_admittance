# Internal air-cavity coupling — theory & sign conventions (REFERENCE)

This file is the single source of truth for the time convention, block-matrix
coefficients, and interface signs used by every air-coupling module
(`acoustic_helmholtz.py`, `air_acoustics.py`, `fsi_coupling.py`,
`modal_coupled_admittance.py`, `fenics_admittance_coupled.py`). Module code must
match this document, not the other way round. **Signs/coefficients are pinned
here, not left to qualitative "added-mass looks plausible" checks.**

## Time convention

Harmonic convention **`exp(+iωt)`** everywhere. Therefore
`∂/∂t → +iω`, `∂²/∂t² → −ω²`. This is stated at the top of each module.

## Domains (3A conformal multi-domain)

- `Ω_s` = wood / structure volume (tet cells, physical `SOLID_WOOD`).
- `Ω_a` = internal air cavity volume ONLY (tet cells, physical `AIR_INTERNAL`).
  Excludes wall thickness, braces, blocks, bridge — all wood.
- `Ω_s ∩ Ω_a = ∅`. They share ONLY the interface surface `Γ_sa`
  (physical `FSI_TOP_INNER`, `FSI_BACK_INNER`, `FSI_SIDE_INNER`).
- `Γ_h` = soundhole footprint (physical `SOUNDHOLE`), the open part of the air
  boundary (no wood on the other side).
- Solid and air are DIFFERENT cells produced by an OCC BooleanFragments call so
  the interface triangulation is shared (conformal).

### Normal convention

`n_a` = outward unit normal of the AIR domain on `Γ_sa` (points from air into
wood). `n_s = −n_a` = outward structural normal there. All coupling integrals
below use `n_a`.

## Structural subproblem (existing, unchanged)

    Z_s(ω) u = F + f_coupling
    Z_s(ω) = K_s − ω² M_s + iω C_s         (C_s = α M_s + β K_s, Rayleigh)

`u` nodal displacement (3 dof/node), `F` the unit bridge load, `Y_b = iω u_z(bridge)/F`.

## Acoustic subproblem (pressure form)

Helmholtz in the cavity, `k = ω/c`, `c` = speed of sound in air, `ρ0` = air density:

    ∇²p + k² p = 0    in Ω_a

Weak form with test `q` (FE matrices, real, frequency-independent):

    K_a = ∫_{Ω_a} ∇p·∇q dΩ
    M_a = ∫_{Ω_a} p q dΩ
    Z_a(ω) = K_a − (ω²/c²) M_a

(Note `M_a` here is the plain mass matrix `∫pq`; the `1/c²` lives in `Z_a`.)

## Interface conditions (Everstine unsymmetric u–p form)

Reference: Everstine-type unsymmetric pressure–displacement FSI.

1. **Pressure traction on structure:** `σ(u)·n_s = −p n_s`.
2. **Moving-wall acoustic Neumann** (from `ρ0 ∂v/∂t = −∇p`, `v = ∂u/∂t`, with
   `exp(iωt)` so `∂²/∂t² → −ω²`):

       ∂p/∂n_a = ρ0 ω² (u · n_a)

### Coupling matrix (THE definition — used identically by all modules)

    G_{j i} = ∫_{Γ_sa} N_j^p (N_i^u · n_a) dS

(`N^p` scalar pressure shape fns; `N^u` vector displacement shape fns; `n_a` air
outward normal.) `G` has shape `(n_p, n_u)`.

### Derivation of the two off-diagonal blocks

- Structural pressure load (virtual work, `n_s = −n_a`):
  `∫_Γ (−p n_s)·v dS = +∫_Γ p (v·n_a) dS = (Gᵀ p)` → moved to LHS as `−Gᵀ p`.
- Acoustic moving-wall term (RHS of weak acoustic eq):
  `∫_Γ (∂p/∂n_a) q dS = ρ0 ω² ∫_Γ (u·n_a) q dS = ρ0 ω² G u` → LHS `−ρ0 ω² G u`.

### Full coupled block system (pinned)

    ⎡ Z_s(ω)        −Gᵀ      ⎤ ⎡u⎤   ⎡F⎤
    ⎢                        ⎥ ⎢ ⎥ = ⎢ ⎥
    ⎣ −ρ0 ω² G       Z_a(ω)' ⎦ ⎣p⎦   ⎣0⎦

This is **unsymmetric** (the `−Gᵀ` vs `−ρ0ω² G` asymmetry is physical for the
u–p form; a symmetric Morand–Ohayon form exists but we anchor to Everstine).
`Z_a'` = `Z_a` plus the soundhole port term below.

## Soundhole boundary conditions

`--soundhole-bc` selects the `Γ_h` treatment:

- `impedance` (**DEFAULT**): nonlocal lumped port (below). Models neck inertance
  + radiation → produces the A0/Helmholtz resonance.
- `throat`: explicit short throat volume + end correction; `pressure_release`
  only at the FAR end of the extended throat.
- `pressure_release` (**DEBUG ONLY**): `p = 0` directly on `Γ_h`. Removes neck
  inertance/radiation → generally does NOT reproduce A0. Never the default.
- `closed` (**DEBUG ONLY**): rigid wall `∂p/∂n = 0` on `Γ_h` (sealed cavity).

### Nonlocal lumped impedance port (rank-1, default)

Piston assumption over the hole (uniform `v_n`). Average pressure and total
volume velocity:

    p_bar = (1/S) ∫_{Γ_h} p dS ,   U_h = ∫_{Γ_h} v_n dS ,   p_bar = Z_h(ω) U_h

with acoustic inertance and impedance

    M_h   = ρ0 L_eff / S
    Z_h(ω) = R_rad(ω) + iω M_h          (R_rad: radiation resistance, small)

Let `b_j = ∫_{Γ_h} N_j^p dS` (so `∫_{Γ_h} p dS = bᵀ p`, `S = Σ_j b_j`). Using the
acoustic Neumann `∂p/∂n_a = −iωρ0 v_n` and `U_h = p_bar/Z_h`, the soundhole
boundary integral contributes a **rank-1, nonlocal** term to the acoustic block:

    Z_a'(ω) = Z_a(ω) + [ iωρ0 / (S² Z_h(ω)) ] · (b bᵀ)

This is the static-condensation of an explicit scalar port DOF `U_h` and is NOT a
local `p=0`. With `Z_h = iωM_h` (lossless) the term reduces to a real positive
rank-1 stiffness `ρ0/(S² M_h)·b bᵀ = 1/(S L_eff)·... ` that, against the cavity
compliance in `Z_a`, sets the Helmholtz resonance.

## A0 / Helmholtz analytic estimate

For a circular soundhole of physical thickness `t_hole` (top-plate thickness):

    S      = soundhole area
    V      = cavity volume
    r_eff  = sqrt(S/π)
    L_eff  = t_hole + 0.85·r_eff + 0.85·r_eff      (two end corrections)
    f_H    = (c / 2π) · sqrt( S / (V · L_eff) )

Stored per hollow sample: `cavity_volume, soundhole_area,
effective_soundhole_radius, effective_neck_length, estimated_helmholtz_hz`.

Validation gate (simple rigid box + circular hole benchmark): numerically
observed A0 (via the impedance port harmonic response) within **2–3 %** of `f_H`.
Real guitar geometry: report deviation, do not require 2–3 % (wall motion +
geometry complicate it). `pressure_release` on the raw hole face is expected to
FAIL this and is labeled debug.

## Air-only eigenmodes caveat

With the impedance port, the air eigenproblem is nonlinear in ω (because
`Z_h(ω)`). First pass therefore:
- `closed` and `pressure_release` give LINEAR generalized eigenproblems
  `K_a φ = λ M_a φ`, `λ = (ω/c)²`, used as DIAGNOSTICS only.
- A0 is obtained from the **harmonic** response of `Z_a'(ω)` with the impedance
  port, NOT from `pressure_release` eigenmodes. Do not claim `pressure_release`
  eigenmodes are A0.

## Physical constants (air, 20 °C)

    c   = 343.0 m/s
    ρ0  = 1.204 kg/m³

(Overridable per call; defaults documented in `acoustic_helmholtz.py`.)
