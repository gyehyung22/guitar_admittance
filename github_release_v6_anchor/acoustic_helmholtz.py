"""
acoustic_helmholtz.py
---------------------
Analytic Helmholtz (A0) estimate and the lumped soundhole-port impedance used by
the internal air-cavity coupling.  Pure-numpy, no FEniCSx — runnable and
unit-testable anywhere.

Time convention: exp(+iωt)  (so ∂/∂t → +iω).  See docs/air_coupling_theory.md
for the full block formulation and sign conventions; this module implements the
soundhole port and the f_H estimate documented there.

A0 / Helmholtz:
    r_eff = sqrt(S/π)
    L_eff = t_hole + 0.85 r_eff + 0.85 r_eff
    f_H   = (c / 2π) sqrt( S / (V L_eff) )

Lumped nonlocal port (default soundhole BC):
    M_h    = ρ0 L_eff / S                      (acoustic inertance)
    Z_h(ω) = R_rad(ω) + iω M_h                 (acoustic impedance)
    rank-1 acoustic-block term coefficient:
             κ(ω) = iωρ0 / (S² Z_h(ω))         multiplies (b bᵀ)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Air at ~20 °C (overridable per call).
AIR_C = 343.0       # speed of sound [m/s]
AIR_RHO0 = 1.204    # density [kg/m³]

# End-correction coefficient per open end of a circular aperture (flanged ≈ 0.85).
END_CORRECTION = 0.85


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def effective_radius(soundhole_area: float) -> float:
    """r_eff = sqrt(S/π) for an equivalent circular hole of area S [m²]."""
    return float(np.sqrt(max(soundhole_area, 0.0) / np.pi))


def effective_neck_length(t_hole: float, soundhole_area: float,
                          n_open_ends: int = 2) -> float:
    """L_eff = t_hole + n_open_ends · 0.85 · r_eff.

    n_open_ends=2 models a hole open to the cavity on one side and exterior on the
    other (two flanged ends), the standard first approximation.
    """
    r_eff = effective_radius(soundhole_area)
    return float(t_hole + n_open_ends * END_CORRECTION * r_eff)


# ---------------------------------------------------------------------------
# Helmholtz estimate
# ---------------------------------------------------------------------------

@dataclass
class HelmholtzEstimate:
    cavity_volume: float            # V [m³]
    soundhole_area: float           # S [m²]
    soundhole_thickness: float      # t_hole [m]
    effective_soundhole_radius: float   # r_eff [m]
    effective_neck_length: float    # L_eff [m]
    acoustic_inertance: float       # M_h [kg/m⁴]
    estimated_helmholtz_hz: float   # f_H [Hz]
    speed_of_sound: float
    air_density: float

    def to_dict(self) -> dict:
        return asdict(self)


def helmholtz_estimate(cavity_volume: float, soundhole_area: float,
                       t_hole: float, c: float = AIR_C,
                       rho0: float = AIR_RHO0,
                       n_open_ends: int = 2) -> HelmholtzEstimate:
    """Analytic A0 estimate for a Helmholtz resonator (cavity + circular neck).

    All lengths in metres.  Returns a HelmholtzEstimate dataclass.  If the cavity
    or hole is degenerate (V<=0 or S<=0) f_H is set to 0 (no resonance).
    """
    S = float(soundhole_area)
    V = float(cavity_volume)
    r_eff = effective_radius(S)
    L_eff = effective_neck_length(t_hole, S, n_open_ends)
    M_h = acoustic_inertance(L_eff, S, rho0)

    if V > 0 and S > 0 and L_eff > 0:
        f_H = (c / (2.0 * np.pi)) * np.sqrt(S / (V * L_eff))
    else:
        f_H = 0.0

    return HelmholtzEstimate(
        cavity_volume=V, soundhole_area=S, soundhole_thickness=float(t_hole),
        effective_soundhole_radius=r_eff, effective_neck_length=L_eff,
        acoustic_inertance=M_h, estimated_helmholtz_hz=float(f_H),
        speed_of_sound=c, air_density=rho0,
    )


# ---------------------------------------------------------------------------
# Lumped soundhole-port impedance
# ---------------------------------------------------------------------------

def acoustic_inertance(L_eff: float, soundhole_area: float,
                       rho0: float = AIR_RHO0) -> float:
    """M_h = ρ0 L_eff / S  [kg/m⁴]."""
    S = float(soundhole_area)
    if S <= 0:
        return float("inf")
    return float(rho0 * L_eff / S)


def radiation_resistance(omega, soundhole_area: float, c: float = AIR_C,
                         rho0: float = AIR_RHO0):
    """Low-ka piston radiation resistance R_rad(ω) = ρ0 c k² / (2π) per port.

    Small near A0 (lightly damped); included so Z_h is not purely reactive.
    `omega` may be scalar or array; returns matching shape.
    """
    omega = np.asarray(omega, dtype=float)
    k = omega / c
    # Specific radiation resistance of a piston (one side), Rayleigh low-ka limit:
    #   Z_rad ≈ ρ0 c (ka)²/2 / S  (acoustic ohms).  Here as acoustic impedance:
    return rho0 * c * (k ** 2) / (2.0 * np.pi)


def port_impedance(omega, M_h: float, soundhole_area: float = None,
                   c: float = AIR_C, rho0: float = AIR_RHO0,
                   include_radiation: bool = True):
    """Z_h(ω) = R_rad(ω) + iω M_h  [acoustic ohms].

    omega scalar or array → complex of same shape.
    """
    omega = np.asarray(omega, dtype=float)
    R = (radiation_resistance(omega, soundhole_area, c, rho0)
         if (include_radiation and soundhole_area) else 0.0)
    return R + 1j * omega * M_h


def port_term_coefficient(omega, soundhole_area: float, Z_h,
                          rho0: float = AIR_RHO0):
    """Rank-1 acoustic-block coefficient κ(ω) = iωρ0 / (S² Z_h(ω)).

    The soundhole port contributes  κ(ω) · (b bᵀ)  to the acoustic block Z_a',
    where b_j = ∫_{Γ_h} N_j^p dS (so S = Σ b_j).  See theory doc §"port".
    omega scalar or array; Z_h matching shape.
    """
    omega = np.asarray(omega, dtype=float)
    S = float(soundhole_area)
    Z_h = np.asarray(Z_h, dtype=complex)
    return 1j * omega * rho0 / (S ** 2 * Z_h)


# ---------------------------------------------------------------------------
# Lumped 0-D Helmholtz transfer (benchmark cross-check, no FEM)
# ---------------------------------------------------------------------------

def lumped_cavity_pressure(omega, cavity_volume: float, soundhole_area: float,
                           t_hole: float, c: float = AIR_C, rho0: float = AIR_RHO0,
                           n_open_ends: int = 2):
    """0-D Helmholtz resonator average-pressure response to unit volume velocity.

    A purely-lumped sanity model (NOT the FEM): cavity acoustic compliance
    C_a = V/(ρ0 c²) in series-resonance with neck inertance M_h gives a peak at
    f_H.  Used to cross-check the analytic f_H and the FE port harmonic A0.
    Returns complex p_bar/U_source(ω).
    """
    omega = np.asarray(omega, dtype=float)
    est = helmholtz_estimate(cavity_volume, soundhole_area, t_hole, c, rho0, n_open_ends)
    M_h = est.acoustic_inertance
    C_a = cavity_volume / (rho0 * c ** 2)          # acoustic compliance
    R = radiation_resistance(omega, soundhole_area, c, rho0)
    # p across cavity compliance for a parallel C_a with neck (M_h,R) to outside:
    #   Y = iωC_a + 1/(R + iωM_h);  p_bar = U / Y
    Y = 1j * omega * C_a + 1.0 / (R + 1j * omega * M_h)
    return 1.0 / Y


if __name__ == "__main__":
    # Quick demo for a small box benchmark
    V = 0.002      # 2 L cavity
    S = np.pi * (0.02 ** 2)   # 20 mm radius hole
    t = 0.003      # 3 mm plate
    est = helmholtz_estimate(V, S, t)
    print("Helmholtz estimate:")
    for k, v in est.to_dict().items():
        print(f"  {k:32s} {v}")
