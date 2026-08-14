"""
materials.py
------------
Orthotropic material database for guitar body FEM.
All constants in SI units (Pa, kg/m3).
Axes: 1=L (longitudinal/along grain), 2=R (radial), 3=T (tangential).
"""

MATERIALS = {
    "engelmann_spruce": {
        "E1": 9.79e9,  "E2": 1.25e9,  "E3": 0.58e9,
        "G12": 1.21e9, "G13": 1.17e9, "G23": 0.10e9,
        "nu12": 0.422, "nu13": 0.462, "nu23": 0.53,
        "density": 350.0,
    },
    "sitka_spruce": {
        "E1": 13.0e9,  "E2": 0.9e9,   "E3": 0.5e9,
        "G12": 1.50e9, "G13": 1.20e9, "G23": 0.06e9,
        "nu12": 0.372, "nu13": 0.435, "nu23": 0.47,
        "density": 440.0,
    },
    "maple": {
        "E1": 12.6e9,  "E2": 2.2e9,   "E3": 0.9e9,
        "G12": 1.63e9, "G13": 1.63e9, "G23": 0.26e9,
        "nu12": 0.424, "nu13": 0.476, "nu23": 0.36,
        "density": 600.0,
    },
    "mahogany": {
        "E1": 10.1e9,  "E2": 1.5e9,   "E3": 0.5e9,
        "G12": 1.32e9, "G13": 1.10e9, "G23": 0.18e9,
        "nu12": 0.300, "nu13": 0.280, "nu23": 0.35,
        "density": 545.0,
    },
}


def get_material(name: str) -> dict:
    """Return a copy of the material dict. Raises ValueError if not found."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in MATERIALS:
        available = ", ".join(MATERIALS.keys())
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    return dict(MATERIALS[key])


def list_materials() -> None:
    print("Available materials:")
    for name, mat in MATERIALS.items():
        print(f"  {name:<22}  rho={mat['density']:>5.0f} kg/m3  "
              f"E_L={mat['E1']/1e9:.2f} GPa  "
              f"E_R={mat['E2']/1e9:.3f} GPa  "
              f"E_T={mat['E3']/1e9:.3f} GPa")


if __name__ == "__main__":
    list_materials()
