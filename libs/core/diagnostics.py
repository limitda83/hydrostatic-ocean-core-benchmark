#########################################################################
#  Module: diagnostics                                                  #
#  Description: Accuracy and conservation diagnostics                   #
#               (docs/01_experiment_design.md S4.1): relative L2/Linf   #
#               errors, discrete mass and energy, and the modal         #
#               amplitude/phase extraction used for damping and         #
#               dispersion error.                                       #
#  Pipeline: schemes -> diagnostics -> metrics.json                     #
#########################################################################

from __future__ import annotations

import numpy as np

from libs.core.grid import CGrid
from libs.core.schemes import Physics, State


def l2_rel(numeric: np.ndarray, exact: np.ndarray) -> float:
    """Relative L2 error; falls back to absolute when the exact field is zero."""
    denom = float(np.sqrt(np.sum(exact * exact)))
    num = float(np.sqrt(np.sum((numeric - exact) ** 2)))
    return num / denom if denom > 0.0 else num


def linf_rel(numeric: np.ndarray, exact: np.ndarray) -> float:
    denom = float(np.max(np.abs(exact)))
    num = float(np.max(np.abs(numeric - exact)))
    return num / denom if denom > 0.0 else num


def total_mass(state: State, grid: CGrid) -> float:
    """M = sum(eta) * dA."""
    return float(np.sum(state.eta)) * grid.cell_area


def total_energy(state: State, grid: CGrid, physics: Physics) -> float:
    """E = 1/2 sum[ H (u^2 + v^2)|centre + g eta^2 ] dA (spec S2.2)."""
    u2 = 0.5 * (state.u**2 + np.roll(state.u, 1, axis=1) ** 2)
    v2 = 0.5 * (state.v**2 + np.roll(state.v, 1, axis=0) ** 2)
    density = physics.H * (u2 + v2) + physics.g * state.eta**2
    return 0.5 * float(np.sum(density)) * grid.cell_area


def modal_amplitude(field: np.ndarray, grid: CGrid, k: float, l: float,
                    variable: str = "eta") -> complex:
    """Complex amplitude A such that field ~ Re[ A exp(i(kx + ly)) ].

    Used to separate numerical damping (|A| ratio) from dispersion error
    (arg A difference) instead of lumping both into a single L2 number.
    """
    x, y = grid.coords(variable)
    basis = np.exp(-1j * (k * x + l * y))
    return complex(2.0 * np.sum(field * basis) / field.size)


def wave_errors(numeric: State, exact: State, grid: CGrid, k: float,
                l: float) -> dict[str, float]:
    """Amplitude ratio and phase error of the eta mode."""
    a_num = modal_amplitude(numeric.eta, grid, k, l, "eta")
    a_exa = modal_amplitude(exact.eta, grid, k, l, "eta")
    if abs(a_exa) == 0.0:
        return {"amp_ratio": float("nan"), "phase_err_rad": float("nan")}
    ratio = a_num / a_exa
    return {"amp_ratio": float(abs(ratio)),
            "phase_err_rad": float(np.angle(ratio))}


def summarize(numeric: State, exact: State, grid: CGrid, physics: Physics,
              m0: float, e0: float) -> dict[str, float]:
    """Assemble the standard diagnostic block for metrics.json.

    Mass drift is normalised by rms(eta) * domain area rather than by M0.
    Both verification cases are single Fourier modes with zero spatial mean,
    so M0 is ~0 and a relative-to-M0 drift would be meaningless.
    """
    mass = total_mass(numeric, grid)
    energy = total_energy(numeric, grid, physics)
    domain_area = grid.cell_area * grid.nx * grid.ny
    mass_scale = float(np.sqrt(np.mean(exact.eta ** 2))) * domain_area
    if mass_scale == 0.0:
        mass_scale = 1.0
    return {
        "mass_scale": mass_scale,
        "l2_rel_eta": l2_rel(numeric.eta, exact.eta),
        "l2_rel_u": l2_rel(numeric.u, exact.u),
        "l2_rel_v": l2_rel(numeric.v, exact.v),
        "linf_rel_eta": linf_rel(numeric.eta, exact.eta),
        "mass": mass,
        "energy": energy,
        "mass_drift": (mass - m0) / mass_scale,
        "energy_drift": (energy - e0) / abs(e0) if e0 != 0.0 else energy - e0,
    }
