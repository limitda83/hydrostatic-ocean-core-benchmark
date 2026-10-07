#########################################################################
#  Module: cases                                                        #
#  Description: Verification cases with exact solutions                 #
#               (docs/03_discretization_spec.md S5):                    #
#                 V1 igw          - inertia-gravity wave, exact in the  #
#                                   continuum, doubly periodic          #
#                 V2 geo_balance  - geostrophic balance that is an      #
#                                   EXACT steady state of the DISCRETE  #
#                                   operators, built spectrally         #
#  Pipeline: grid/physics -> cases -> driver (init + exact comparison)  #
#########################################################################

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from libs.core.grid import CGrid
from libs.core.operators import (avg_u_to_v, avg_v_to_u, div_eta, gradx_u,
                                 grady_v)
from libs.core.schemes import Physics, State

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseSpec:
    """Resolved description of a verification case."""

    name: str
    mode_x: int
    mode_y: int
    eta0: float
    t_final: float
    period: float          # characteristic time scale [s] (wave period, or f^-1)
    has_exact: bool


class InertiaGravityWave:
    """V1: linear inertia-gravity wave on an f-plane, doubly periodic.

    omega^2 = f^2 + g H (k^2 + l^2); the fields below satisfy the continuous
    linear rotating shallow water equations exactly, so the model error is
    pure truncation error and converges at the scheme order.
    """

    name = "igw"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics, mode_x: int, mode_y: int,
                 eta0: float) -> None:
        self.grid, self.physics, self.eta0 = grid, physics, eta0
        self.k = 2.0 * np.pi * mode_x / grid.Lx
        self.l = 2.0 * np.pi * mode_y / grid.Ly
        self.kappa2 = self.k**2 + self.l**2
        self.omega = float(np.sqrt(physics.f**2 + physics.g * physics.H * self.kappa2))
        self.period = 2.0 * np.pi / self.omega

    def exact(self, t: float) -> State:
        g_, f, H = self.physics.g, self.physics.f, self.physics.H
        amp = self.eta0 / (H * self.kappa2)

        x_e, y_e = self.grid.coords("eta")
        x_u, y_u = self.grid.coords("u")
        x_v, y_v = self.grid.coords("v")

        ph_e = self.k * x_e + self.l * y_e - self.omega * t
        ph_u = self.k * x_u + self.l * y_u - self.omega * t
        ph_v = self.k * x_v + self.l * y_v - self.omega * t

        eta = self.eta0 * np.cos(ph_e)
        u = amp * (self.omega * self.k * np.cos(ph_u) - f * self.l * np.sin(ph_u))
        v = amp * (self.omega * self.l * np.cos(ph_v) + f * self.k * np.sin(ph_v))
        return State(u=u, v=v, eta=eta, t=t)

    def initial(self) -> State:
        return self.exact(0.0)


class InertiaGravityWaveBroadband:
    """V1b: superposition of many inertia-gravity wave modes.

    The equations are linear, so a sum of exact single-mode solutions is
    itself exact - the analytic solution is retained while the field becomes
    broadband. That matters because the single-mode cases make the
    preconditioned Helmholtz right-hand side an eigenvector of the operator,
    collapsing the Krylov space to one dimension and letting PCG converge in
    1-2 iterations. The elliptic solve, which is the expensive and least
    GPU-friendly part of a semi-implicit scheme, was therefore barely being
    exercised (docs/11 section 5.1).

    Mode phases come from integer arithmetic, not an RNG, so every backend
    builds bit-identical initial conditions.
    """

    name = "igw_broadband"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics, n_modes: int, eta0: float,
                 slope: float) -> None:
        self.grid, self.physics, self.eta0 = grid, physics, eta0
        modes = [(m, n) for m in range(1, n_modes + 1) for n in range(1, n_modes + 1)]

        k = np.array([2.0 * np.pi * m / grid.Lx for m, _ in modes])
        l = np.array([2.0 * np.pi * n / grid.Ly for _, n in modes])
        kappa2 = k**2 + l**2
        omega = np.sqrt(physics.f**2 + physics.g * physics.H * kappa2)
        # Red amplitude spectrum a ~ kappa^-slope, then normalised so that
        # rms(eta) = eta0 regardless of the mode count.
        raw = np.array([(m * m + n * n) ** (-0.5 * slope) for m, n in modes])
        amp = raw * eta0 / np.sqrt(np.sum(raw**2) / 2.0)
        # Deterministic phases: pure integer arithmetic, identical everywhere.
        phase = np.array([2.0 * np.pi * ((m * 37 + n * 17) % 101) / 101.0
                          for m, n in modes])

        self.modes = modes
        self.k, self.l, self.kappa2 = k, l, kappa2
        self.omega, self.amp, self.phase = omega, amp, phase
        self.period = 2.0 * np.pi / float(np.min(omega))   # slowest mode

    def exact(self, t: float) -> State:
        g_, f, H = self.physics.g, self.physics.f, self.physics.H
        x_e, y_e = self.grid.coords("eta")
        x_u, y_u = self.grid.coords("u")
        x_v, y_v = self.grid.coords("v")

        eta = np.zeros_like(x_e)
        u = np.zeros_like(x_u)
        v = np.zeros_like(x_v)
        for idx in range(len(self.modes)):
            k, l, om = self.k[idx], self.l[idx], self.omega[idx]
            a, ph = self.amp[idx], self.phase[idx]
            scale = a / (H * self.kappa2[idx])
            pe = k * x_e + l * y_e - om * t + ph
            pu = k * x_u + l * y_u - om * t + ph
            pv = k * x_v + l * y_v - om * t + ph
            eta += a * np.cos(pe)
            u += scale * (om * k * np.cos(pu) - f * l * np.sin(pu))
            v += scale * (om * l * np.cos(pv) + f * k * np.sin(pv))
        return State(u=u, v=v, eta=eta, t=t)

    def initial(self) -> State:
        return self.exact(0.0)


class GeostrophicBalance:
    """V2: single Fourier mode in EXACT discrete geostrophic balance.

    Solving the steady discrete momentum equations in Fourier space gives
        v_hat = (4g/f) (e^{i tx} - 1) / ( dx (1 + e^{-i ty})(1 + e^{i tx}) ) * eta_hat
        u_hat = -(4g/f) (e^{i ty} - 1) / ( dy (1 + e^{-i tx})(1 + e^{i ty}) ) * eta_hat
    with tx = 2 pi m / nx, ty = 2 pi n / ny. This state is divergence free in
    the discrete sense too, so the exact solution for all t is the initial
    state; any drift is purely numerical.
    """

    name = "geo_balance"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics, mode_x: int, mode_y: int,
                 eta0: float) -> None:
        if physics.f == 0.0:
            raise ValueError("geo_balance requires a non-zero Coriolis parameter")
        self.grid, self.physics, self.eta0 = grid, physics, eta0
        self.period = 2.0 * np.pi / abs(physics.f)   # inertial period

        tx = 2.0 * np.pi * mode_x / grid.nx
        ty = 2.0 * np.pi * mode_y / grid.ny
        if np.isclose(abs(np.cos(tx / 2.0)), 0.0) or np.isclose(abs(np.cos(ty / 2.0)), 0.0):
            raise ValueError("geo_balance mode hits the grid Nyquist limit; "
                             "the averaging operator is singular there")

        ex, ey = np.exp(1j * tx), np.exp(1j * ty)
        fac = 4.0 * physics.g / physics.f
        self._v_hat = fac * (ex - 1.0) / (grid.dx * (1.0 + np.conj(ey)) * (1.0 + ex)) * eta0
        self._u_hat = -fac * (ey - 1.0) / (grid.dy * (1.0 + np.conj(ex)) * (1.0 + ey)) * eta0

        ii, jj = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), indexing="xy")
        self._phase = np.exp(1j * (tx * ii + ty * jj))

    def initial(self) -> State:
        eta = self.eta0 * np.real(self._phase)
        u = np.real(self._u_hat * self._phase)
        v = np.real(self._v_hat * self._phase)
        state = State(u=u, v=v, eta=eta, t=0.0)
        self._assert_balanced(state)
        return state

    def exact(self, t: float) -> State:
        init = State(u=np.real(self._u_hat * self._phase),
                     v=np.real(self._v_hat * self._phase),
                     eta=self.eta0 * np.real(self._phase), t=t)
        return init

    def _assert_balanced(self, s: State) -> None:
        """Verify the discrete steady-state residuals are at round-off."""
        g_, f = self.physics.g, self.physics.f
        res_u = f * avg_v_to_u(s.v) - g_ * gradx_u(s.eta, self.grid)
        res_v = -f * avg_u_to_v(s.u) - g_ * grady_v(s.eta, self.grid)
        res_e = div_eta(s.u, s.v, self.grid)
        scale = g_ * np.max(np.abs(gradx_u(s.eta, self.grid))) + 1e-300
        worst = max(np.max(np.abs(res_u)), np.max(np.abs(res_v))) / scale
        div_scale = np.max(np.abs(s.u)) / self.grid.dx + 1e-300
        LOGGER.debug(f"geo_balance discrete residuals: momentum={worst:.3e}, "
                     f"divergence={np.max(np.abs(res_e)) / div_scale:.3e}")
        if worst > 1e-10:
            raise AssertionError(
                f"geo_balance is not an exact discrete steady state "
                f"(relative momentum residual {worst:.3e}); the spectral "
                f"symbols in libs/core/cases.py disagree with libs/core/operators.py"
            )


def build_case(name: str, grid: CGrid, physics: Physics, config):
    """Instantiate a verification case from config/cases.toml."""
    section = config.section(f"case.{name}")
    if name == "igw":
        case = InertiaGravityWave(grid, physics, int(section["mode_x"]),
                                  int(section["mode_y"]), float(section["eta0"]))
        t_final = float(section["n_periods"]) * case.period
    elif name == "igw_broadband":
        case = InertiaGravityWaveBroadband(grid, physics, int(section["n_modes"]),
                                           float(section["eta0"]),
                                           float(section["slope"]))
        t_final = float(section["n_periods"]) * case.period
    elif name == "geo_balance":
        case = GeostrophicBalance(grid, physics, int(section["mode_x"]),
                                  int(section["mode_y"]), float(section["eta0"]))
        t_final = float(section["n_days"]) * 86400.0
    else:
        raise ValueError(f"unknown case '{name}' "
                         f"(implemented: igw|igw_broadband|geo_balance)")

    spec = CaseSpec(name=name, mode_x=int(section.get("mode_x", 0)),
                    mode_y=int(section.get("mode_y", 0)), eta0=float(section["eta0"]),
                    t_final=t_final, period=case.period, has_exact=case.has_exact)
    return case, spec
