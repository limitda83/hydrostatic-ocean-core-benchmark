#########################################################################
#  Module: cases3d                                                      #
#  Description: 3D verification cases of docs/03_discretization_spec.md #
#               S7.6:                                                   #
#                 V3D-1 barotropic3d   - exact reduction to the 2D case #
#                 V3D-2 vdiffusion     - exact in space, so the         #
#                                        temporal order of the batched  #
#                                        tridiagonal solve is isolated  #
#                 V3D-3 baroclinic_igw - exact internal wave mode       #
#  Pipeline: grid/physics -> cases3d -> driver3d                        #
#########################################################################

from __future__ import annotations

import numpy as np

from libs.core.grid import CGrid
from libs.core.model3d import Physics3D, State3D


class Barotropic3D:
    """V3D-1: the 2D inertia-gravity wave replicated over nz layers.

    With nu = 0 and b = 0 the 3D algorithm must collapse onto the 2D one, so
    this is the strongest available check of the whole free-surface path -
    the tridiagonal solves, the effective-depth profile q, the Helmholtz
    coefficient and the back-substitution all have to be exactly right for
    the answer to match the 2D model to round-off.
    """

    name = "barotropic3d"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics3D, mode_x: int, mode_y: int,
                 eta0: float) -> None:
        self.grid, self.physics, self.eta0 = grid, physics, eta0
        self.k = 2.0 * np.pi * mode_x / grid.Lx
        self.l = 2.0 * np.pi * mode_y / grid.Ly
        self.kappa2 = self.k**2 + self.l**2
        self.omega = float(np.sqrt(physics.f**2
                                   + physics.g * physics.H * self.kappa2))
        self.period = 2.0 * np.pi / self.omega

    def exact(self, t: float) -> State3D:
        g_, f, H = self.physics.g, self.physics.f, self.physics.H
        amp = self.eta0 / (H * self.kappa2)
        x_e, y_e = self.grid.coords("eta")
        x_u, y_u = self.grid.coords("u")
        x_v, y_v = self.grid.coords("v")
        pe = self.k * x_e + self.l * y_e - self.omega * t
        pu = self.k * x_u + self.l * y_u - self.omega * t
        pv = self.k * x_v + self.l * y_v - self.omega * t

        eta = self.eta0 * np.cos(pe)
        u2 = amp * (self.omega * self.k * np.cos(pu) - f * self.l * np.sin(pu))
        v2 = amp * (self.omega * self.l * np.cos(pv) + f * self.k * np.sin(pv))
        ones = np.ones((self.grid.nz, 1, 1))
        return State3D(u=ones * u2, v=ones * v2,
                       b=np.zeros(self.grid.shape3d), eta=eta, t=t)

    def initial(self) -> State3D:
        return self.exact(0.0)


class VerticalDiffusion:
    """V3D-2: decay of a single vertical mode under vertical viscosity.

    `lam` is the DISCRETE eigenvalue of the tridiagonal, so the exact solution
    is exact for the discretisation too and the measured error sits at roundoff
    (1e-13). That makes this a **structural check** - a wrong tridiagonal, a
    wrong boundary row or a wrong theta_v weighting all break it - and NOT a
    convergence-order test: the "order" column computed from roundoff-level
    errors is noise. Refining dt would be the way to measure the time order,
    and `cmd_verify3d` cannot do that (it sweeps nx or nz and derives dt from
    dx), so no order is claimed here (docs/90 N34).
    """

    name = "vdiffusion"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics3D, mode_z: int,
                 u0: float) -> None:
        if physics.nu <= 0.0:
            raise ValueError("vdiffusion requires physics.nu > 0")
        # The exact solution is a pure exponential decay of a v-free profile.
        # With f != 0 the model performs ~1600 inertial rotations over the
        # e-folding time and the "error" is 100-190 %, which nobody noticed
        # because verify3d never judged anything (docs/90 N34).
        if abs(getattr(physics, "f", 0.0)) > 0.0:
            raise ValueError(
                "VerticalDiffusion needs f = 0 (its exact solution has no "
                "Coriolis); pass --set physics.f0=0")
        self.grid, self.physics, self.u0 = grid, physics, u0
        self.m = np.pi * mode_z / physics.H
        self.lam = (2.0 * np.cos(np.pi * mode_z / grid.nz) - 2.0) / grid.dz**2
        self.rate = physics.nu * self.lam                     # negative
        self.period = -1.0 / self.rate if self.rate < 0.0 else np.inf
        self._profile = np.cos(self.m * grid.z_centre())[:, None, None]

    def exact(self, t: float) -> State3D:
        shape = self.grid.shape3d
        u = self.u0 * self._profile * np.exp(self.rate * t) * np.ones(shape)
        return State3D(u=u, v=np.zeros(shape), b=np.zeros(shape),
                       eta=np.zeros(self.grid.shape), t=t)

    def initial(self) -> State3D:
        return self.exact(0.0)


class BaroclinicIGW:
    """V3D-3: exact internal gravity wave for vertical mode n.

    The horizontal structure is the 2D inertia-gravity wave with gH replaced
    by c_n^2 = (N/m)^2, multiplied by cos(m*zt); buoyancy and w carry the
    sin(m*zt) structure. Because m*H = n*pi, w vanishes at both boundaries and
    the depth-integrated transport is zero, so eta stays identically zero -
    but only when the baroclinic pressure gradient has its depth mean removed
    (spec S7.2).
    """

    name = "baroclinic_igw"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics3D, mode_x: int, mode_y: int,
                 mode_z: int, eta0: float) -> None:
        if physics.N2 <= 0.0:
            raise ValueError("baroclinic_igw requires physics.N2 > 0")
        self.grid, self.physics, self.eta0 = grid, physics, eta0
        self.k = 2.0 * np.pi * mode_x / grid.Lx
        self.l = 2.0 * np.pi * mode_y / grid.Ly
        self.kappa2 = self.k**2 + self.l**2
        self.m = np.pi * mode_z / physics.H
        self.c2 = physics.N2 / self.m**2                      # c_n^2
        self.omega = float(np.sqrt(physics.f**2 + self.c2 * self.kappa2))
        self.period = 2.0 * np.pi / self.omega

        zt = grid.z_centre()[:, None, None]
        self._cos_z = np.cos(self.m * zt)
        self._sin_z = np.sin(self.m * zt)

    def exact(self, t: float) -> State3D:
        g_, f = self.physics.g, self.physics.f
        amp = g_ * self.eta0 / (self.c2 * self.kappa2)
        x_u, y_u = self.grid.coords("u")
        x_v, y_v = self.grid.coords("v")
        x_e, y_e = self.grid.coords("eta")
        pu = self.k * x_u + self.l * y_u - self.omega * t
        pv = self.k * x_v + self.l * y_v - self.omega * t
        pe = self.k * x_e + self.l * y_e - self.omega * t

        u = amp * (self.omega * self.k * np.cos(pu) - f * self.l * np.sin(pu)) * self._cos_z
        v = amp * (self.omega * self.l * np.cos(pv) + f * self.k * np.sin(pv)) * self._cos_z
        b = -g_ * self.eta0 * self.m * np.cos(pe) * self._sin_z
        return State3D(u=u, v=v, b=b,
                       eta=np.zeros(self.grid.shape), t=t)

    def initial(self) -> State3D:
        return self.exact(0.0)

    def exact_w(self, t: float) -> np.ndarray:
        """Vertical velocity at interfaces, for the diagnostic check."""
        x_e, y_e = self.grid.coords("eta")
        pe = self.k * x_e + self.l * y_e - self.omega * t
        zt = self.grid.z_interface()[:, None, None]
        scale = self.physics.g * self.eta0 * self.omega / (self.c2 * self.m)
        return scale * np.sin(pe) * np.sin(self.m * zt)


class TracerAdvection:
    """V3D-4: a tracer sinusoid carried by a uniform flow (spec S9.5).

    Uniform flow does not advect itself and is divergence free, so momentum and
    the free surface stay exactly at rest and the case isolates the tracer
    advection operator. Setting alpha_T = beta_S = 0 makes the tracer passive:
    it never feeds buoyancy, so no pressure gradient appears to disturb the flow.
    """

    name = "tracer_advect"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics3D, mode_x: int, mode_y: int,
                 amp: float, u0: float, v0: float) -> None:
        if physics.tracers != "TS":
            raise ValueError("tracer_advect requires physics3d_v04.tracers = 'TS'")
        if physics.f != 0.0:
            raise ValueError("tracer_advect requires f = 0; Coriolis would turn "
                             "the uniform flow and break the exact solution")
        self.grid, self.physics = grid, physics
        self.amp, self.u0, self.v0 = amp, u0, v0
        self.k = 2.0 * np.pi * mode_x / grid.Lx
        self.l = 2.0 * np.pi * mode_y / grid.Ly
        speed = np.hypot(u0, v0)
        self.period = (grid.Lx / speed) if speed > 0.0 else np.inf

    def exact(self, t: float) -> State3D:
        shape = self.grid.shape3d
        x, y = self.grid.coords("eta")
        phase = self.k * (x - self.u0 * t) + self.l * (y - self.v0 * t)
        T = self.physics.T0 + self.amp * np.cos(phase)[None, :, :] * np.ones(shape)
        S = np.full(shape, self.physics.S0)
        return State3D(u=np.full(shape, self.u0), v=np.full(shape, self.v0),
                       b=self.physics.buoyancy(T, S), eta=np.zeros(self.grid.shape),
                       t=t, T=T, S=S)

    def initial(self) -> State3D:
        return self.exact(0.0)


class EkmanSpiral:
    """V3D-5: steady wind-driven Ekman spiral (spec S9.5).

    Exact steady solution of the finite-depth problem with a free-slip bottom:
        W = tau / (rho0 nu a sinh(a H)) * cosh(a (z + H)),  a = (1+i)/D,
        D = sqrt(2 nu / f),   W = u + i v
    It is not an exact solution of the DISCRETE vertical operator, so refining
    nz gives second-order convergence. Verifies the surface stress boundary
    condition, the implicit vertical viscosity and Coriolis together.
    """

    name = "ekman"
    has_exact = True

    def __init__(self, grid: CGrid, physics: Physics3D, n_inertial: float) -> None:
        if physics.nu <= 0.0 or physics.f == 0.0:
            raise ValueError("ekman requires nu > 0 and f != 0")
        if physics.tau_x == 0.0 and physics.tau_y == 0.0:
            raise ValueError("ekman requires a non-zero surface stress")
        self.grid, self.physics = grid, physics
        self.D = np.sqrt(2.0 * physics.nu / abs(physics.f))
        a = (1.0 + 1j) / self.D
        tau = complex(physics.tau_x, physics.tau_y)
        amp = tau / (physics.rho0 * physics.nu * a * np.sinh(a * physics.H))
        zt = grid.z_centre()[:, None, None]
        self._W = amp * np.cosh(a * zt)
        self.period = 2.0 * np.pi / abs(physics.f)      # inertial period
        self.n_inertial = n_inertial

    def exact(self, t: float) -> State3D:
        shape = self.grid.shape3d
        u = np.real(self._W) * np.ones(shape)
        v = np.imag(self._W) * np.ones(shape)
        return State3D(u=u, v=v, b=np.zeros(shape),
                       eta=np.zeros(self.grid.shape), t=t)

    def initial(self) -> State3D:
        return self.exact(0.0)

    def surface_angle(self) -> float:
        """Angle of the surface current relative to the wind, in degrees."""
        return float(np.degrees(np.angle(self._W[0, 0, 0])))


def build_case3d(name: str, grid: CGrid, physics: Physics3D, config):
    """Instantiate a 3D verification case from config/cases.toml."""
    section = config.section(f"case.{name}")
    if name == "barotropic3d":
        case = Barotropic3D(grid, physics, int(section["mode_x"]),
                            int(section["mode_y"]), float(section["eta0"]))
        t_final = float(section["n_periods"]) * case.period
    elif name == "vdiffusion":
        case = VerticalDiffusion(grid, physics, int(section["mode_z"]),
                                 float(section["u0"]))
        t_final = float(section["n_efolds"]) * case.period
    elif name == "tracer_advect":
        case = TracerAdvection(grid, physics, int(section["mode_x"]),
                               int(section["mode_y"]), float(section["amp"]),
                               float(section["u0"]), float(section["v0"]))
        t_final = float(section["n_transits"]) * case.period
    elif name == "ekman":
        case = EkmanSpiral(grid, physics, float(section["n_inertial"]))
        t_final = float(section["n_inertial"]) * case.period
    elif name == "baroclinic_igw":
        case = BaroclinicIGW(grid, physics, int(section["mode_x"]),
                             int(section["mode_y"]), int(section["mode_z"]),
                             float(section["eta0"]))
        t_final = float(section["n_periods"]) * case.period
    else:
        raise ValueError(f"unknown 3D case '{name}' (implemented: barotropic3d|"
                         f"vdiffusion|baroclinic_igw|tracer_advect|ekman)")
    return case, t_final
