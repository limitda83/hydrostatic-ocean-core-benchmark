#########################################################################
#  Module: cases3d_v05                                                  #
#  Description: Verification cases for spec v0.5 (docs/03_             #
#               discretization_spec.md S10.8). Unlike V1-V3D-5 these    #
#               mostly have no closed-form solution; each defines a     #
#               diagnostic that is provably zero for the exact          #
#               equations, so the measured value IS the error.          #
#  Pipeline: domain/model3d_v05 -> cases3d_v05 -> verify3d5             #
#########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from libs.core.domain import Domain
from libs.core.model3d import State3D
from libs.core.model3d_v05 import PhysicsV05


@dataclass
class CaseResult:
    """What a v0.5 case reports. `metric` is the number under test."""

    name: str
    metric: float
    metric_name: str
    extra: dict


class SeamountRest:
    """V5-3: Beckmann & Haidvogel (1993) resting seamount test.

    A horizontally uniform stratification over a seamount is an exact steady
    solution: rho depends on z only, so the horizontal pressure gradient
    vanishes at every depth. Any velocity the model develops is the
    discretisation's pressure-gradient error, which is the classic failure
    mode of terrain-following coordinates.

    Reported as max|u| in cm/s, the unit the literature uses.
    """

    name = "seamount_rest"
    metric_name = "max_abs_u_cm_s"

    def __init__(self, domain: Domain, physics: PhysicsV05, delta_rho: float,
                 n_days: float) -> None:
        self.domain, self.physics = domain, physics
        self.delta_rho = delta_rho
        self.t_final = n_days * 86400.0

    def initial(self) -> State3D:
        d, p = self.domain, self.physics
        z = d.depth_at_centres()                 # [nz,ny,nx], positive down
        # An exponential profile in DEPTH, identical in every column, so the
        # exact horizontal pressure gradient is zero by construction.
        rho_prime = self.delta_rho * np.exp(-z / 500.0)
        b = -p.g * rho_prime / p.rho0 * d.mask3
        zero3 = np.zeros(d.grid.shape3d)
        return State3D(u=zero3, v=zero3, b=b, eta=np.zeros(d.grid.shape), t=0.0)

    def evaluate(self, s: State3D) -> CaseResult:
        speed = np.sqrt(s.u**2 + s.v**2)
        return CaseResult(self.name, float(np.max(speed) * 100.0),
                          self.metric_name,
                          {"rms_speed_cm_s": float(np.sqrt(np.mean(speed**2)) * 100.0),
                           "rx0": self.domain.roughness.rx0,
                           "vcoord": self.domain.vcoord})


class ZStarConstancy:
    """V5-6: a uniform tracer must stay uniform while the free surface moves.

    On a moving vertical coordinate the layer thickness changes every step;
    if the thickness update and the tracer advection are not written in the
    same flux form, a uniform field develops structure. The metric is the
    peak-to-peak spread of an initially uniform tracer.
    """

    name = "zstar_constancy"
    metric_name = "tracer_spread"

    def __init__(self, domain: Domain, physics: PhysicsV05, eta0: float,
                 n_periods: float) -> None:
        self.domain, self.physics = domain, physics
        self.eta0 = eta0
        c = np.sqrt(physics.g * float(np.max(domain.H)))
        self.t_final = n_periods * domain.grid.Lx / c

    def initial(self) -> State3D:
        d = self.domain
        x, y = d.grid.coords("eta")
        eta = self.eta0 * np.cos(2.0 * np.pi * x / d.grid.Lx) * d.mask
        zero3 = np.zeros(d.grid.shape3d)
        T = np.full(d.grid.shape3d, self.physics.T0) * d.mask3
        S = np.full(d.grid.shape3d, self.physics.S0) * d.mask3
        b = np.zeros(d.grid.shape3d)
        wet = d.mask3 > 0
        area = d.grid.dx * d.grid.dy
        self._inv0 = float(np.sum(T[wet] * (d.dz3_ref * area)[wet]))
        self._inv0_live = float(np.sum(T[wet] * (self._live_thickness(eta) * area)[wet]))
        return State3D(u=zero3, v=zero3, b=b, eta=eta, t=0.0, T=T, S=S)

    def _live_thickness(self, eta: np.ndarray) -> np.ndarray:
        """Layer thickness including the volume the free surface holds. For
        `zlevel` the model keeps dz fixed (S10.3), so the surface volume lives
        entirely in the top cell; for a moving coordinate `layer_thickness`
        already carries it."""
        d = self.domain
        if d.vcoord != "zlevel":
            return d.layer_thickness(eta)
        dz = d.dz3_ref.copy()
        dz[0] = dz[0] + eta * (d.mask3[0] > 0)
        return dz

    def evaluate(self, s: State3D) -> CaseResult:
        """Report constancy AND conservation, the latter on BOTH weightings.

        An earlier version of this docstring claimed the two "cannot both be
        exact" with a linear free surface. That was wrong, and the error was in
        the diagnostic, not the model (docs/90 N31): with a linear free surface
        the layer thickness is fixed but the top cell's VOLUME still changes by
        eta, so the physically meaningful inventory is sum(c * (dz + eta at the
        surface)). Measured that way the tracer is conserved to ~1e-8 while
        constancy holds to ~1e-13; measured on the resting thickness alone a
        1e-5 "drift" appears that is just the volume the surface let in. Both
        numbers are reported so the difference stays visible.
        """
        d = self.domain
        wet = d.mask3 > 0
        T = s.T[wet]
        area = d.grid.dx * d.grid.dy
        vol_ref = (d.dz3_ref * area)[wet]
        vol_live = (self._live_thickness(s.eta) * area)[wet]
        inv0, inv0_live = getattr(self, "_inv0", None), getattr(self, "_inv0_live", None)
        inv = float(np.sum(T * vol_ref))
        inv_live = float(np.sum(T * vol_live))
        drift = 0.0 if not inv0 else (inv - inv0) / abs(inv0)
        drift_live = 0.0 if not inv0_live else (inv_live - inv0_live) / abs(inv0_live)
        return CaseResult(self.name, float(np.max(T) - np.min(T)),
                          self.metric_name,
                          {"T_mean": float(np.mean(T)),
                           "tracer_integral_drift": drift,
                           "tracer_integral_drift_live": drift_live,
                           "vcoord": d.vcoord})


class FlatherRadiation:
    """V5-5: a surface bump must leave through an open boundary.

    The domain is a channel, open in x and closed in y. A Gaussian elevation
    splits into two waves; after they have had time to reach and cross the
    boundary, whatever surface energy remains is reflection.
    """

    name = "flather_radiate"
    metric_name = "reflected_energy_fraction"

    def __init__(self, domain: Domain, physics: PhysicsV05, eta0: float,
                 width: float, n_transits: float) -> None:
        self.domain, self.physics = domain, physics
        self.eta0, self.width = eta0, width
        c = np.sqrt(physics.g * float(np.mean(domain.H)))
        self.t_final = n_transits * domain.grid.Lx / c

    def initial(self) -> State3D:
        d = self.domain
        x, _ = d.grid.coords("eta")
        eta = self.eta0 * np.exp(-((x - 0.5 * d.grid.Lx) / self.width) ** 2)
        self._e0 = float(np.sum(eta**2))
        zero3 = np.zeros(d.grid.shape3d)
        return State3D(u=zero3, v=zero3, b=zero3,
                       eta=eta * d.mask, t=0.0)

    def evaluate(self, s: State3D) -> CaseResult:
        e = float(np.sum(s.eta**2))
        return CaseResult(self.name, e / self._e0, self.metric_name,
                          {"eta_max_final": float(np.max(np.abs(s.eta))),
                           "eta_max_initial": self.eta0})


class LockExchange:
    """A gravity current released over topography. The diagnostic is the
    reference potential energy (RPE): the exact equations can only raise it
    through real mixing, so its rise IS the model's spurious mixing - the
    accuracy measure the ocean community actually cares about."""

    name = "lock_exchange"
    metric_name = "rpe_drift_rel"

    def __init__(self, domain: Domain, physics: PhysicsV05, delta_T: float,
                 n_hours: float) -> None:
        self.domain, self.physics = domain, physics
        self.delta_T = delta_T
        self.t_final = n_hours * 3600.0
        self._hyps: tuple[np.ndarray, np.ndarray] | None = None

    def _hypsometry(self) -> tuple[np.ndarray, np.ndarray]:
        """The basin's volume-below-height curve, cached.

        The reference state puts the densest water in the DEEPEST part of the
        basin, so the level a given accumulated volume reaches depends on how
        the basin's area varies with height. Dividing by the surface area
        instead - the old code - assumes a box, and over `rough` bathymetry
        that overstated the spurious mixing by ~11 % (docs/90 N37).
        """
        if getattr(self, "_hyps", None) is None:
            d = self.domain
            area = d.grid.dx * d.grid.dy
            dz = d.dz3_ref
            wet = d.mask3 > 0
            # height of each cell's bottom and top above the deepest point
            zi = np.concatenate([np.zeros((1,) + dz.shape[1:]), np.cumsum(dz, axis=0)], axis=0)
            deepest = float(np.max(zi[-1]))
            top = deepest - zi[:-1]                      # height of the cell top
            bot = deepest - zi[1:]                       # height of the cell bottom
            zb, zt = bot[wet], top[wet]
            grid_z = np.unique(np.concatenate([zb, zt, [0.0, deepest]]))
            # volume below each height z: sum over cells of clip(z - zb, 0, zt - zb)
            #   = sum_{zb < z} (z - zb) - sum_{zt < z} (z - zt), evaluated for all z at once
            # with sorted prefix sums (O(N log N); the per-z loop it replaces was O(N_z * N)
            # and took >20 min at 400^2 x 30 on a loaded host, docs/90 N44).
            def below(edges: np.ndarray) -> np.ndarray:
                e = np.sort(edges); cs = np.concatenate([[0.0], np.cumsum(e)])
                k = np.searchsorted(e, grid_z, side="left")
                return k * grid_z - cs[k]
            vol_below = area * (below(zb) - below(zt))
            self._hyps = (vol_below, grid_z)
        return self._hyps

    def _rpe(self, b: np.ndarray) -> float:
        """Sort the buoyancy into a stable column and integrate b*z, with the
        reference level taken from the basin's own hypsometric curve."""
        d = self.domain
        wet = d.mask3 > 0
        vol = (d.grid.dx * d.grid.dy * self.domain.dz3_ref)[wet]
        bb = b[wet]
        order = np.argsort(bb)               # densest (lowest b) first
        v = vol[order]
        vol_below, grid_z = self._hypsometry()
        z = np.interp(np.cumsum(v), vol_below, grid_z)
        return float(np.sum(bb[order] * z * v))

    def initial(self) -> State3D:
        d, p = self.domain, self.physics
        x, _ = d.grid.coords("eta")
        warm = (x > 0.5 * d.grid.Lx).astype(np.float64)
        T = (p.T0 + self.delta_T * warm)[None, :, :] * np.ones(d.grid.shape3d)
        S = np.full(d.grid.shape3d, p.S0)
        z = d.depth_at_centres()
        from libs.core.eos import buoyancy as _b
        b = _b(p.eos, T, S, z, g=p.g, rho0=p.rho0, alpha_T=p.alpha_T,
               beta_S=p.beta_S, T0=p.T0, S0=p.S0) * d.mask3
        self._rpe0 = self._rpe(b)
        zero3 = np.zeros(d.grid.shape3d)
        return State3D(u=zero3, v=zero3, b=b, eta=np.zeros(d.grid.shape),
                       t=0.0, T=T * d.mask3, S=S * d.mask3)

    def evaluate(self, s: State3D) -> CaseResult:
        rpe = self._rpe(s.b)
        drift = (rpe - self._rpe0) / abs(self._rpe0) if self._rpe0 else 0.0
        return CaseResult(self.name, float(drift), self.metric_name,
                          {"rpe_initial": self._rpe0, "rpe_final": rpe})


class BasinSeiche:
    """Tier-2 case C2: a free barotropic seiche in a closed basin over topography.

    The barotropic mode is fully ACTIVE here, which is what separates the
    time-integration schemes on accuracy: with it at rest (docs/22) they tie
    to twelve digits. There is no closed-form solution over topography, so
    the sweep judges accuracy against a converged reference (small dt, same
    grid); this class reports the two conserved-quantity diagnostics that
    are exact for the inviscid equations - total energy and the mean of eta.
    """

    name = "basin_seiche"
    metric_name = "energy_drift_rel"

    def __init__(self, domain: Domain, physics: PhysicsV05, eta0: float,
                 n_periods: float) -> None:
        self.domain, self.physics = domain, physics
        self.eta0 = eta0
        h_mean = float(np.mean(domain.H[domain.mask > 0]))
        self.period = 2.0 * domain.grid.Lx / np.sqrt(physics.g * h_mean)
        self.t_final = n_periods * self.period

    def _energy(self, s: State3D) -> float:
        """Total energy of the linear free-surface system.

        `u` and `v` live on FACES, so the kinetic energy must be weighted by the
        FACE thickness, not the cell-centre thickness: over a seamount the two
        differ in 5.7 % of cells and the drift they report differs by 7-27 %
        (docs/90 N32). On a flat bottom they coincide, which is why this went
        unnoticed until the case was run over topography.
        """
        d, g = self.domain, self.physics.g
        dzu, dzv = d.face_thickness(d.layer_thickness(s.eta))
        ke = 0.5 * float(np.sum(dzu * s.u ** 2 + dzv * s.v ** 2))
        pe = 0.5 * g * float(np.sum(d.mask * s.eta ** 2))
        return (ke + pe) * d.grid.dx * d.grid.dy

    def initial(self) -> State3D:
        d = self.domain
        x, _ = d.grid.coords("eta")
        eta = self.eta0 * np.cos(np.pi * x / d.grid.Lx) * d.mask
        eta -= np.sum(eta) / max(1.0, float(np.sum(d.mask)))
        eta *= d.mask
        zero3 = np.zeros(d.grid.shape3d)
        s0 = State3D(u=zero3, v=zero3, b=zero3, eta=eta, t=0.0)
        self._e0 = self._energy(s0)
        self._m0 = float(np.sum(eta))
        return s0

    def evaluate(self, s: State3D) -> CaseResult:
        e = self._energy(s)
        return CaseResult(self.name, (e - self._e0) / abs(self._e0), self.metric_name,
                          {"eta_max": float(np.max(np.abs(s.eta))),
                           "eta_mean_drift": float(np.sum(s.eta)) - self._m0,
                           "period_s": self.period})


class BaroclinicIGWTopo:
    """Tier-2 case C1: the V3D-3 internal-wave mode launched over topography.

    The exact flat-bottom mode (cases3d.BaroclinicIGW) is used as the INITIAL
    state on the nominal z-levels and masked to the wet cells; over a seamount
    it is no longer an eigenmode, so there is no closed-form solution and the
    sweep judges accuracy against a converged reference on the same grid.
    This is the "IGW over topography" case of docs/04 S2, kept periodic so
    the compiled backends run it unchanged.
    """

    name = "baroclinic_igw"
    metric_name = "eta_rms"

    def __init__(self, domain: Domain, physics: PhysicsV05, mode_x: int, mode_y: int,
                 mode_z: int, eta0: float, n_periods: float) -> None:
        if physics.N2 <= 0.0:
            raise ValueError("baroclinic_igw requires physics3d.N2 > 0")
        self.domain, self.physics, self.eta0 = domain, physics, eta0
        g = domain.grid
        self.k = 2.0 * np.pi * mode_x / g.Lx
        self.l = 2.0 * np.pi * mode_y / g.Ly
        self.kappa2 = self.k ** 2 + self.l ** 2
        self.m = np.pi * mode_z / physics.H
        self.c2 = physics.N2 / self.m ** 2
        self.omega = float(np.sqrt(physics.f ** 2 + self.c2 * self.kappa2))
        self.period = 2.0 * np.pi / self.omega
        self.t_final = n_periods * self.period

    def initial(self) -> State3D:
        d, g_, f = self.domain, self.physics.g, self.physics.f
        amp = g_ * self.eta0 / (self.c2 * self.kappa2)
        zt = d.grid.z_centre()[:, None, None]
        cos_z, sin_z = np.cos(self.m * zt), np.sin(self.m * zt)
        x_u, y_u = d.grid.coords("u")
        x_v, y_v = d.grid.coords("v")
        x_e, y_e = d.grid.coords("eta")
        pu = self.k * x_u + self.l * y_u
        pv = self.k * x_v + self.l * y_v
        pe = self.k * x_e + self.l * y_e
        u = amp * (self.omega * self.k * np.cos(pu) - f * self.l * np.sin(pu)) * cos_z * d.mask3u
        v = amp * (self.omega * self.l * np.cos(pv) + f * self.k * np.sin(pv)) * cos_z * d.mask3v
        b = -g_ * self.eta0 * self.m * np.cos(pe) * sin_z * d.mask3
        return State3D(u=u, v=v, b=b, eta=np.zeros(d.grid.shape), t=0.0)

    def evaluate(self, s: State3D) -> CaseResult:
        d = self.domain
        wet = max(1.0, float(np.sum(d.mask)))
        rms = float(np.sqrt(np.sum((s.eta * d.mask) ** 2) / wet))
        return CaseResult(self.name, rms, self.metric_name,
                          {"eta_max": float(np.max(np.abs(s.eta))), "period_s": self.period,
                           "u_max": float(np.max(np.abs(s.u)))})


def build_case3d_v05(name: str, domain: Domain, physics: PhysicsV05, config):
    """Instantiate a v0.5 case from config/cases.toml (R3)."""
    s = config.section(f"case.{name}")
    if name == "seamount_rest":
        return SeamountRest(domain, physics, float(s["delta_rho"]),
                            float(s["n_days"]))
    if name == "zstar_constancy":
        return ZStarConstancy(domain, physics, float(s["eta0"]),
                              float(s["n_periods"]))
    if name == "flather_radiate":
        return FlatherRadiation(domain, physics, float(s["eta0"]),
                                float(s["width"]), float(s["n_transits"]))
    if name == "lock_exchange":
        return LockExchange(domain, physics, float(s["delta_T"]),
                            float(s["n_hours"]))
    if name == "basin_seiche":
        return BasinSeiche(domain, physics, float(s["eta0"]), float(s["n_periods"]))
    if name == "baroclinic_igw":
        return BaroclinicIGWTopo(domain, physics, int(s["mode_x"]), int(s["mode_y"]),
                                 int(s["mode_z"]), float(s["eta0"]), float(s["n_periods"]))
    raise ValueError(f"unknown v0.5 case '{name}' (implemented: seamount_rest|"
                     "zstar_constancy|flather_radiate|lock_exchange|basin_seiche|"
                     "baroclinic_igw)")
