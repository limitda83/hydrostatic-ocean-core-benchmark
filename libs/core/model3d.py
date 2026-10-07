#########################################################################
#  Module: model3d                                                      #
#  Description: 3D hydrostatic linear Boussinesq solver following the   #
#               Casulli decomposition of docs/03_discretization_spec.md #
#               S7.5: two batched tridiagonal solves per Picard pass,   #
#               then one 2D free-surface Helmholtz, then back-          #
#               substitution. With nu = 0 and b = 0 it reduces exactly  #
#               to the 2D scheme of S3.3.                               #
#  Pipeline: grid/operators/vertical/solvers -> model3d -> driver       #
#########################################################################

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from libs.core.advection import (laplacian_h, momentum_advection,
                                 tracer_advection)
from libs.core.grid import CGrid
from libs.core.operators import (avg_u_to_v, avg_v_to_u, div_eta, gradx_u,
                                 grady_v)
from libs.core.schemes import SchemeParams, explicit_dt_max
from libs.core.solvers import build_solver
from libs.core.vertical import (apply_diffusion, buoyancy_potential,
                                diffusion_coeffs, remove_depth_mean,
                                thomas_batched, w_at_centres,
                                w_from_divergence)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class State3D:
    """Prognostic state: horizontal velocity and buoyancy in 3D, free surface in 2D."""

    u: np.ndarray      # [nz, ny, nx] at east faces, layer centres
    v: np.ndarray      # [nz, ny, nx] at north faces, layer centres
    b: np.ndarray      # [nz, ny, nx] buoyancy at cell centres
    eta: np.ndarray    # [ny, nx] free surface
    t: float = 0.0
    # Prognostic tracers when physics.tracers == "TS"; b is then diagnostic.
    T: np.ndarray | None = None
    S: np.ndarray | None = None
    # Turbulent kinetic energy on interfaces [nz+1, ny, nx] (spec v0.6 closure).
    e: np.ndarray | None = None

    def astype(self, dtype) -> "State3D":
        out = replace(self, u=self.u.astype(dtype), v=self.v.astype(dtype),
                      b=self.b.astype(dtype), eta=self.eta.astype(dtype))
        if self.T is not None:
            out = replace(out, T=self.T.astype(dtype), S=self.S.astype(dtype))
        return out


@dataclass(frozen=True)
class Physics3D:
    """Physical constants for the 3D model."""

    g: float
    f: float
    H: float
    N2: float = 0.0          # buoyancy frequency squared [s-2]
    nu: float = 0.0          # vertical viscosity [m2 s-1]
    kappa: float = 0.0       # vertical diffusivity for buoyancy [m2 s-1]
    rho0: float = 1025.0
    tau_x: float = 0.0       # surface stress [N m-2]
    tau_y: float = 0.0
    bottom_drag: float = 0.0  # linear drag coefficient [m s-1]
    # --- spec v0.4 full core ---
    tracers: str = "buoyancy"   # "buoyancy" | "TS"
    alpha_T: float = 2.0e-4     # thermal expansion [K-1]
    beta_S: float = 7.4e-4      # haline contraction [psu-1]
    T0: float = 10.0
    S0: float = 35.0
    cp: float = 3990.0          # heat capacity [J kg-1 K-1]
    A_h: float = 0.0            # horizontal viscosity [m2 s-1]
    K_h: float = 0.0            # horizontal tracer diffusivity [m2 s-1]
    q_heat: float = 0.0         # surface heat flux [W m-2], positive downward
    q_salt: float = 0.0         # surface virtual salt flux [psu m s-1]

    def buoyancy(self, T: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Linear equation of state (spec S9.1)."""
        return self.g * (self.alpha_T * (T - self.T0)
                         - self.beta_S * (S - self.S0))

    @property
    def c(self) -> float:
        return float(np.sqrt(self.g * self.H))


class Stepper3D:
    """One-step 3D integrator for a fixed (grid, physics, dt, params)."""

    def __init__(self, grid: CGrid, physics: Physics3D, params: SchemeParams,
                 dt: float) -> None:
        params.validate()
        if grid.nz < 1 or grid.depth <= 0.0:
            raise ValueError("the 3D model needs grid.nz >= 1 and physics.H > 0")
        self.grid = grid
        self.physics = physics
        self.params = params
        self.dt = float(dt)
        self.solver_iterations = 0
        self.solver_failures = 0
        self.tridiagonal_solves = 0
        self.barotropic_substeps = 0

        nz, ny, nx = grid.shape3d
        # Viscosity lives at interfaces on the tracer grid and is used for u, v
        # and b alike; averaging nu to the C-grid momentum points is a Phase 2b
        # refinement and is a no-op for the uniform nu used here.
        self.nu = np.full((nz + 1, ny, nx), physics.nu, dtype=np.float64)
        self.kappa = np.full((nz + 1, ny, nx), physics.kappa, dtype=np.float64)

        thv = params.theta_v
        self.tri_m = diffusion_coeffs(self.nu, grid.dz, self.dt, thv,
                                      physics.bottom_drag)
        self.tri_b = diffusion_coeffs(self.kappa, grid.dz, self.dt, thv, 0.0)

        # q solves A q = 1 and is the "effective depth profile" of spec S7.5.
        # nu is uniform here, so Hh is a scalar and the 2D Helmholtz solver
        # applies unchanged; a variable-coefficient Helmholtz is Phase 2b.
        ones = np.ones(grid.shape3d, dtype=np.float64)
        self.q = thomas_batched(*self.tri_m, ones)
        self.tridiagonal_solves += 1
        h_eff = grid.dz * np.sum(self.q, axis=0)
        spread = float(np.max(h_eff) - np.min(h_eff))
        if spread > 1e-10 * max(1.0, float(np.mean(h_eff))):
            raise NotImplementedError(
                "spec v0.2 assumes a horizontally uniform effective depth "
                f"(variable-coefficient Helmholtz is Phase 2b); spread={spread:.3e}")
        self.h_eff = float(np.mean(h_eff))

        coef = physics.g * self.h_eff * params.theta**2 * self.dt**2
        self.solver = (build_solver(params.solver, grid, coef, params.rtol,
                                    params.max_iter)
                       if params.name == "theta" else None)

        # Barotropic substep count for split_explicit (spec S8.4). The
        # substep must satisfy the surface-gravity CFL; the baroclinic step
        # is limited only by the much slower internal wave speed.
        self.n_split = 0
        if params.name == "split_explicit":
            dt_baro = explicit_dt_max(grid, physics)
            self.n_split = (params.n_split if params.n_split > 0
                            else max(1, int(np.ceil(1.2 * self.dt / dt_baro))))
            if self.dt / self.n_split > dt_baro:
                raise ValueError(
                    f"n_split={self.n_split} leaves a barotropic substep of "
                    f"{self.dt / self.n_split:.1f}s above the stability limit "
                    f"{dt_baro:.1f}s")

    # ------------------------------------------------------------------ parts
    def _pressure_gradients(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Baroclinic pressure gradient from the depth-mean-free potential."""
        phi = remove_depth_mean(buoyancy_potential(b, self.grid.dz))
        return gradx_u(phi, self.grid), grady_v(phi, self.grid)

    def _solve_column(self, coeffs, rhs: np.ndarray) -> np.ndarray:
        self.tridiagonal_solves += 1
        return thomas_batched(*coeffs, rhs)

    def _momentum_extras(self, u: np.ndarray, v: np.ndarray, w: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Advection and horizontal viscosity, both explicit (spec S9.2-S9.3)."""
        ex_u = np.zeros_like(u)
        ex_v = np.zeros_like(v)
        if self.params.advection != "none":
            adv_u, adv_v = momentum_advection(u, v, w, self.grid)
            ex_u -= adv_u
            ex_v -= adv_v
        if self.physics.A_h > 0.0:
            ex_u += self.physics.A_h * laplacian_h(u, self.grid)
            ex_v += self.physics.A_h * laplacian_h(v, self.grid)
        return ex_u, ex_v

    def _advance_tracer(self, c: np.ndarray, u: np.ndarray, v: np.ndarray,
                        w: np.ndarray, surf_flux: np.ndarray,
                        source: np.ndarray | None = None) -> np.ndarray:
        """One tracer step: explicit advection and horizontal diffusion, then
        the implicit vertical solve that also carries the surface flux."""
        dt, thv, dz = self.dt, self.params.theta_v, self.grid.dz
        tend = np.zeros_like(c) if source is None else source
        if self.params.advection != "none":
            tend = tend - tracer_advection(c, u, v, w, self.grid,
                                           self.params.advection)
        if self.physics.K_h > 0.0:
            tend = tend + self.physics.K_h * laplacian_h(c, self.grid)
        rhs = (c + dt * tend
               + (1.0 - thv) * dt * apply_diffusion(c, self.kappa, dz, surf_flux))
        rhs[0] = rhs[0] + thv * dt * surf_flux / dz
        return self._solve_column(self.tri_b, rhs)

    def _barotropic(self, u_int: np.ndarray, v_int: np.ndarray, eta: np.ndarray,
                    fx: np.ndarray, fy: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Substep the depth-integrated transports and free surface (spec S8.3-4).

        Forward-backward, entirely local stencils - no elliptic solve and no
        global reduction. That absence is the point of the scheme: docs/21 S3
        measured the semi-implicit alternative paying 30-173 us per PCG
        iteration on GPU for exactly this work.
        """
        g_, f, h_ = self.physics.g, self.physics.f, self.physics.H
        grid, ddt = self.grid, self.dt / self.n_split
        u_i, v_i, e = u_int, v_int, eta
        acc_u, acc_v = np.zeros_like(u_int), np.zeros_like(v_int)

        tc, npic = self.params.theta_cor, self.params.n_picard
        for _ in range(self.n_split):
            # Same theta_cor-weighted Picard Coriolis as the 2D forward-backward
            # scheme (spec S3.1). A sequential Coriolis here is only first order
            # and drags the whole scheme down with it.
            av_v, av_u = avg_v_to_u(v_i), avg_u_to_v(u_i)
            gxe = gradx_u(e, grid)
            gye = grady_v(e, grid)
            ui, vi = u_i, v_i
            for _ in range(npic):
                v_cor = (1.0 - tc) * av_v + tc * avg_v_to_u(vi)
                u_cor = (1.0 - tc) * av_u + tc * avg_u_to_v(ui)
                ui = u_i + ddt * (f * v_cor - g_ * h_ * gxe + fx)
                vi = v_i + ddt * (-f * u_cor - g_ * h_ * gye + fy)
            u_i, v_i = ui, vi
            e = e - ddt * div_eta(u_i, v_i, grid)
            acc_u += u_i
            acc_v += v_i
        self.barotropic_substeps += self.n_split

        if self.params.barotropic_average == "mean":
            # Production models filter the substeps (Shchepetkin & McWilliams
            # 2005). A plain mean is not the value at t+dt, so it breaks
            # convergence to the exact solution - kept as a measurable axis.
            return acc_u / self.n_split, acc_v / self.n_split, e
        return u_i, v_i, e

    def _step_split(self, s: State3D) -> State3D:
        """Split-explicit barotropic mode splitting (spec S8.3)."""
        g_, f, dt = self.physics.g, self.physics.f, self.dt
        tc, thv = self.params.theta_cor, self.params.theta_v
        grid, dz, h_ = self.grid, self.grid.dz, self.physics.H
        rho0 = self.physics.rho0

        pgx, pgy = self._pressure_gradients(s.b)
        av_prev_v = avg_v_to_u(s.v)
        av_prev_u = avg_u_to_v(s.u)
        tau_u = np.full(grid.shape, self.physics.tau_x / rho0)
        tau_v = np.full(grid.shape, self.physics.tau_y / rho0)
        dz_u = apply_diffusion(s.u, self.nu, dz, tau_u, self.physics.bottom_drag)
        dz_v = apply_diffusion(s.v, self.nu, dz, tau_v, self.physics.bottom_drag)
        w_old = w_from_divergence(div_eta(s.u, s.v, grid), dz)
        ex_u, ex_v = self._momentum_extras(s.u, s.v, w_old)
        stress_u = np.zeros(grid.shape3d)
        stress_v = np.zeros(grid.shape3d)
        stress_u[0] = thv * dt * tau_u / dz
        stress_u = stress_u + dt * ex_u
        stress_v[0] = thv * dt * tau_v / dz
        stress_v = stress_v + dt * ex_v

        u_int0 = dz * np.sum(s.u, axis=0)
        v_int0 = dz * np.sum(s.v, axis=0)

        u_it, v_it = s.u, s.v
        gh_u = gh_v = None
        cor_x = cor_y = None
        for _ in range(self.params.n_picard):
            v_cor = (1.0 - tc) * av_prev_v + tc * avg_v_to_u(v_it)
            u_cor = (1.0 - tc) * av_prev_u + tc * avg_u_to_v(u_it)
            # Baroclinic predictor WITHOUT the barotropic pressure gradient.
            gu = s.u + dt * (f * v_cor + pgx) + (1.0 - thv) * dt * dz_u
            gv = s.v + dt * (-f * u_cor + pgy) + (1.0 - thv) * dt * dz_v
            gh_u = self._solve_column(self.tri_m, gu + stress_u)
            gh_v = self._solve_column(self.tri_m, gv + stress_v)
            cor_x = dz * np.sum(f * v_cor, axis=0)
            cor_y = -dz * np.sum(f * u_cor, axis=0)
            u_it, v_it = gh_u, gh_v

        # Depth-integrated forcing with the Coriolis term removed: the
        # barotropic system carries its own, so leaving it in double-counts.
        fx = (dz * np.sum(gh_u, axis=0) - u_int0) / dt - cor_x
        fy = (dz * np.sum(gh_v, axis=0) - v_int0) / dt - cor_y

        u_int, v_int, eta_new = self._barotropic(u_int0, v_int0, s.eta, fx, fy)

        # Replace the depth mean of the baroclinic solution with the
        # barotropic result, so that int u dz = U exactly.
        u_new = gh_u - np.mean(gh_u, axis=0, keepdims=True) + u_int / h_
        v_new = gh_v - np.mean(gh_v, axis=0, keepdims=True) + v_int / h_

        w = w_from_divergence(div_eta(u_new, v_new, grid), dz)
        return self._advance_tracers(s, u_new, v_new, w, eta_new)

    def _advance_tracers(self, s: State3D, u: np.ndarray, v: np.ndarray,
                         w: np.ndarray, eta_new: np.ndarray) -> State3D:
        """Advance buoyancy, or temperature and salinity with the equation of
        state (spec S9.1). The -w N^2 term is the advection of the prescribed
        background stratification and belongs only to the buoyancy path; in the
        TS path the tracers carry the full stratification themselves."""
        dt, dz = self.dt, self.grid.dz
        rho0, cp = self.physics.rho0, self.physics.cp
        zero = np.zeros(self.grid.shape)

        if self.physics.tracers == "TS":
            flux_T = np.full(self.grid.shape, self.physics.q_heat / (rho0 * cp))
            flux_S = np.full(self.grid.shape, self.physics.q_salt)
            T_new = self._advance_tracer(s.T, u, v, w, flux_T)
            S_new = self._advance_tracer(s.S, u, v, w, flux_S)
            b_new = self.physics.buoyancy(T_new, S_new)
            return State3D(u=u, v=v, b=b_new, eta=eta_new, t=s.t + dt,
                           T=T_new, S=S_new)

        source = -self.physics.N2 * w_at_centres(w)
        b_new = self._advance_tracer(s.b, u, v, w, zero, source=source)
        return State3D(u=u, v=v, b=b_new, eta=eta_new, t=s.t + dt)

    # ----------------------------------------------------------------- public
    def step(self, s: State3D) -> State3D:
        if self.params.name == "split_explicit":
            return self._step_split(s)
        g_, f, dt = self.physics.g, self.physics.f, self.dt
        th, tc, thv = self.params.theta, self.params.theta_cor, self.params.theta_v
        if self.params.name != "theta":   # fb: full explicit surface gradient (docs/90 N14)
            th = 0.0
        grid, dz = self.grid, self.grid.dz
        rho0 = self.physics.rho0

        pgx, pgy = self._pressure_gradients(s.b)
        gx_eta = gradx_u(s.eta, grid)
        gy_eta = grady_v(s.eta, grid)
        av_prev_v = avg_v_to_u(s.v)
        av_prev_u = avg_u_to_v(s.u)
        div_old = div_eta(dz * np.sum(s.u, axis=0), dz * np.sum(s.v, axis=0), grid)

        # Explicit part of the vertical diffusion, plus the surface stress.
        tau_u = np.full(grid.shape, self.physics.tau_x / rho0)
        tau_v = np.full(grid.shape, self.physics.tau_y / rho0)
        dz_u = apply_diffusion(s.u, self.nu, dz, tau_u, self.physics.bottom_drag)
        dz_v = apply_diffusion(s.v, self.nu, dz, tau_v, self.physics.bottom_drag)
        w_old = w_from_divergence(div_eta(s.u, s.v, grid), dz)
        ex_u, ex_v = self._momentum_extras(s.u, s.v, w_old)
        # theta_v-implicit surface stress contribution to the tridiagonal rhs.
        stress_u = np.zeros(grid.shape3d)
        stress_v = np.zeros(grid.shape3d)
        stress_u[0] = thv * dt * tau_u / dz
        stress_u = stress_u + dt * ex_u
        stress_v[0] = thv * dt * tau_v / dz
        stress_v = stress_v + dt * ex_v

        u_it, v_it, eta_new = s.u, s.v, s.eta
        for _ in range(self.params.n_picard):
            v_cor = (1.0 - tc) * av_prev_v + tc * avg_v_to_u(v_it)
            u_cor = (1.0 - tc) * av_prev_u + tc * avg_u_to_v(u_it)

            gu = (s.u + dt * (f * v_cor + pgx) - g_ * dt * (1.0 - th) * gx_eta
                  + (1.0 - thv) * dt * dz_u)
            gv = (s.v + dt * (-f * u_cor + pgy) - g_ * dt * (1.0 - th) * gy_eta
                  + (1.0 - thv) * dt * dz_v)

            gh_u = self._solve_column(self.tri_m, gu + stress_u)
            gh_v = self._solve_column(self.tri_m, gv + stress_v)

            if self.params.name == "theta":
                div_g = div_eta(dz * np.sum(gh_u, axis=0),
                                dz * np.sum(gh_v, axis=0), grid)
                rhs = s.eta - dt * ((1.0 - th) * div_old + th * div_g)
                eta_new, report = self.solver.solve(rhs)
                self.solver_iterations += report.iterations
                if not report.converged:
                    self.solver_failures += 1
                u_it = gh_u - g_ * dt * th * self.q * gradx_u(eta_new, grid)
                v_it = gh_v - g_ * dt * th * self.q * grady_v(eta_new, grid)
            else:  # forward-backward: no implicit surface gradient
                u_it, v_it = gh_u, gh_v

        if self.params.name == "fb":
            eta_new = s.eta - dt * div_eta(dz * np.sum(u_it, axis=0),
                                           dz * np.sum(v_it, axis=0), grid)

        # Vertical velocity diagnosed from the updated horizontal flow, then
        # buoyancy stepped forward-backward against it (mode-split style).
        w = w_from_divergence(div_eta(u_it, v_it, grid), dz)
        return self._advance_tracers(s, u_it, v_it, w, eta_new)

    def integrate(self, state: State3D, n_steps: int) -> State3D:
        for _ in range(n_steps):
            state = self.step(state)
        return state

    def diagnose_w(self, state: State3D) -> np.ndarray:
        return w_from_divergence(div_eta(state.u, state.v, self.grid), self.grid.dz)
