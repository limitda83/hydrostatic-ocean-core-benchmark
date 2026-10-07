#########################################################################
#  Module: model3d_v05                                                  #
#  Description: The v0.5 core (docs/03_discretization_spec.md S10):     #
#               real bathymetry, z-level/z*/sigma vertical coordinates, #
#               closed and Flather-radiating boundaries and a nonlinear #
#               equation of state, on top of the Casulli semi-implicit  #
#               and split-explicit schemes of S7 and S8.                #
#               It is a separate module from model3d.py on purpose:     #
#               V5-1 checks the two against each other on the           #
#               configuration where they must agree (R10).              #
#  Pipeline: domain/eos/solvers_var/vertical_var -> model3d_v05         #
#########################################################################

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from libs.core.advection import laplacian_h, momentum_advection
from libs.core.advection_v06 import kappa_face, kappa_face_z, momentum_advection_up3
from libs.core.closure import TKEParams, tke_step
from libs.core.domain import Domain
from libs.core.eos import FLOPS_PER_CELL, buoyancy
from libs.core.model3d import Physics3D, State3D
from libs.core.operators_masked import (avg_u_to_v_m, avg_v_to_u_m, div_m,
                                        gradx_u_m, grady_v_m, laplacian_h_m)
from libs.core.schemes import SchemeParams
from libs.core.solvers_var import HelmholtzVar
from libs.core.vertical import thomas_batched
from libs.core.vertical_var import (apply_diffusion_var, buoyancy_potential_var,
                                    diffusion_coeffs_var, remove_depth_mean_var,
                                    w_at_centres_var,
                                    w_from_transport_divergence)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhysicsV05:
    """v0.4 physics plus the v0.5 additions (spec S10.6, S10.4)."""

    base: Physics3D
    eos: str = "linear"          # linear | seos | teos10
    eta_ext: float = 0.0         # Flather reference elevation [m]
    u_ext: float = 0.0           # Flather reference normal transport [m2 s-1]
    n_relax: int = 0             # sponge width in cells for the open boundary
    tau_relax: float = 3600.0    # innermost sponge time scale [s]
    # "remove" is the mode-split convention of S7.2 and the condition under
    # which the linear internal-wave solutions of S7.6 are exact. It is only
    # neutral on a flat bottom: over topography the column mean of Phi varies
    # horizontally, so removing it invents a pressure gradient. "keep" uses
    # the full Phi and is the setting the realistic v0.5 cases run under.
    pgf: str = "remove"          # remove | keep
    # False reproduces the naive along-coordinate difference, kept as a
    # measurable axis: it is what "no pressure-gradient scheme" costs.
    pgf_correction: bool = True
    # spec v0.6 (S11.2): vertical turbulence closure. 'none' reproduces v0.5.
    closure: str = "none"        # none | tke
    c_k: float = 0.1
    c_eps: float = 0.7
    pr_t: float = 1.0
    kappa_vk: float = 0.4
    z_0: float = 0.1
    e_min: float = 1.0e-6
    n2_min: float = 1.0e-8
    l_min: float = 0.01
    e_bb: float = 3.75
    # S11.2 mixing-length form: "integral" is the O(nz^2) energy budget of
    # Gaspar et al. (1990); "recursive" the two O(nz) sweeps of NEMO nn_mxl=2.
    mxl: str = "integral"

    def __getattr__(self, name):     # delegate the v0.4 parameters
        return getattr(self.base, name)


class Stepper3DV05:
    """One-step v0.5 integrator over a Domain."""

    def __init__(self, domain: Domain, physics: PhysicsV05,
                 params: SchemeParams, dt: float) -> None:
        params.validate()
        self.domain = domain
        self.grid = domain.grid
        self.physics = physics
        self.params = params
        self.dt = float(dt)
        self.solver_iterations = 0
        self.solver_failures = 0
        self.solver_rebuilds = 0
        self.tridiagonal_solves = 0
        self.barotropic_substeps = 0
        self.eos_cells = 0

        nz, ny, nx = self.grid.shape3d
        self.nu = np.full((nz + 1, ny, nx), physics.nu)
        self.kappa = np.full((nz + 1, ny, nx), physics.kappa)
        # Only the theta scheme owns an elliptic solve; fb and split-explicit
        # never touch params.solver, so a leftover 'fft' default must not stop
        # them from running over topography.
        if (params.name == "theta" and params.solver == "fft"
                and not domain.is_flat_periodic):
            raise ValueError(
                "solver 'fft' needs a flat bottom and periodic boundaries; "
                "with topography the Helmholtz operator is variable-coefficient "
                "(spec S10.5). Use pcg_jacobi | pcg_rbgs | rbgs | multigrid.")

        self._build_geometry(np.zeros(self.grid.shape))

        if params.name == "split_explicit":
            # Barotropic CFL uses the DEEPEST column: that column sets the
            # fastest surface gravity wave in the domain.
            h_max = float(np.max(domain.H[domain.mask > 0]))
            c = np.sqrt(physics.g * h_max)
            dt_baro = 1.0 / (c * np.sqrt(1.0 / self.grid.dx**2
                                         + 1.0 / self.grid.dy**2))
            self.n_split = (params.n_split if params.n_split > 0
                            else max(1, int(np.ceil(1.2 * self.dt / dt_baro))))
            if self.dt / self.n_split > dt_baro:
                raise ValueError(
                    f"n_split={self.n_split} leaves a barotropic substep of "
                    f"{self.dt / self.n_split:.1f}s above the limit {dt_baro:.1f}s")
        else:
            self.n_split = 0

    # -------------------------------------------------------------- geometry
    def _build_geometry(self, eta: np.ndarray) -> None:
        """Layer thickness, tridiagonal coefficients and the Helmholtz
        coefficients for the current free surface. z-level geometry is
        time-independent, so this runs once; z*/sigma rebuild every step,
        which is the cost of a moving vertical coordinate (measured as
        `geometry_rebuilds` in the metrics)."""
        d = self.domain
        self.dz3 = d.layer_thickness(eta)
        self.dz3u, self.dz3v = d.face_thickness(self.dz3)
        self.z_centre = d.depth_at_centres(eta)
        self.solver = None
        self._build_coefficients()

    def _build_coefficients(self) -> None:
        """Tridiagonal coefficients, q = A^-1 1 and the Helmholtz operator for
        the CURRENT vertical viscosity/diffusivity. Runs once for constant
        coefficients and every step under the TKE closure (spec S11.5)."""
        d = self.domain
        thv = self.params.theta_v
        self.tri_m = diffusion_coeffs_var(self.nu, self.dz3, self.dt, thv,
                                          self.physics.bottom_drag, d.mask3)
        self.tri_b = diffusion_coeffs_var(self.kappa, self.dz3, self.dt, thv,
                                          0.0, d.mask3)
        # q solves A q = 1: the effective depth weighting of spec S7.5.
        self.q = thomas_batched(*self.tri_m, np.ones(self.grid.shape3d))
        self.tridiagonal_solves += 1
        qu = np.minimum(self.q, np.roll(self.q, -1, axis=-1))
        qv = np.minimum(self.q, np.roll(self.q, -1, axis=-2))
        self.Ku = np.sum(self.dz3u * qu, axis=0) * d.mask_u
        self.Kv = np.sum(self.dz3v * qv, axis=0) * d.mask_v
        if self.params.name == "theta":
            if self.solver is None:
                coef = self.physics.g * self.params.theta**2 * self.dt**2
                self.solver = HelmholtzVar(self.grid, coef, self.Ku, self.Kv,
                                           d.mask, kind=self.params.solver,
                                           rtol=self.params.rtol,
                                           max_iter=self.params.max_iter)
            else:
                self.solver.update(self.Ku, self.Kv)
                self.solver_rebuilds += 1

    def _closure_step(self, s: State3D) -> np.ndarray:
        """spec S11.2: advance the TKE from the current state and refresh the
        eddy viscosity/diffusivity and every coefficient that depends on it."""
        p, d = self.physics, self.domain
        tke = TKEParams(p.c_k, p.c_eps, p.pr_t, p.kappa_vk, p.z_0, p.e_min,
                        p.n2_min, p.l_min, p.e_bb, p.mxl)
        e = s.e if s.e is not None else np.full((self.grid.nz + 1,) + self.grid.shape, p.e_min)
        ustar2 = np.full(self.grid.shape, np.hypot(p.tau_x, p.tau_y) / p.rho0)
        e_new, K_m, K_h = tke_step(e, s.u, s.v, s.b, self.dz3, d.mask3, ustar2, self.dt,
                                   tke, p.nu, p.kappa, self.nu, self.kappa)
        self.nu, self.kappa = K_m, K_h
        self.tridiagonal_solves += 1
        self._build_coefficients()
        return e_new

    @property
    def moving_coordinate(self) -> bool:
        return self.domain.vcoord in ("zstar", "sigma")

    # -------------------------------------------------------------- physics
    def buoyancy_of(self, T: np.ndarray, S: np.ndarray) -> np.ndarray:
        p = self.physics
        self.eos_cells += T.size
        return buoyancy(p.eos, T, S, self.z_centre, g=p.g, rho0=p.rho0,
                        alpha_T=p.alpha_T, beta_S=p.beta_S, T0=p.T0, S0=p.S0)

    def _pressure_gradients(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Baroclinic pressure gradient at a common depth (spec S10.7).

        The naive along-coordinate difference Phi[k,i+1] - Phi[k,i] compares
        pressures at two DIFFERENT depths whenever the layer interfaces are
        not horizontal - which is every terrain-following coordinate, and
        also every z-level bottom partial cell. With a uniform stratification
        over a seamount that error alone produced 74 cm/s of spurious flow in
        a hundred steps.

        The fix is the standard one: extrapolate each column's potential to
        the depth of the face, using dPhi/dz = b, before differencing. The
        remainder is O(dz^2 N^2) instead of O(dz b), and the same expression
        serves z-level partial cells, z* and sigma alike, so the coordinates
        can be compared without also comparing two different algorithms.
        """
        d, grid = self.domain, self.grid
        phi = buoyancy_potential_var(b, self.dz3)
        if self.physics.pgf == "remove":
            phi = remove_depth_mean_var(phi, self.dz3)
        if not self.physics.pgf_correction:
            return (gradx_u_m(phi, grid, d.mask3u) * d.mask3u,
                    grady_v_m(phi, grid, d.mask3v) * d.mask3v)

        zc = self.z_centre
        pg = []
        for axis, fmask, delta in ((-1, d.mask3u, grid.dx),
                                   (-2, d.mask3v, grid.dy)):
            phi_r, b_r, z_r = (np.roll(phi, -1, axis=axis),
                               np.roll(b, -1, axis=axis),
                               np.roll(zc, -1, axis=axis))
            z_face = 0.5 * (zc + z_r)
            left = phi + b * (z_face - zc)
            right = phi_r + b_r * (z_face - z_r)
            pg.append(fmask * (right - left) / delta)
        return pg[0], pg[1]

    def _w_diag(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Diagnostic vertical velocity from the layer transport divergence."""
        return w_from_transport_divergence(
            div_m(self.dz3u * u, self.dz3v * v, self.grid, self.domain.mask3))

    def _solve_column(self, coeffs, rhs):
        self.tridiagonal_solves += 1
        return thomas_batched(*coeffs, rhs)

    def _momentum_extras(self, u, v, w):
        d = self.domain
        ex_u = np.zeros_like(u)
        ex_v = np.zeros_like(v)
        if self.params.advection in ("up3", "up3_tvd"):
            adv_u, adv_v = momentum_advection_up3(
                u, v, w, self.grid.dx, self.grid.dy, self.grid.dz, d.mask3u, d.mask3v,
                self.params.advection == "up3_tvd")
            ex_u -= adv_u * d.mask3u
            ex_v -= adv_v * d.mask3v
        elif self.params.advection != "none":
            adv_u, adv_v = momentum_advection(u, v, w, self.grid)
            ex_u -= adv_u * d.mask3u
            ex_v -= adv_v * d.mask3v
        if self.physics.A_h > 0.0:
            ex_u += self.physics.A_h * laplacian_h_m(u, self.grid, d.mask3u,
                                                     d.mask3v, d.mask3u)
            ex_v += self.physics.A_h * laplacian_h_m(v, self.grid, d.mask3u,
                                                     d.mask3v, d.mask3v)
        return ex_u, ex_v

    def _flather(self, eta: np.ndarray) -> dict:
        """Flather radiation fluxes on the four domain edges (spec S10.4).

        Returned as separate boundary transports rather than written into the
        interior face array. On the periodic index layout the face west of
        column 0 and the face east of column nx-1 are the SAME array slot, so
        writing a radiating flux there would connect the two ends of the
        domain to each other - which is what made the first version of this
        test diverge to 1e75.
        """
        d, p = self.domain, self.physics
        bnd = {}
        if d.bc_x == "open":
            Hw = np.maximum(d.H[:, 0], 1.0)
            He = np.maximum(d.H[:, -1], 1.0)
            bnd["west"] = (p.u_ext
                           - np.sqrt(p.g * Hw) * (eta[:, 0] - p.eta_ext))
            bnd["east"] = (p.u_ext
                           + np.sqrt(p.g * He) * (eta[:, -1] - p.eta_ext))
        if d.bc_y == "open":
            Hs = np.maximum(d.H[0, :], 1.0)
            Hn = np.maximum(d.H[-1, :], 1.0)
            bnd["south"] = (p.u_ext
                            - np.sqrt(p.g * Hs) * (eta[0, :] - p.eta_ext))
            bnd["north"] = (p.u_ext
                            + np.sqrt(p.g * Hn) * (eta[-1, :] - p.eta_ext))
        return bnd

    def _div_bt(self, U: np.ndarray, V: np.ndarray, eta: np.ndarray
                ) -> np.ndarray:
        """Barotropic divergence including any open-boundary flux."""
        d = self.domain
        div = div_m(U, V, self.grid, d.mask)
        if d.bc_x != "open" and d.bc_y != "open":
            return div
        b = self._flather(eta)
        div = div.copy()
        if "west" in b:
            div[:, 0] += (U[:, 0] - b["west"]) / self.grid.dx - (
                (U[:, 0] - U[:, -1]) / self.grid.dx)
            div[:, -1] += (b["east"] - U[:, -2]) / self.grid.dx - (
                (U[:, -1] - U[:, -2]) / self.grid.dx)
        if "south" in b:
            div[0, :] += (V[0, :] - b["south"]) / self.grid.dy - (
                (V[0, :] - V[-1, :]) / self.grid.dy)
            div[-1, :] += (b["north"] - V[-2, :]) / self.grid.dy - (
                (V[-1, :] - V[-2, :]) / self.grid.dy)
        return div * d.mask

    # ------------------------------------------------------------- stepping
    def _barotropic(self, U0, V0, eta, fx, fy):
        d, p = self.domain, self.physics
        grid, ddt = self.grid, self.dt / self.n_split
        U, V, e = U0, V0, eta
        acc_u, acc_v = np.zeros_like(U0), np.zeros_like(V0)
        tc, npic = self.params.theta_cor, self.params.n_picard
        for _ in range(self.n_split):
            av_v, av_u = avg_v_to_u_m(V, d.mask_v, d.mask_u), avg_u_to_v_m(U, d.mask_u, d.mask_v)
            gxe = gradx_u_m(e, grid, d.mask_u)
            gye = grady_v_m(e, grid, d.mask_v)
            Ui, Vi = U, V
            for _ in range(npic):
                v_cor = (1.0 - tc) * av_v + tc * avg_v_to_u_m(Vi, d.mask_v, d.mask_u)
                u_cor = (1.0 - tc) * av_u + tc * avg_u_to_v_m(Ui, d.mask_u, d.mask_v)
                Ui = (U + ddt * (p.f * v_cor - p.g * self.Ku * gxe + fx)) * d.mask_u
                Vi = (V + ddt * (-p.f * u_cor - p.g * self.Kv * gye + fy)) * d.mask_v
            U, V = Ui, Vi
            e = e - ddt * self._div_bt(U, V, e)
            acc_u += U
            acc_v += V
        self.barotropic_substeps += self.n_split
        if self.params.barotropic_average == "mean":
            return acc_u / self.n_split, acc_v / self.n_split, e
        return U, V, e

    def step(self, s: State3D) -> State3D:
        d, p, grid = self.domain, self.physics, self.grid
        dt, th, tc, thv = (self.dt, self.params.theta, self.params.theta_cor,
                           self.params.theta_v)
        # Forward-backward applies the FULL explicit surface gradient (spec
        # S3.2); theta belongs to the theta scheme only. Reading the configured
        # theta here gave fb half the barotropic pressure gradient (docs/90 N14).
        if self.params.name != "theta":
            th = 0.0
        if self.moving_coordinate:
            self._build_geometry(s.eta)
        e_new = self._closure_step(s) if p.closure == "tke" else s.e
        self._e_new = e_new
        dz3, mask3 = self.dz3, d.mask3

        pgx, pgy = self._pressure_gradients(s.b)
        av_prev_v = avg_v_to_u_m(s.v, d.mask3v, d.mask3u)
        av_prev_u = avg_u_to_v_m(s.u, d.mask3u, d.mask3v)
        tau_u = np.full(grid.shape, p.tau_x / p.rho0) * d.mask3u[0]
        tau_v = np.full(grid.shape, p.tau_y / p.rho0) * d.mask3v[0]
        dzu_diff = apply_diffusion_var(s.u, self.nu, self.dz3u, tau_u,
                                       p.bottom_drag, mask3)
        dzv_diff = apply_diffusion_var(s.v, self.nu, self.dz3v, tau_v,
                                       p.bottom_drag, mask3)
        w_old = self._w_diag(s.u, s.v)
        ex_u, ex_v = self._momentum_extras(s.u, s.v, w_old)
        stress_u = np.zeros(grid.shape3d)
        stress_v = np.zeros(grid.shape3d)
        inv_dzu = np.where(self.dz3u[0] > 0, 1.0 / np.where(self.dz3u[0] > 0, self.dz3u[0], 1.0), 0.0)
        inv_dzv = np.where(self.dz3v[0] > 0, 1.0 / np.where(self.dz3v[0] > 0, self.dz3v[0], 1.0), 0.0)
        stress_u[0] = thv * dt * tau_u * inv_dzu
        stress_v[0] = thv * dt * tau_v * inv_dzv
        stress_u = stress_u + dt * ex_u
        stress_v = stress_v + dt * ex_v

        if self.params.name == "split_explicit":
            return self._step_split(s, pgx, pgy, av_prev_u, av_prev_v,
                                    dzu_diff, dzv_diff, stress_u, stress_v)

        gx_eta = gradx_u_m(s.eta, grid, d.mask_u)[None, :, :] * d.mask3u
        gy_eta = grady_v_m(s.eta, grid, d.mask_v)[None, :, :] * d.mask3v
        U_old = np.sum(self.dz3u * s.u, axis=0)
        V_old = np.sum(self.dz3v * s.v, axis=0)
        # The open-boundary flux is lagged for the semi-implicit scheme: a
        # Flather condition ties U to eta, so making it implicit would add a
        # boundary term to the Helmholtz operator. Flather belongs to the
        # barotropic mode, where split_explicit treats it exactly.
        div_old = self._div_bt(U_old, V_old, s.eta)

        u_it, v_it, eta_new = s.u, s.v, s.eta
        for _ in range(self.params.n_picard):
            v_cor = (1.0 - tc) * av_prev_v + tc * avg_v_to_u_m(v_it, d.mask3v, d.mask3u)
            u_cor = (1.0 - tc) * av_prev_u + tc * avg_u_to_v_m(u_it, d.mask3u, d.mask3v)
            gu = (s.u + dt * (p.f * v_cor + pgx) - p.g * dt * (1.0 - th) * gx_eta
                  + (1.0 - thv) * dt * dzu_diff)
            gv = (s.v + dt * (-p.f * u_cor + pgy) - p.g * dt * (1.0 - th) * gy_eta
                  + (1.0 - thv) * dt * dzv_diff)
            gh_u = self._solve_column(self.tri_m, gu + stress_u) * mask3
            gh_v = self._solve_column(self.tri_m, gv + stress_v) * mask3

            if self.params.name == "theta":
                div_g = self._div_bt(np.sum(self.dz3u * gh_u, axis=0),
                                     np.sum(self.dz3v * gh_v, axis=0), s.eta)
                rhs = s.eta - dt * ((1.0 - th) * div_old + th * div_g)
                eta_new, report = self.solver.solve(rhs)
                self.solver_iterations += report.iterations
                self.solver_failures += 0 if report.converged else 1
                u_it = gh_u - p.g * dt * th * self.q * gradx_u_m(eta_new, grid, d.mask_u)[None, :, :] * d.mask3u
                v_it = gh_v - p.g * dt * th * self.q * grady_v_m(eta_new, grid, d.mask_v)[None, :, :] * d.mask3v
                u_it, v_it = u_it * mask3, v_it * mask3
            else:
                u_it, v_it = gh_u, gh_v

        if self.params.name == "fb":
            eta_new = s.eta - dt * self._div_bt(
                np.sum(self.dz3u * u_it, axis=0),
                np.sum(self.dz3v * v_it, axis=0), s.eta)
        w = self._w_diag(u_it, v_it)
        return self._advance_tracers(s, u_it, v_it, w, eta_new)

    def _step_split(self, s, pgx, pgy, av_prev_u, av_prev_v,
                    dzu_diff, dzv_diff, stress_u, stress_v):
        d, p, grid = self.domain, self.physics, self.grid
        dt, tc, thv = self.dt, self.params.theta_cor, self.params.theta_v
        mask3 = d.mask3
        U0 = np.sum(self.dz3u * s.u, axis=0)
        V0 = np.sum(self.dz3v * s.v, axis=0)
        u_it, v_it = s.u, s.v
        gh_u = gh_v = cor_x = cor_y = None
        for _ in range(self.params.n_picard):
            v_cor = (1.0 - tc) * av_prev_v + tc * avg_v_to_u_m(v_it, d.mask3v, d.mask3u)
            u_cor = (1.0 - tc) * av_prev_u + tc * avg_u_to_v_m(u_it, d.mask3u, d.mask3v)
            gu = s.u + dt * (p.f * v_cor + pgx) + (1.0 - thv) * dt * dzu_diff
            gv = s.v + dt * (-p.f * u_cor + pgy) + (1.0 - thv) * dt * dzv_diff
            gh_u = self._solve_column(self.tri_m, gu + stress_u) * mask3
            gh_v = self._solve_column(self.tri_m, gv + stress_v) * mask3
            u_it, v_it = gh_u, gh_v

        # The Coriolis removed from the barotropic forcing is computed with the
        # SAME 2D transport operator the barotropic step adds back (spec S8.3
        # step 3, variable-depth form). Removing the depth integral of the 3D
        # masked average instead left an O(1) mode inconsistency over steep
        # topography (docs/90 N15).
        Um, Vm = np.sum(self.dz3u * gh_u, axis=0), np.sum(self.dz3v * gh_v, axis=0)
        cor_x = p.f * ((1.0 - tc) * avg_v_to_u_m(V0, d.mask_v, d.mask_u)
                       + tc * avg_v_to_u_m(Vm, d.mask_v, d.mask_u))
        cor_y = -p.f * ((1.0 - tc) * avg_u_to_v_m(U0, d.mask_u, d.mask_v)
                        + tc * avg_u_to_v_m(Um, d.mask_u, d.mask_v))
        fx = (Um - U0) / dt - cor_x
        fy = (Vm - V0) / dt - cor_y
        U, V, eta_new = self._barotropic(U0, V0, s.eta, fx, fy)

        # Replace the depth mean with the barotropic result so that the
        # column transport matches exactly. The weights are the real layer
        # thicknesses, so partial cells are accounted for.
        Hu = np.sum(self.dz3u, axis=0)
        Hv = np.sum(self.dz3v, axis=0)
        inv_hu = np.where(Hu > 0, 1.0 / np.where(Hu > 0, Hu, 1.0), 0.0)
        inv_hv = np.where(Hv > 0, 1.0 / np.where(Hv > 0, Hv, 1.0), 0.0)
        # Face masks, not the cell mask: the barotropic transport must not be
        # written onto a wall face (a wet cell next to a partial-cell step);
        # with the cell mask it was, and polluted u by 300 % of its norm in the
        # closed seamount basin (docs/90 N15).
        u_new = (gh_u - np.sum(self.dz3u * gh_u, axis=0) * inv_hu + U * inv_hu) * d.mask3u
        v_new = (gh_v - np.sum(self.dz3v * gh_v, axis=0) * inv_hv + V * inv_hv) * d.mask3v
        w = self._w_diag(u_new, v_new)
        return self._advance_tracers(s, u_new, v_new, w, eta_new)

    # --------------------------------------------------------------- tracers
    def _advance_tracer(self, c, u, v, w, surf_flux, source=None):
        d, dt = self.domain, self.dt
        thv = self.params.theta_v
        tend = np.zeros_like(c) if source is None else source
        if self.params.advection != "none":
            tend = tend - self._tracer_advection(c, u, v, w)
        if self.physics.K_h > 0.0:
            tend = tend + self.physics.K_h * laplacian_h_m(
                c, self.grid, d.mask3u, d.mask3v, d.mask3)
        rhs = (c + dt * tend
               + (1.0 - thv) * dt * apply_diffusion_var(
                   c, self.kappa, self.dz3, surf_flux, 0.0, d.mask3))
        inv_top = np.where(self.dz3[0] > 0, 1.0 / np.where(self.dz3[0] > 0, self.dz3[0], 1.0), 0.0)
        rhs[0] = rhs[0] + thv * dt * surf_flux * inv_top
        return self._solve_column(self.tri_b, rhs) * d.mask3

    def _tracer_advection(self, c, u, v, w):
        """Flux-form advection with the face masks and real face thickness."""
        d, grid = self.domain, self.grid
        scheme = self.params.advection
        if scheme == "centered2":
            cx = 0.5 * (c + np.roll(c, -1, axis=-1))
            cy = 0.5 * (c + np.roll(c, -1, axis=-2))
        elif scheme in ("up3", "up3_tvd"):       # spec S11.3
            cx = kappa_face(c, u, -1, d.mask3, scheme == "up3_tvd")
            cy = kappa_face(c, v, -2, d.mask3, scheme == "up3_tvd")
        else:                                    # upwind1
            cx = np.where(u > 0, c, np.roll(c, -1, axis=-1))
            cy = np.where(v > 0, c, np.roll(c, -1, axis=-2))
        fx = self.dz3u * u * cx
        fy = self.dz3v * v * cy
        inv = np.where(self.dz3 > 0, 1.0 / np.where(self.dz3 > 0, self.dz3, 1.0), 0.0)
        hdiv = ((fx - np.roll(fx, 1, axis=-1)) / grid.dx
                + (fy - np.roll(fy, 1, axis=-2)) / grid.dy) * inv
        # Interface values. The SURFACE interface carries a flux too: with a
        # linear free surface w[0] = d(eta)/dt is the volume entering the top
        # cell, and dropping its tracer flux is exactly what breaks constancy
        # preservation (a uniform tracer then develops a 0.26 K spread).
        nz = c.shape[0]
        cf = np.zeros((nz + 1,) + c.shape[1:], dtype=c.dtype)
        if scheme in ("up3", "up3_tvd"):
            cf[1:nz] = kappa_face_z(c, w, d.mask3, scheme == "up3_tvd")
        else:
            cf[1:nz] = 0.5 * (c[:nz - 1] + c[1:nz])
        cf[0] = c[0]
        vdiv = (w[:nz] * cf[:nz] - w[1:] * cf[1:]) * inv
        return (hdiv + vdiv) * d.mask3

    def _advance_tracers(self, s, u, v, w, eta_new):
        d, p, dt = self.domain, self.physics, self.dt
        zero = np.zeros(self.grid.shape)
        if p.tracers == "TS":
            flux_T = np.full(self.grid.shape, p.q_heat / (p.rho0 * p.cp))
            flux_S = np.full(self.grid.shape, p.q_salt)
            T_new = self._advance_tracer(s.T, u, v, w, flux_T)
            S_new = self._advance_tracer(s.S, u, v, w, flux_S)
            # Dry cells hold T = S = 0, which the EOS would turn into a large
            # non-zero buoyancy. It never reaches a wet cell (every use is
            # thickness- or face-mask-weighted), but the state must be clean:
            # the R2 gate compares whole arrays, and it was the dry-cell
            # garbage that made b disagree by 0.9 while T and S matched to
            # 1e-17.
            b_new = self.buoyancy_of(T_new, S_new) * d.mask3
            return State3D(u=u, v=v, b=b_new, eta=eta_new, t=s.t + dt,
                           T=T_new, S=S_new, e=self._e_new)
        source = -p.N2 * w_at_centres_var(w)
        b_new = self._advance_tracer(s.b, u, v, w, zero, source=source)
        return State3D(u=u, v=v, b=b_new, eta=eta_new, t=s.t + dt, e=self._e_new)

    def integrate(self, state: State3D, n_steps: int) -> State3D:
        for _ in range(n_steps):
            state = self.step(state)
        return state

    def diagnose_w(self, state: State3D) -> np.ndarray:
        return self._w_diag(state.u, state.v)

    @property
    def eos_flops(self) -> int:
        return self.eos_cells * FLOPS_PER_CELL[self.physics.eos]
