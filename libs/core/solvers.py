#########################################################################
#  Module: solvers                                                      #
#  Description: Free-surface Helmholtz solvers (axis B) for             #
#               ( I - g H theta^2 dt^2 L ) eta = rhs.                   #
#               'fft' is exact on periodic grids; 'pcg_jacobi' is the   #
#               general matrix-free path that carries the global        #
#               reductions studied in RQ2.                              #
#  Pipeline: schemes -> solvers -> eta^{n+1}                            #
#########################################################################

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from libs.core.grid import CGrid
from libs.core.operators import laplacian_eta

LOGGER = logging.getLogger(__name__)


@dataclass
class SolveReport:
    """Per-step solver diagnostics, accumulated into metrics.json."""

    iterations: int = 0
    residual: float = 0.0
    converged: bool = True


class HelmholtzFFT:
    """Spectral solve, exact to round-off. Doubly periodic grids only."""

    kind = "fft"

    def __init__(self, grid: CGrid, coef: float) -> None:
        # coef = g * H * theta^2 * dt^2  in  A = I - coef * L
        self.grid = grid
        self.coef = coef
        theta_x = 2.0 * np.pi * np.arange(grid.nx // 2 + 1) / grid.nx
        theta_y = 2.0 * np.pi * np.arange(grid.ny) / grid.ny
        lam_x = 2.0 * (np.cos(theta_x) - 1.0) / grid.dx**2
        lam_y = 2.0 * (np.cos(theta_y) - 1.0) / grid.dy**2
        lam = lam_y[:, None] + lam_x[None, :]          # eigenvalues of L, <= 0
        self._symbol = 1.0 - coef * lam                # >= 1, always invertible

    def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        spectrum = np.fft.rfft2(rhs)
        eta = np.fft.irfft2(spectrum / self._symbol, s=self.grid.shape)
        return eta, SolveReport(iterations=0, residual=0.0, converged=True)


class HelmholtzPCG:
    """Matrix-free preconditioned conjugate gradient with Jacobi preconditioner.

    Two global reductions per iteration - the scalability bottleneck that
    RQ2 measures on GPU backends.
    """

    kind = "pcg_jacobi"

    def __init__(self, grid: CGrid, coef: float, rtol: float = 1e-12,
                 max_iter: int = 2000, precond: str = "jacobi") -> None:
        self.grid = grid
        self.coef = coef
        self.rtol = rtol
        self.max_iter = max_iter
        self.precond = precond
        self._inv_diag = 1.0 / (1.0 + coef * (2.0 / grid.dx**2 + 2.0 / grid.dy**2))
        self._masks = _colour_masks(grid.ny, grid.nx) if precond == "rbgs" else None

    def _precondition(self, r: np.ndarray) -> np.ndarray:
        """M^-1 r. The RBGS variant does a symmetric sweep (forward then
        backward), which is what keeps M symmetric and CG valid."""
        if self.precond == "jacobi":
            return self._inv_diag * r
        z = np.zeros_like(r)
        _rbgs_sweep(z, r, self.coef, self.grid.dx, self.grid.dy, self._masks)
        _rbgs_sweep(z, r, self.coef, self.grid.dx, self.grid.dy, self._masks,
                    reverse=True)
        return z

    def _apply(self, x: np.ndarray) -> np.ndarray:
        return x - self.coef * laplacian_eta(x, self.grid)

    def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        x = np.zeros_like(rhs)
        r = rhs - self._apply(x)
        norm_b = float(np.sqrt(np.sum(rhs * rhs)))
        if norm_b == 0.0:
            return x, SolveReport(0, 0.0, True)

        z = self._precondition(r)
        p = z.copy()
        rz = float(np.sum(r * z))

        for iteration in range(1, self.max_iter + 1):
            ap = self._apply(p)
            alpha = rz / float(np.sum(p * ap))
            x = x + alpha * p
            r = r - alpha * ap
            residual = float(np.sqrt(np.sum(r * r))) / norm_b
            if residual < self.rtol:
                return x, SolveReport(iteration, residual, True)
            z = self._precondition(r)
            rz_new = float(np.sum(r * z))
            p = z + (rz_new / rz) * p
            rz = rz_new

        LOGGER.error(
            f"PCG failed to converge: {self.max_iter} iterations, "
            f"residual={residual:.3e} > rtol={self.rtol:.3e}"
        )
        return x, SolveReport(self.max_iter, residual, False)


def _rbgs_sweep(x: np.ndarray, b: np.ndarray, coef: float, dx: float, dy: float,
                masks: tuple[np.ndarray, np.ndarray], reverse: bool = False) -> None:
    """One red-black Gauss-Seidel sweep, in place.

    On a 5-point stencil every neighbour of a red cell is black, so each colour
    updates fully in parallel: two kernels, zero reductions. That is the whole
    point of multi-colouring - plain Gauss-Seidel is sequential and unusable on
    a GPU. Periodic wrap-around only 2-colours consistently when nx and ny are
    even, which build_solver enforces.
    """
    cx, cy = coef / dx**2, coef / dy**2
    diag = 1.0 + 2.0 * cx + 2.0 * cy
    order = masks[::-1] if reverse else masks
    for m in order:
        nb = (cx * (np.roll(x, -1, axis=-1) + np.roll(x, 1, axis=-1))
              + cy * (np.roll(x, -1, axis=-2) + np.roll(x, 1, axis=-2)))
        x[m] = ((b + nb) / diag)[m]


def _apply(x: np.ndarray, coef: float, dx: float, dy: float) -> np.ndarray:
    """A x = x - coef * L x, at any level of the hierarchy."""
    lap = ((np.roll(x, -1, axis=-1) - 2.0 * x + np.roll(x, 1, axis=-1)) / dx**2
           + (np.roll(x, -1, axis=-2) - 2.0 * x + np.roll(x, 1, axis=-2)) / dy**2)
    return x - coef * lap


def _colour_masks(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    j, i = np.indices((ny, nx))
    red = ((i + j) % 2 == 0)
    return red, ~red


class HelmholtzRBGS:
    """Red-black Gauss-Seidel used directly as the solver.

    Included so the multi-colour smoother can be measured on its own. It is not
    competitive: Gauss-Seidel converges in O(cond) sweeps while CG needs
    O(sqrt(cond)), and cond grows as CFL^2 here. Its real use is as the
    smoother inside HelmholtzMultigrid.
    """

    kind = "rbgs"

    def __init__(self, grid: CGrid, coef: float, rtol: float = 1e-12,
                 max_iter: int = 20000, check_every: int = 10) -> None:
        self.grid, self.coef, self.rtol = grid, coef, rtol
        self.max_iter, self.check_every = max_iter, check_every
        self.masks = _colour_masks(grid.ny, grid.nx)

    def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        x = np.zeros_like(rhs)
        norm_b = float(np.sqrt(np.sum(rhs * rhs))) or 1.0
        residual = 0.0
        for it in range(1, self.max_iter + 1):
            _rbgs_sweep(x, rhs, self.coef, self.grid.dx, self.grid.dy, self.masks)
            if it % self.check_every == 0 or it == self.max_iter:
                r = rhs - _apply(x, self.coef, self.grid.dx, self.grid.dy)
                residual = float(np.sqrt(np.sum(r * r))) / norm_b
                if residual < self.rtol:
                    return x, SolveReport(it, residual, True)
        LOGGER.error(f"RBGS failed to converge: {self.max_iter} sweeps, "
                     f"residual={residual:.3e}")
        return x, SolveReport(self.max_iter, residual, False)


class HelmholtzMultigrid:
    """Geometric multigrid V-cycle with a red-black Gauss-Seidel smoother.

    This is where multi-colouring pays off. Each V-cycle costs a handful of
    smoothing sweeps - all local, no reductions - and the iteration count is
    essentially independent of the condition number, so it stays near O(10)
    where Jacobi-PCG needs hundreds at high CFL. On GPU that converts roughly
    3 synchronisations per CG iteration into one per V-cycle.

    Cell-centred coarsening by 2, full-weighting restriction and piecewise-
    constant prolongation (an adjoint pair). Levels stop where the grid stops
    being even, so nx = 2^k * small is what the method wants; nx = 100 only
    coarsens twice and the method degrades to a smoother on a 25x25 problem.
    """

    kind = "multigrid"

    def __init__(self, grid: CGrid, coef: float, rtol: float = 1e-12,
                 max_iter: int = 100, n_pre: int = 2, n_post: int = 2,
                 n_coarse: int = 40) -> None:
        self.grid, self.coef, self.rtol = grid, coef, rtol
        self.max_iter = max_iter
        self.n_pre, self.n_post, self.n_coarse = n_pre, n_post, n_coarse

        self.levels = []
        nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
        while True:
            self.levels.append({"nx": nx, "ny": ny, "dx": dx, "dy": dy,
                                "masks": _colour_masks(ny, nx)})
            if nx % 2 or ny % 2 or min(nx, ny) <= 4:
                break
            nx, ny, dx, dy = nx // 2, ny // 2, dx * 2, dy * 2
        self.n_levels = len(self.levels)

    @staticmethod
    def _restrict(r: np.ndarray) -> np.ndarray:
        ny, nx = r.shape
        return r.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))

    @staticmethod
    def _prolong(e: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(e, 2, axis=0), 2, axis=1)

    def _vcycle(self, x: np.ndarray, b: np.ndarray, lvl: int) -> np.ndarray:
        L = self.levels[lvl]
        if lvl == self.n_levels - 1:
            for _ in range(self.n_coarse):
                _rbgs_sweep(x, b, self.coef, L["dx"], L["dy"], L["masks"])
            return x
        for k in range(self.n_pre):
            _rbgs_sweep(x, b, self.coef, L["dx"], L["dy"], L["masks"],
                        reverse=bool(k % 2))
        r = b - _apply(x, self.coef, L["dx"], L["dy"])
        ec = self._vcycle(np.zeros((L["ny"] // 2, L["nx"] // 2)),
                          self._restrict(r), lvl + 1)
        x = x + self._prolong(ec)
        for k in range(self.n_post):
            _rbgs_sweep(x, b, self.coef, L["dx"], L["dy"], L["masks"],
                        reverse=not bool(k % 2))
        return x

    def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        x = np.zeros_like(rhs)
        norm_b = float(np.sqrt(np.sum(rhs * rhs))) or 1.0
        residual = 0.0
        for it in range(1, self.max_iter + 1):
            x = self._vcycle(x, rhs, 0)
            r = rhs - _apply(x, self.coef, self.grid.dx, self.grid.dy)
            residual = float(np.sqrt(np.sum(r * r))) / norm_b
            if residual < self.rtol:
                return x, SolveReport(it, residual, True)
        LOGGER.error(f"multigrid failed to converge: {self.max_iter} V-cycles, "
                     f"residual={residual:.3e}")
        return x, SolveReport(self.max_iter, residual, False)


def build_solver(kind: str, grid: CGrid, coef: float, rtol: float,
                 max_iter: int):
    """Factory for the elliptic solver named in config/schemes.toml."""
    if kind == "fft":
        if grid.bc_x != "periodic" or grid.bc_y != "periodic":
            raise ValueError("the fft solver requires doubly periodic boundaries")
        return HelmholtzFFT(grid, coef)
    if kind == "pcg_jacobi":
        return HelmholtzPCG(grid, coef, rtol=rtol, max_iter=max_iter)
    if kind in ("rbgs", "multigrid", "pcg_rbgs"):
        if grid.nx % 2 or grid.ny % 2:
            raise ValueError(f"{kind} needs even nx and ny for a consistent "
                             f"red-black colouring under periodic wrap; "
                             f"got {grid.nx}x{grid.ny}")
        if kind == "rbgs":
            return HelmholtzRBGS(grid, coef, rtol=rtol, max_iter=max_iter)
        if kind == "multigrid":
            return HelmholtzMultigrid(grid, coef, rtol=rtol, max_iter=max_iter)
        return HelmholtzPCG(grid, coef, rtol=rtol, max_iter=max_iter,
                            precond="rbgs")
    raise ValueError(f"unknown solver kind '{kind}' "
                     f"(expected fft|pcg_jacobi|pcg_rbgs|rbgs|multigrid)")
