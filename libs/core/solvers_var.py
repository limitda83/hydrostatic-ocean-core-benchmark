#########################################################################
#  Module: solvers_var                                                  #
#  Description: Variable-coefficient free-surface Helmholtz solvers of  #
#               docs/03_discretization_spec.md S10.5:                   #
#                   eta - c * D[ K G[eta] ] = rhs                       #
#               With real bathymetry K varies from face to face, the    #
#               operator stops being a convolution and the FFT solve of #
#               libs/core/solvers.py no longer applies. The iteration   #
#               count then becomes a function of the topography, which  #
#               is the measurement RQ6 is built on.                     #
#  Pipeline: domain -> solvers_var -> model3d_v05                       #
#########################################################################

from __future__ import annotations

import numpy as np

from libs.core.grid import CGrid
from libs.core.solvers import SolveReport, _colour_masks


class VarOperator:
    """A = I - c * D[K G[.]] on a masked C-grid.

    Ku[j,i] is the coefficient on the face between (j,i) and (j,i+1); a dry
    face carries Ku = 0, which is exactly the no-normal-flow wall condition
    and keeps A symmetric positive definite (spec S10.4).
    """

    def __init__(self, grid: CGrid, coef: float, Ku: np.ndarray,
                 Kv: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.grid, self.coef = grid, float(coef)
        self.Ku, self.Kv = np.ascontiguousarray(Ku), np.ascontiguousarray(Kv)
        self.mask = (np.ones(grid.shape) if mask is None
                     else np.ascontiguousarray(mask))
        idx2, idy2 = 1.0 / grid.dx**2, 1.0 / grid.dy**2
        self.cx = self.coef * idx2
        self.cy = self.coef * idy2
        # Diagonal: 1 + c*(Ku_e + Ku_w)/dx^2 + c*(Kv_n + Kv_s)/dy^2.
        self.diag = (1.0
                     + self.cx * (self.Ku + np.roll(self.Ku, 1, axis=-1))
                     + self.cy * (self.Kv + np.roll(self.Kv, 1, axis=-2)))
        # A dry cell is decoupled and solved as eta = rhs (which is 0 there).
        self.diag = np.where(self.mask > 0, self.diag, 1.0)
        self.inv_diag = 1.0 / self.diag

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = x * self.mask
        fx = self.Ku * (np.roll(x, -1, axis=-1) - x)
        fy = self.Kv * (np.roll(x, -1, axis=-2) - x)
        lap = ((fx - np.roll(fx, 1, axis=-1)) / self.grid.dx**2
               + (fy - np.roll(fy, 1, axis=-2)) / self.grid.dy**2)
        return (x - self.coef * lap) * self.mask + x * (1.0 - self.mask)

    def neighbour_sum(self, x: np.ndarray) -> np.ndarray:
        """The off-diagonal part, -(A - diag) x, used by the smoothers."""
        return (self.cx * (self.Ku * np.roll(x, -1, axis=-1)
                           + np.roll(self.Ku, 1, axis=-1) * np.roll(x, 1, axis=-1))
                + self.cy * (self.Kv * np.roll(x, -1, axis=-2)
                             + np.roll(self.Kv, 1, axis=-2) * np.roll(x, 1, axis=-2)))


def _rbgs_sweep_var(x: np.ndarray, b: np.ndarray, op: VarOperator,
                    masks, reverse: bool = False) -> None:
    order = masks[::-1] if reverse else masks
    for colour in order:
        upd = op.inv_diag * (b + op.neighbour_sum(x))
        x[colour] = upd[colour]
    x *= op.mask


class HelmholtzVar:
    """PCG / RBGS / multigrid on the variable-coefficient operator."""

    def __init__(self, grid: CGrid, coef: float, Ku: np.ndarray, Kv: np.ndarray,
                 mask: np.ndarray | None = None, kind: str = "pcg_jacobi",
                 rtol: float = 1e-12, max_iter: int = 2000,
                 n_pre: int = 2, n_post: int = 2, min_size: int = 8) -> None:
        self.kind = kind
        self.op = VarOperator(grid, coef, Ku, Kv, mask)
        self.rtol, self.max_iter = rtol, max_iter
        self.grid = grid
        self.masks = _colour_masks(grid.ny, grid.nx)
        self.n_pre, self.n_post = n_pre, n_post
        self._min_size = min_size
        if kind == "multigrid":
            self.levels = self._build_levels(grid, coef, Ku, Kv,
                                             self.op.mask, min_size)
        elif kind not in ("pcg_jacobi", "pcg_rbgs", "rbgs"):
            raise ValueError(f"unknown variable-coefficient solver '{kind}' "
                             "(expected pcg_jacobi|pcg_rbgs|rbgs|multigrid)")

    def update(self, Ku: np.ndarray, Kv: np.ndarray) -> None:
        """Rebuild the operator (and every multigrid level) for new face
        coefficients - what a step-dependent vertical viscosity costs (S11.5)."""
        self.op = VarOperator(self.grid, self.op.coef, Ku, Kv, self.op.mask)
        if self.kind == "multigrid":
            self.levels = self._build_levels(self.grid, self.op.coef, Ku, Kv,
                                             self.op.mask, self._min_size)

    # ------------------------------------------------------------ multigrid
    @staticmethod
    def _build_levels(grid, coef, Ku, Kv, mask, min_size):
        """Coarse operators by face-coefficient agglomeration.

        A coarse face is the sum of the two fine faces it replaces (flux
        conservation), halved because the coarse face is twice as long; a
        coarse cell is wet if any of its four children is.
        """
        levels = [(grid, VarOperator(grid, coef, Ku, Kv, mask),
                   _colour_masks(grid.ny, grid.nx))]
        while (levels[-1][0].nx % 2 == 0 and levels[-1][0].ny % 2 == 0
               and levels[-1][0].nx // 2 >= min_size
               and levels[-1][0].ny // 2 >= min_size):
            g, op, _ = levels[-1]
            cg = CGrid(nx=g.nx // 2, ny=g.ny // 2, Lx=g.Lx, Ly=g.Ly,
                       bc_x=g.bc_x, bc_y=g.bc_y, nz=g.nz, depth=g.depth)
            # East face of a coarse cell = the two fine east faces on its
            # right-hand edge, averaged.
            fu = op.Ku
            cKu = 0.5 * (fu[0::2, 1::2] + fu[1::2, 1::2])
            fv = op.Kv
            cKv = 0.5 * (fv[1::2, 0::2] + fv[1::2, 1::2])
            cm = np.maximum.reduce([op.mask[0::2, 0::2], op.mask[0::2, 1::2],
                                    op.mask[1::2, 0::2], op.mask[1::2, 1::2]])
            levels.append((cg, VarOperator(cg, coef, cKu, cKv, cm),
                           _colour_masks(cg.ny, cg.nx)))
        return levels

    @staticmethod
    def _restrict(r):
        return 0.25 * (r[0::2, 0::2] + r[0::2, 1::2]
                       + r[1::2, 0::2] + r[1::2, 1::2])

    @staticmethod
    def _prolong(e):
        return np.repeat(np.repeat(e, 2, axis=0), 2, axis=1)

    def _vcycle(self, x, b, lvl):
        _, op, masks = self.levels[lvl]
        for _ in range(self.n_pre):
            _rbgs_sweep_var(x, b, op, masks)
        if lvl + 1 < len(self.levels):
            r = (b - op.apply(x)) * op.mask
            _, cop, cmasks = self.levels[lvl + 1]
            e = np.zeros(self.levels[lvl + 1][0].shape)
            e = self._vcycle(e, self._restrict(r), lvl + 1)
            x = x + self._prolong(e) * op.mask
        for _ in range(self.n_post):
            _rbgs_sweep_var(x, b, op, masks, reverse=True)
        return x

    # ---------------------------------------------------------------- solve
    def _precondition(self, r):
        if self.kind == "pcg_jacobi":
            return self.op.inv_diag * r
        z = np.zeros_like(r)
        _rbgs_sweep_var(z, r, self.op, self.masks)
        _rbgs_sweep_var(z, r, self.op, self.masks, reverse=True)
        return z

    def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        rhs = rhs * self.op.mask
        norm_b = float(np.sqrt(np.sum(rhs * rhs)))
        x = np.zeros_like(rhs)
        if norm_b == 0.0:
            return x, SolveReport(0, 0.0, True)
        tol = self.rtol * norm_b

        if self.kind in ("rbgs", "multigrid"):
            step = ((lambda x: self._vcycle(x, rhs, 0)) if self.kind == "multigrid"
                    else (lambda x: (_rbgs_sweep_var(x, rhs, self.op, self.masks),
                                     x)[1]))
            for it in range(1, self.max_iter + 1):
                x = step(x)
                res = float(np.sqrt(np.sum((rhs - self.op.apply(x)) ** 2)))
                if res <= tol:
                    return x, SolveReport(it, res / norm_b, True)
            return x, SolveReport(self.max_iter, res / norm_b, False)

        r = rhs.copy()
        z = self._precondition(r)
        p = z.copy()
        rz = float(np.sum(r * z))
        for it in range(1, self.max_iter + 1):
            Ap = self.op.apply(p)
            alpha = rz / float(np.sum(p * Ap))
            x += alpha * p
            r -= alpha * Ap
            res = float(np.sqrt(np.sum(r * r)))
            if res <= tol:
                return x, SolveReport(it, res / norm_b, True)
            z = self._precondition(r)
            rz_new = float(np.sum(r * z))
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x, SolveReport(self.max_iter, res / norm_b, False)
