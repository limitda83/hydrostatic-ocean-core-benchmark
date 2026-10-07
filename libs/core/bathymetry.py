#########################################################################
#  Module: bathymetry                                                   #
#  Description: Bathymetry generators and roughness metrics of          #
#               docs/03_discretization_spec.md S10.2. Roughness is a    #
#               controlled experimental factor (axis H), so every       #
#               generator is deterministic and reports both roughness   #
#               measures used in the literature.                        #
#  Pipeline: config -> bathymetry -> domain -> model3d_v05              #
#########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from libs.core.grid import CGrid


@dataclass(frozen=True)
class Roughness:
    """The two roughness measures reported with every topography."""

    r_std: float      # std(H) / mean(H)
    rx0: float        # Haney number max |dH| / (2 H_mean_face)
    h_ratio: float    # H_max / H_min

    def as_dict(self) -> dict[str, float]:
        return {"r_std": self.r_std, "rx0": self.rx0, "h_ratio": self.h_ratio}


def roughness(H: np.ndarray, mask: np.ndarray | None = None) -> Roughness:
    """Amplitude and slope roughness of a bathymetry field."""
    wet = np.ones_like(H, dtype=bool) if mask is None else mask.astype(bool)
    h = H[wet]
    r_std = float(np.std(h) / np.mean(h))
    rx = 0.0
    for axis in (-1, -2):
        a, b = H, np.roll(H, -1, axis=axis)
        both = wet & np.roll(wet, -1, axis=axis)
        if np.any(both):
            rx = max(rx, float(np.max(np.abs(a - b)[both] / (a + b)[both])))
    return Roughness(r_std=r_std, rx0=rx,
                     h_ratio=float(np.max(h) / np.min(h)))


def _spectral_field(nx: int, ny: int, slope: float, seed: int,
                    kmax: int = 0) -> np.ndarray:
    """Zero-mean periodic random field with amplitude spectrum k^-slope.

    ``kmax`` band-limits the field to |k| <= kmax modes per direction. Without
    it the field gains new small scales at every resolution, so a grid
    refinement changes the topography as well as the discretisation and the
    two effects cannot be separated: measured rx0 for the same nominal
    roughness moved from 0.0260 at nx=512 to 0.0236 at nx=1024. Set it for
    any sweep over nx; leave it 0 to compare roughnesses at fixed nx.

    The phases are drawn on a fixed kmax-sized lattice so that the SAME
    physical field comes out at every resolution.
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kk = np.hypot(*np.meshgrid(kx, ky, indexing="xy"))
    amp = np.zeros_like(kk)
    nz_ = kk > 0
    amp[nz_] = kk[nz_] ** (-slope)
    if kmax > 0:
        amp[kk > kmax] = 0.0
        # Draw the phases on the band-limited lattice only, and place them by
        # (kx, ky) so the same mode gets the same phase at any nx.
        phase = np.zeros_like(kk)
        m = int(kmax)
        lattice = rng.uniform(0.0, 2.0 * np.pi, size=(2 * m + 1, 2 * m + 1))
        ix = np.round(np.meshgrid(kx, ky, indexing="xy")[0]).astype(int)
        iy = np.round(np.meshgrid(kx, ky, indexing="xy")[1]).astype(int)
        sel = (np.abs(ix) <= m) & (np.abs(iy) <= m)
        phase[sel] = lattice[iy[sel] + m, ix[sel] + m]
    else:
        phase = rng.uniform(0.0, 2.0 * np.pi, size=kk.shape)
    field = np.fft.ifft2(amp * np.exp(1j * phase)).real
    return field - field.mean()


def build_bathymetry(grid: CGrid, kind: str, H0: float, *,
                     h_rel: float = 0.9, length: float = 0.0,
                     slope: float = 0.5, spectrum_slope: float = 1.5,
                     spectrum_kmax: int = 0,
                     r_target: float = 0.1, seed: int = 20260911,
                     island: bool = False
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (H[ny,nx], mask[ny,nx]) for the named topography (spec S10.2)."""
    x, y = grid.coords("eta")
    Lx, Ly = grid.Lx, grid.Ly
    L = length if length > 0.0 else 0.1 * Lx

    if kind == "flat":
        H = np.full(grid.shape, H0)
    elif kind == "slope":
        H = H0 * (1.0 - slope * x / Lx)
    elif kind == "seamount":
        # Beckmann & Haidvogel (1993): a Gaussian seamount at the centre.
        r2 = (x - 0.5 * Lx) ** 2 + (y - 0.5 * Ly) ** 2
        H = H0 * (1.0 - h_rel * np.exp(-r2 / L ** 2))
    elif kind == "ridge":
        H = H0 * (1.0 - h_rel * np.exp(-((x - 0.5 * Lx) / L) ** 2))
    elif kind == "rough":
        # Scale a fixed spectral field so that std(H)/mean(H) hits r_target.
        f = _spectral_field(grid.nx, grid.ny, spectrum_slope, seed,
                            spectrum_kmax)
        f = f / np.std(f)
        H = H0 * (1.0 + r_target * f)
        # Keep the column positive with a floor at 10% of H0.
        H = np.maximum(H, 0.1 * H0)
    else:
        raise ValueError(f"unknown bathymetry '{kind}' (expected flat|slope|"
                         "seamount|ridge|rough)")

    mask = np.ones(grid.shape, dtype=np.float64)
    if island:
        # A square island in the middle third, for the closed-boundary tests.
        r = (np.abs(x - 0.5 * Lx) < 0.1 * Lx) & (np.abs(y - 0.5 * Ly) < 0.1 * Ly)
        mask[r] = 0.0
    return H, mask


def bathymetry_from_config(grid: CGrid, config) -> tuple[np.ndarray, np.ndarray]:
    """Build the bathymetry named in config (R3: no hard-coded parameters)."""
    g = lambda k, d: config.get(f"bathymetry.{k}", d)   # noqa: E731
    return build_bathymetry(
        grid, str(g("kind", "flat")), float(config.get("physics.H")),
        h_rel=float(g("h_rel", 0.9)), length=float(g("length", 0.0)),
        slope=float(g("slope", 0.5)),
        spectrum_slope=float(g("spectrum_slope", 1.5)),
        spectrum_kmax=int(g("spectrum_kmax", 0)),
        r_target=float(g("r_target", 0.1)), seed=int(g("seed", 20260911)),
        island=bool(g("island", False)))
