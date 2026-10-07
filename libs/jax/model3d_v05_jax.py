#########################################################################
#  Module: model3d_v05_jax                                              #
#  Description: JAX implementation of the spec v0.5 full core (docs/03  #
#               S10) - z-level partial cells over bathymetry, land and  #
#               face masks, periodic or closed walls, flux-form          #
#               advection, T/S with three equations of state, the        #
#               common-depth pressure gradient, the variable-coefficient #
#               Helmholtz solve (PCG-Jacobi / PCG-RBGS / multigrid) and  #
#               forward-backward, theta and split-explicit stepping.     #
#               Operator-for-operator port of libs/core/model3d_v05.py;  #
#               the geometry is built once in NumPy (mirroring           #
#               mod_domain5.f90) and every step is one jitted function   #
#               whose state stays on the device.                         #
#  Pipeline: cfd_exp3d5_jax -> model3d_v05_jax -> state3d5.bin          #
#########################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from libs.core import eos as eos_mod

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------ geometry
def face_of(a: np.ndarray, b: np.ndarray, rule: str) -> np.ndarray:
    if rule == "mean":
        return 0.5 * (a + b)
    if rule == "harmonic":
        s = a + b
        return np.where(s > 0.0, 2.0 * a * b / np.where(s > 0.0, s, 1.0), 0.0)
    return np.minimum(a, b)


def build_geometry(h: np.ndarray, mask: np.ndarray, nz: int, face_rule: str,
                   min_partial: float, bc_x: str, bc_y: str) -> dict[str, np.ndarray]:
    """z-level partial-cell geometry, [k,j,i] layout (mod_domain5.f90)."""
    ny, nx = h.shape
    mask_u = mask * np.roll(mask, -1, axis=-1)
    mask_v = mask * np.roll(mask, -1, axis=-2)
    if bc_x != "periodic":
        mask_u[:, -1] = 0.0
    if bc_y != "periodic":
        mask_v[-1, :] = 0.0
    hmax = float(np.max(h[mask > 0]))
    dzr = hmax / nz
    dz3 = np.zeros((nz, ny, nx))
    edge = 0.0
    for k in range(nz):
        rem = h - edge
        th = np.clip(rem, 0.0, dzr)
        th = np.where((th > 0.0) & (th < min_partial * dzr), 0.0, th)
        dz3[k] = np.where(mask > 0.0, th, 0.0)
        edge += dzr
    mask3 = (dz3 > 0.0).astype(float)
    mask3u = mask3 * np.roll(mask3, -1, axis=-1) * mask_u[None]
    mask3v = mask3 * np.roll(mask3, -1, axis=-2) * mask_v[None]
    dz3u = mask3u * face_of(dz3, np.roll(dz3, -1, axis=-1), face_rule)
    dz3v = mask3v * face_of(dz3, np.roll(dz3, -1, axis=-2), face_rule)
    inv_dz3 = np.where(dz3 > 0.0, 1.0 / np.where(dz3 > 0.0, dz3, 1.0), 0.0)
    above = np.concatenate([np.zeros((1, ny, nx)), np.cumsum(dz3, axis=0)[:-1]], axis=0)
    zc = above + 0.5 * dz3
    hcu, hcv = np.sum(dz3u, axis=0), np.sum(dz3v, axis=0)
    inv_hcu = np.where(hcu > 0.0, 1.0 / np.where(hcu > 0.0, hcu, 1.0), 0.0)
    inv_hcv = np.where(hcv > 0.0, 1.0 / np.where(hcv > 0.0, hcv, 1.0), 0.0)
    return dict(h=h, mask=mask, mask_u=mask_u, mask_v=mask_v, dz3=dz3, dz3u=dz3u, dz3v=dz3v,
                mask3=mask3, mask3u=mask3u, mask3v=mask3v, inv_dz3=inv_dz3, zc=zc,
                inv_hcu=inv_hcu, inv_hcv=inv_hcv, hmax=hmax)


def _safe_np(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, 1.0 / np.where(x > 0.0, x, 1.0), 0.0)


def diffusion_coeffs(nu: float, dz3: np.ndarray, mask3: np.ndarray, dt: float, thv: float,
                     drag: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal A = I - thv dt Dz on partial cells (vertical_var.py)."""
    nz = dz3.shape[0]
    dzi = np.empty((nz + 1,) + dz3.shape[1:])
    dzi[0] = dz3[0]
    dzi[1:nz] = 0.5 * (dz3[:-1] + dz3[1:])
    dzi[nz] = dz3[-1]
    inv_dz = _safe_np(dz3)
    at = thv * dt * inv_dz * nu * _safe_np(dzi[:nz])
    ab = thv * dt * inv_dz * nu * _safe_np(dzi[1:])
    sub, sup, diag = -at, -ab, 1.0 + at + ab
    diag[0] = 1.0 + ab[0]
    sub[0] = 0.0
    below = np.concatenate([mask3[1:], np.zeros((1,) + mask3.shape[1:])], axis=0)
    bottom = (mask3 > 0) & (below == 0)
    diag = np.where(bottom, 1.0 + at + thv * dt * drag * inv_dz, diag)
    sup = np.where(bottom, 0.0, sup)
    dry = mask3 == 0
    return (np.where(dry, 0.0, sub), np.where(dry, 1.0, diag), np.where(dry, 0.0, sup))


def thomas_np(sub, diag, sup, rhs):
    nz = rhs.shape[0]
    c = np.empty_like(rhs)
    x = np.empty_like(rhs)
    c[0] = sup[0] / diag[0]
    x[0] = rhs[0] / diag[0]
    for k in range(1, nz):
        den = diag[k] - sub[k] * c[k - 1]
        c[k] = sup[k] / den
        x[k] = (rhs[k] - sub[k] * x[k - 1]) / den
    for k in range(nz - 2, -1, -1):
        x[k] -= c[k] * x[k + 1]
    return x


# ------------------------------------------------------------ jnp operators
def _safe(x):
    return jnp.where(x > 0.0, 1.0 / jnp.where(x > 0.0, x, 1.0), 0.0)


def thomas(sub, diag, sup, rhs):
    """Batched Thomas sweep as two lax.scan passes over k."""
    zeros = jnp.zeros_like(rhs[0])

    def fwd(carry, xs):
        c_prev, d_prev = carry
        s, g, u, r = xs
        den = g - s * c_prev
        c = u / den
        d = (r - s * d_prev) / den
        return (c, d), (c, d)

    _, (cs, ds) = lax.scan(fwd, (zeros, zeros), (sub, diag, sup, rhs))

    def bwd(x_next, xs):
        c, d = xs
        x = d - c * x_next
        return x, x

    _, x = lax.scan(bwd, zeros, (cs, ds), reverse=True)
    return x


def diffusion_coeffs_jnp(nu3, dz3, mask3, dt, thv, drag):
    """jnp twin of diffusion_coeffs (vertical_var.py) for interface arrays."""
    nz = dz3.shape[0]
    dzi = jnp.concatenate([dz3[:1], 0.5 * (dz3[:-1] + dz3[1:]), dz3[-1:]], axis=0)
    inv_dz = _safe(dz3)
    at = thv * dt * inv_dz * nu3[:nz] * _safe(dzi[:nz])
    ab = thv * dt * inv_dz * nu3[1:] * _safe(dzi[1:])
    sub, sup, diag = -at, -ab, 1.0 + at + ab
    diag = jnp.concatenate([(1.0 + ab[:1]), diag[1:]], axis=0)
    sub = jnp.concatenate([jnp.zeros_like(sub[:1]), sub[1:]], axis=0)
    below = jnp.concatenate([mask3[1:], jnp.zeros_like(mask3[:1])], axis=0)
    bottom = (mask3 > 0) & (below == 0)
    diag = jnp.where(bottom, 1.0 + at + thv * dt * drag * inv_dz, diag)
    sup = jnp.where(bottom, 0.0, sup)
    dry = mask3 == 0
    return (jnp.where(dry, 0.0, sub), jnp.where(dry, 1.0, diag), jnp.where(dry, 0.0, sup))


def kappa_face(c, vel, axis, wet, limited):
    """Third-order upwind-biased face value (spec S11.3, advection_v06.py)."""
    KAP = 1.0 / 3.0
    cm, cp, cpp = jnp.roll(c, 1, axis=axis), jnp.roll(c, -1, axis=axis), jnp.roll(c, -2, axis=axis)
    if limited:
        d_c = cp - c
        eps = jnp.where(d_c == 0.0, 1.0, d_c)
        r_pos = jnp.where(d_c == 0.0, 0.0, (c - cm) / eps)
        r_neg = jnp.where(d_c == 0.0, 0.0, (cpp - cp) / eps)
        phi_pos = jnp.maximum(0.0, jnp.maximum(jnp.minimum(2.0 * r_pos, 1.0), jnp.minimum(r_pos, 2.0)))
        phi_neg = jnp.maximum(0.0, jnp.maximum(jnp.minimum(2.0 * r_neg, 1.0), jnp.minimum(r_neg, 2.0)))
        f_pos = c + 0.5 * phi_pos * d_c
        f_neg = cp - 0.5 * phi_neg * d_c
    else:
        f_pos = c + 0.25 * ((1.0 - KAP) * (c - cm) + (1.0 + KAP) * (cp - c))
        f_neg = cp - 0.25 * ((1.0 - KAP) * (cpp - cp) + (1.0 + KAP) * (cp - c))
    f_pos = jnp.where(jnp.roll(wet, 1, axis=axis) > 0.0, f_pos, c)
    f_neg = jnp.where(jnp.roll(wet, -2, axis=axis) > 0.0, f_neg, cp)
    return jnp.where(vel > 0.0, f_pos, f_neg)


def kappa_face_z(c, w, wet3, limited):
    KAP = 1.0 / 3.0
    nz = c.shape[0]
    above, below = c[:-1], c[1:]
    pad = jnp.zeros_like(c[:1])
    far_up = jnp.concatenate([pad, c[:-2]], axis=0)
    far_dn = jnp.concatenate([c[2:], pad], axis=0)
    has_up = jnp.concatenate([jnp.zeros_like(wet3[:1]), wet3[:-2]], axis=0) > 0.0
    has_dn = jnp.concatenate([wet3[2:], jnp.zeros_like(wet3[:1])], axis=0) > 0.0
    d = above - below
    if limited:
        eps = jnp.where(d == 0.0, 1.0, d)
        r_up = jnp.where(d == 0.0, 0.0, (below - far_dn) / eps)
        r_dn = jnp.where(d == 0.0, 0.0, (far_up - above) / eps)
        phi_up = jnp.maximum(0.0, jnp.maximum(jnp.minimum(2.0 * r_up, 1.0), jnp.minimum(r_up, 2.0)))
        phi_dn = jnp.maximum(0.0, jnp.maximum(jnp.minimum(2.0 * r_dn, 1.0), jnp.minimum(r_dn, 2.0)))
        f_up = below + 0.5 * phi_up * d
        f_dn = above - 0.5 * phi_dn * d
    else:
        f_up = below + 0.25 * ((1.0 - KAP) * (below - far_dn) + (1.0 + KAP) * (above - below))
        f_dn = above - 0.25 * ((1.0 - KAP) * (far_up - above) + (1.0 + KAP) * (above - below))
    f_up = jnp.where(has_dn, f_up, below)
    f_dn = jnp.where(has_up, f_dn, above)
    return jnp.where(w[1:nz] > 0.0, f_up, f_dn)


def momentum_advection_up3(u, v, w, dx, dy, dz, mask3u, mask3v, limited):
    nz = u.shape[0]
    ubar_c = 0.5 * (jnp.roll(u, 1, axis=-1) + u)
    u_at_c = jnp.roll(kappa_face(u, jnp.roll(ubar_c, -1, axis=-1), -1, mask3u, limited), 1, axis=-1)
    adv_ux = (jnp.roll(ubar_c * u_at_c, -1, axis=-1) - ubar_c * u_at_c) / dx
    vbar = 0.5 * (v + jnp.roll(v, -1, axis=-1))
    fy_u = vbar * kappa_face(u, vbar, -2, mask3u, limited)
    adv_uy = (fy_u - jnp.roll(fy_u, 1, axis=-2)) / dy
    wbar_u = 0.5 * (w + jnp.roll(w, -1, axis=-1))
    fz_u = jnp.concatenate([jnp.zeros_like(u[:1]), wbar_u[1:nz] * kappa_face_z(u, wbar_u, mask3u, limited),
                            jnp.zeros_like(u[:1])], axis=0)
    adv_uz = (fz_u[:nz] - fz_u[1:]) / dz
    vbar_c = 0.5 * (jnp.roll(v, 1, axis=-2) + v)
    v_at_c = jnp.roll(kappa_face(v, jnp.roll(vbar_c, -1, axis=-2), -2, mask3v, limited), 1, axis=-2)
    adv_vy = (jnp.roll(vbar_c * v_at_c, -1, axis=-2) - vbar_c * v_at_c) / dy
    ubar = 0.5 * (u + jnp.roll(u, -1, axis=-2))
    fx_v = ubar * kappa_face(v, ubar, -1, mask3v, limited)
    adv_vx = (fx_v - jnp.roll(fx_v, 1, axis=-1)) / dx
    wbar_v = 0.5 * (w + jnp.roll(w, -1, axis=-2))
    fz_v = jnp.concatenate([jnp.zeros_like(v[:1]), wbar_v[1:nz] * kappa_face_z(v, wbar_v, mask3v, limited),
                            jnp.zeros_like(v[:1])], axis=0)
    adv_vz = (fz_v[:nz] - fz_v[1:]) / dz
    return adv_ux + adv_uy + adv_uz, adv_vx + adv_vy + adv_vz


def tke_step(e, u, v, b, dz3, mask3, ustar2, dt, p, nu_b, kappa_b, K_m_old, K_h_old,
             mxl: str = "integral"):
    """jnp twin of closure.py::tke_step (interfaces [nz+1, ny, nx])."""
    nz = dz3.shape[0]
    zi = jnp.concatenate([jnp.zeros_like(dz3[:1]), jnp.cumsum(dz3, axis=0)], axis=0)
    z_s = zi
    z_b = jnp.maximum(zi[-1][None] - zi, 0.0)
    wet = jnp.concatenate([jnp.zeros_like(e[:1]), mask3[:-1] * mask3[1:], jnp.zeros_like(e[:1])], axis=0)
    dzi = jnp.concatenate([dz3[:1], 0.5 * (dz3[:-1] + dz3[1:]), dz3[-1:]], axis=0)
    inv_dzi = _safe(dzi)
    uc = 0.5 * (u + jnp.roll(u, 1, axis=-1))
    vc = 0.5 * (v + jnp.roll(v, 1, axis=-2))
    z1 = jnp.zeros_like(e[:1])
    du = jnp.concatenate([z1, (uc[:-1] - uc[1:]) * inv_dzi[1:nz], z1], axis=0)
    dv = jnp.concatenate([z1, (vc[:-1] - vc[1:]) * inv_dzi[1:nz], z1], axis=0)
    db = jnp.concatenate([z1, (b[:-1] - b[1:]) * inv_dzi[1:nz], z1], axis=0)
    shear2 = (du ** 2 + dv ** 2) * wet
    n2 = db * wet
    # Gaspar lengths (closure.py::gaspar_lengths). A python loop over the nz
    # offsets unrolls into nz*2 copies of a [nz+1,ny,nx] graph - 28.8 GiB at
    # 2000^2 x 30 (docs/90 N21). lax.fori_loop keeps one body and carries the
    # travelled distance and the energy budget, so the peak is a few arrays.
    n2p = jnp.maximum(n2, 0.0)
    e_k = jnp.maximum(e, p["e_min"])
    kidx = jnp.arange(nz + 1)[:, None, None]
    if mxl == "recursive":
        # S11.2 `recursive`: buoyancy length capped by both walls, then limited
        # to grow no faster than the grid. The two sweeps are sequential in k,
        # so they are lax.scan (like thomas), not a fori_loop over offsets.
        l0 = jnp.sqrt(2.0 * e_k / jnp.maximum(n2, p["n2_min"]))
        l0 = jnp.minimum(l0, jnp.minimum(z_s + p["z_0"], z_b + p["z_0"]))

        def sweep(prev, xs):
            l_k, dzi_k = xs
            cur = jnp.minimum(l_k, prev + dzi_k)
            return cur, cur

        _, ldown = lax.scan(sweep, l0[0], (l0[1:], dzi[1:]))
        ldown = jnp.concatenate([l0[:1], ldown], axis=0)
        _, lup_rev = lax.scan(sweep, ldown[nz], (ldown[nz - 1::-1], dzi[nz:0:-1]))
        l_all = jnp.concatenate([lup_rev[::-1], ldown[nz:]], axis=0) * wet
        # The other three backends reach the same value as
        # max(l_min, sqrt(l_up*l_dn)) with l_up == l_dn == l. sqrt(l*l) == l
        # bit for bit over the whole normal range (checked on 8e6 samples at
        # four scales); it only breaks where l*l overflows or underflows, i.e.
        # |l| > 1.3e154 or < 1.5e-154, and a mixing length is O(1e-2..1e2) m.
        l_k = jnp.maximum(p["l_min"], l_all)
        l_eps = l_k
    else:
        l_k, l_eps = None, None

    def march(carry, j, sign):
        l, budget = carry
        shifted_n2 = jnp.roll(n2p, sign * j, axis=0)
        shifted_dz = jnp.roll(dzi, sign * j, axis=0)
        inside = (kidx - sign * j >= 0) & (kidx - sign * j <= nz)
        n2_j = jnp.where(inside, shifted_n2, 0.0)
        dz_j = jnp.where(inside, shifted_dz, 0.0)
        cost = n2_j * dz_j * (l + 0.5 * dz_j)
        full = cost <= budget
        a = jnp.where(n2_j > 0.0, n2_j, 1.0)
        s_part = jnp.sqrt(jnp.maximum(l * l + 2.0 * budget / a, 0.0)) - l
        step = jnp.where(full, dz_j, jnp.where(n2_j > 0.0, jnp.clip(s_part, 0.0, dz_j), dz_j))
        return (l + jnp.where(budget > 0.0, step, 0.0), jnp.where(full, budget - cost, 0.0))

    def body(j, carry):
        l_up, b_up, l_dn, b_dn = carry
        l_up, b_up = march((l_up, b_up), j, 1)      # upward: interface k-j
        l_dn, b_dn = march((l_dn, b_dn), j, -1)     # downward: interface k+j
        return (l_up, b_up, l_dn, b_dn)

    if mxl != "recursive":
        zero = jnp.zeros_like(e)
        l_up, _, l_dn, _ = lax.fori_loop(1, nz + 1, body, (zero, e_k, zero, e_k))
        l_up = jnp.minimum(l_up, z_s + p["z_0"]) * wet
        l_dn = jnp.minimum(l_dn, z_b + p["z_0"]) * wet
        l_k = jnp.maximum(p["l_min"], jnp.sqrt(l_up * l_dn))
        l_eps = jnp.maximum(p["l_min"], jnp.minimum(l_up, l_dn))
    prod = (K_m_old - nu_b) * shear2 - (K_h_old - kappa_b) * n2
    diss_coef = p["c_eps"] * jnp.sqrt(jnp.maximum(e, p["e_min"])) / l_eps
    K_lay = 0.5 * (K_m_old[:-1] + K_m_old[1:])
    inv_dz3 = _safe(dz3)
    a_up = jnp.concatenate([z1, dt * K_lay[:-1] * inv_dz3[:-1] * inv_dzi[1:nz], z1], axis=0)
    a_dn = jnp.concatenate([z1, dt * K_lay[1:] * inv_dz3[1:] * inv_dzi[1:nz], z1], axis=0)
    sub, sup = -a_up, -a_dn
    diag = 1.0 + a_up + a_dn + dt * diss_coef
    rhs = e + dt * prod
    bnd = wet <= 0.0
    diag = jnp.where(bnd, 1.0, diag); sub = jnp.where(bnd, 0.0, sub); sup = jnp.where(bnd, 0.0, sup)
    rhs = jnp.where(bnd, p["e_min"], rhs)
    e_surf = jnp.where(mask3[0] > 0.0, p["e_bb"] * ustar2, p["e_min"])
    rhs = jnp.concatenate([e_surf[None], rhs[1:]], axis=0)
    e_new = thomas(sub, diag, sup, rhs)
    e_new = jnp.maximum(e_new, p["e_min"])
    e_new = jnp.where(wet > 0.0, e_new, p["e_min"])
    e_new = jnp.concatenate([rhs[:1], e_new[1:]], axis=0)
    K_m = p["c_k"] * l_k * jnp.sqrt(e_new) * wet + nu_b
    K_h = (K_m - nu_b) / p["pr_t"] + kappa_b
    return e_new, K_m, K_h


def gradx(eta, mu, dx):
    return mu * (jnp.roll(eta, -1, axis=-1) - eta) / dx


def grady(eta, mv, dy):
    return mv * (jnp.roll(eta, -1, axis=-2) - eta) / dy


def div(u, v, m, dx, dy):
    return m * ((u - jnp.roll(u, 1, axis=-1)) / dx + (v - jnp.roll(v, 1, axis=-2)) / dy)


def _avg(pairs):
    num = sum(w * f for f, w in pairs)
    den = sum(w for _, w in pairs)
    return jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 0.0)


def avg_v_to_u(v, mv, mu):
    vs, ms = jnp.roll(v, 1, axis=-2), jnp.roll(mv, 1, axis=-2)
    return _avg([(v, mv), (vs, ms), (jnp.roll(v, -1, axis=-1), jnp.roll(mv, -1, axis=-1)),
                 (jnp.roll(vs, -1, axis=-1), jnp.roll(ms, -1, axis=-1))]) * mu


def avg_u_to_v(u, mu, mv):
    uw, mw = jnp.roll(u, 1, axis=-1), jnp.roll(mu, 1, axis=-1)
    return _avg([(u, mu), (uw, mw), (jnp.roll(u, -1, axis=-2), jnp.roll(mu, -1, axis=-2)),
                 (jnp.roll(uw, -1, axis=-2), jnp.roll(mw, -1, axis=-2))]) * mv


def laplacian(a, mu, mv, mc, dx, dy):
    return div(gradx(a, mu, dx), grady(a, mv, dy), mc, dx, dy)


def apply_diffusion(f, nu, dzx, sflux, drag, mask3):
    nz = f.shape[0]
    dzi = 0.5 * (dzx[:-1] + dzx[1:])
    below = jnp.concatenate([mask3[1:], jnp.zeros_like(mask3[:1])], axis=0)
    bottom = (mask3 > 0) & (below == 0)
    nu_in = nu[1:nz] if hasattr(nu, "shape") and nu.ndim == 3 else nu
    inner = nu_in * (f[:-1] - f[1:]) * _safe(dzi)
    inner = jnp.where(bottom[:nz - 1], 0.0, inner)
    flux = jnp.concatenate([sflux[None], inner, jnp.zeros_like(f[:1])], axis=0)
    out = (flux[:nz] - flux[1:]) * _safe(dzx)
    return (out - drag * f * bottom * _safe(dzx)) * (mask3 > 0)


def w_from_transport(u, v, g):
    tdiv = div(g["dz3u"] * u, g["dz3v"] * v, g["mask3"], g["dx"], g["dy"])
    w = -jnp.cumsum(tdiv[::-1], axis=0)[::-1]
    return jnp.concatenate([w, jnp.zeros_like(w[:1])], axis=0)


def pressure_gradients(b, g, remove: bool, corr: bool):
    dz3, zc = g["dz3"], g["zc"]
    above = jnp.concatenate([jnp.zeros_like(b[:1]), jnp.cumsum((dz3 * b)[:-1], axis=0)], axis=0)
    phi = 0.5 * dz3 * b + above
    if remove:
        tot = jnp.sum(dz3, axis=0, keepdims=True)
        phi = phi - jnp.sum(dz3 * phi, axis=0, keepdims=True) * _safe(tot)
    if not corr:
        return (gradx(phi, g["mask3u"], g["dx"]) * g["mask3u"],
                grady(phi, g["mask3v"], g["dy"]) * g["mask3v"])
    out = []
    for axis, fmask, delta in ((-1, g["mask3u"], g["dx"]), (-2, g["mask3v"], g["dy"])):
        phi_r, b_r, z_r = (jnp.roll(phi, -1, axis=axis), jnp.roll(b, -1, axis=axis),
                           jnp.roll(zc, -1, axis=axis))
        z_face = 0.5 * (zc + z_r)
        out.append(fmask * ((phi_r + b_r * (z_face - z_r)) - (phi + b * (z_face - zc))) / delta)
    return out[0], out[1]


def momentum_advection(u, v, w, dx, dy, dz):
    nz = u.shape[0]
    ubar_c = 0.5 * (jnp.roll(u, 1, axis=-1) + u)
    adv_ux = (jnp.roll(ubar_c, -1, axis=-1) ** 2 - ubar_c ** 2) / dx
    fy_u = 0.5 * (v + jnp.roll(v, -1, axis=-1)) * 0.5 * (u + jnp.roll(u, -1, axis=-2))
    adv_uy = (fy_u - jnp.roll(fy_u, 1, axis=-2)) / dy
    wbar_u = 0.5 * (w + jnp.roll(w, -1, axis=-1))
    fz_u = jnp.concatenate([jnp.zeros_like(u[:1]), wbar_u[1:nz] * 0.5 * (u[:-1] + u[1:]),
                            jnp.zeros_like(u[:1])], axis=0)
    adv_uz = (fz_u[:nz] - fz_u[1:]) / dz
    vbar_c = 0.5 * (jnp.roll(v, 1, axis=-2) + v)
    adv_vy = (jnp.roll(vbar_c, -1, axis=-2) ** 2 - vbar_c ** 2) / dy
    fx_v = 0.5 * (u + jnp.roll(u, -1, axis=-2)) * 0.5 * (v + jnp.roll(v, -1, axis=-1))
    adv_vx = (fx_v - jnp.roll(fx_v, 1, axis=-1)) / dx
    wbar_v = 0.5 * (w + jnp.roll(w, -1, axis=-2))
    fz_v = jnp.concatenate([jnp.zeros_like(v[:1]), wbar_v[1:nz] * 0.5 * (v[:-1] + v[1:]),
                            jnp.zeros_like(v[:1])], axis=0)
    adv_vz = (fz_v[:nz] - fz_v[1:]) / dz
    return adv_ux + adv_uy + adv_uz, adv_vx + adv_vy + adv_vz


def tracer_advection(c, u, v, w, g, upwind: bool):
    nz = c.shape[0]
    if upwind:
        cx = jnp.where(u > 0, c, jnp.roll(c, -1, axis=-1))
        cy = jnp.where(v > 0, c, jnp.roll(c, -1, axis=-2))
    else:
        cx = 0.5 * (c + jnp.roll(c, -1, axis=-1))
        cy = 0.5 * (c + jnp.roll(c, -1, axis=-2))
    fx = g["dz3u"] * u * cx
    fy = g["dz3v"] * v * cy
    inv = g["inv_dz3"]
    hdiv = ((fx - jnp.roll(fx, 1, axis=-1)) / g["dx"] + (fy - jnp.roll(fy, 1, axis=-2)) / g["dy"]) * inv
    cf = jnp.concatenate([c[:1], 0.5 * (c[:-1] + c[1:]), jnp.zeros_like(c[:1])], axis=0)
    vdiv = (w[:nz] * cf[:nz] - w[1:] * cf[1:]) * inv
    return (hdiv + vdiv) * g["mask3"]


def _rho_teos10(ct, sa, z):
    """polyTEOS10-bsq (libs/core/eos.py::_rho_teos10) with jnp.sqrt."""
    E = eos_mod
    zh = z * E.R1_Z0
    ss = jnp.sqrt((sa + E.RDELTA_S) * E.R1_S0)
    tt = ct * E.R1_T0
    r0 = (((((E.R05 * zh + E.R04) * zh + E.R03) * zh + E.R02) * zh + E.R01) * zh + E.R00) * zh
    rz3 = E.EOS013 * tt + E.EOS103 * ss + E.EOS003
    rz2 = ((E.EOS022 * tt + E.EOS112 * ss + E.EOS012) * tt
           + (E.EOS202 * ss + E.EOS102) * ss + E.EOS002)
    rz1 = ((((E.EOS041 * tt + E.EOS131 * ss + E.EOS031) * tt
             + (E.EOS221 * ss + E.EOS121) * ss + E.EOS021) * tt
            + ((E.EOS311 * ss + E.EOS211) * ss + E.EOS111) * ss + E.EOS011) * tt
           + (((E.EOS401 * ss + E.EOS301) * ss + E.EOS201) * ss + E.EOS101) * ss
           + E.EOS001)
    rz0 = ((((((E.EOS060 * tt + E.EOS150 * ss + E.EOS050) * tt
               + (E.EOS240 * ss + E.EOS140) * ss + E.EOS040) * tt
              + ((E.EOS330 * ss + E.EOS230) * ss + E.EOS130) * ss + E.EOS030) * tt
             + (((E.EOS420 * ss + E.EOS320) * ss + E.EOS220) * ss + E.EOS120) * ss
             + E.EOS020) * tt
            + ((((E.EOS510 * ss + E.EOS410) * ss + E.EOS310) * ss + E.EOS210) * ss
               + E.EOS110) * ss + E.EOS010) * tt
           + (((((E.EOS600 * ss + E.EOS500) * ss + E.EOS400) * ss + E.EOS300) * ss
               + E.EOS200) * ss + E.EOS100) * ss + E.EOS000)
    return ((rz3 * zh + rz2) * zh + rz1) * zh + rz0 + r0


def tracer_advection_up3(c, u, v, w, g, limited):
    nz = c.shape[0]
    cx = kappa_face(c, u, -1, g["mask3"], limited)
    cy = kappa_face(c, v, -2, g["mask3"], limited)
    fx = g["dz3u"] * u * cx
    fy = g["dz3v"] * v * cy
    inv = g["inv_dz3"]
    hdiv = ((fx - jnp.roll(fx, 1, axis=-1)) / g["dx"] + (fy - jnp.roll(fy, 1, axis=-2)) / g["dy"]) * inv
    cf = jnp.concatenate([c[:1], kappa_face_z(c, w, g["mask3"], limited), jnp.zeros_like(c[:1])], axis=0)
    vdiv = (w[:nz] * cf[:nz] - w[1:] * cf[1:]) * inv
    return (hdiv + vdiv) * g["mask3"]


def buoyancy_eos(kind: str, t, s, zc, g_, rho0, alpha_t, beta_s, t0, s0):
    if kind == "linear":
        rp = rho0 * (-alpha_t * (t - t0) + beta_s * (s - s0))
    elif kind == "seos":
        c = eos_mod.SEOS
        ta, sa = t - 10.0, s - 35.0
        rp = (-c["a0"] * (1.0 + 0.5 * c["lam1"] * ta + c["mu1"] * zc) * ta
              + c["b0"] * (1.0 - 0.5 * c["lam2"] * sa - c["mu2"] * zc) * sa
              - c["nu_ts"] * ta * sa)
    else:
        rp = _rho_teos10(t, s, zc) - rho0
    return -g_ * rp / rho0


# ------------------------------------------------------- variable Helmholtz
def helm_apply(lv, coef, x):
    cx, cy = coef / lv["dx"] ** 2, coef / lv["dy"] ** 2
    ku, kv, msk = lv["ku"], lv["kv"], lv["msk"]
    lap = (cx * (ku * (jnp.roll(x, -1, axis=-1) - x) - jnp.roll(ku, 1, axis=-1) * (x - jnp.roll(x, 1, axis=-1)))
           + cy * (kv * (jnp.roll(x, -1, axis=-2) - x) - jnp.roll(kv, 1, axis=-2) * (x - jnp.roll(x, 1, axis=-2))))
    return jnp.where(msk > 0.0, x - lap, x)


def rbgs(lv, coef, x, b, colour):
    cx, cy = coef / lv["dx"] ** 2, coef / lv["dy"] ** 2
    ku, kv = lv["ku"], lv["kv"]
    nb = (cx * (ku * jnp.roll(x, -1, axis=-1) + jnp.roll(ku, 1, axis=-1) * jnp.roll(x, 1, axis=-1))
          + cy * (kv * jnp.roll(x, -1, axis=-2) + jnp.roll(kv, 1, axis=-2) * jnp.roll(x, 1, axis=-2)))
    return jnp.where(lv["col"][colour], lv["dinv"] * (b + nb), x)


def _dot(a, b):
    return jnp.sum(a * b)


class HelmholtzJAX:
    """Levels are built in NumPy once (mod_helm_var.f90); solves are jitted."""

    def __init__(self, nx, ny, dx, dy, coef, ku, kv, msk, kind, rtol, max_iter):
        self.coef, self.kind, self.rtol, self.max_iter = float(coef), kind, float(rtol), int(max_iter)
        self.levels = []
        cur = dict(nx=nx, ny=ny, dx=dx, dy=dy, ku=ku, kv=kv, msk=msk)
        while True:
            self.levels.append(self._finish(cur))
            if kind != "multigrid" or cur["nx"] % 2 or cur["ny"] % 2 \
                    or cur["nx"] // 2 < 8 or cur["ny"] // 2 < 8:
                break
            ku, kv, msk = cur["ku"], cur["kv"], cur["msk"]
            cku = 0.5 * (ku[0::2, 1::2] + ku[1::2, 1::2])
            ckv = 0.5 * (kv[1::2, 0::2] + kv[1::2, 1::2])
            cmsk = np.maximum(np.maximum(msk[0::2, 0::2], msk[0::2, 1::2]),
                              np.maximum(msk[1::2, 0::2], msk[1::2, 1::2]))
            cur = dict(nx=cur["nx"] // 2, ny=cur["ny"] // 2, dx=2.0 * cur["dx"], dy=2.0 * cur["dy"],
                       ku=cku, kv=ckv, msk=cmsk)

    def make_levels(self, ku, kv):
        """Levels for new face coefficients inside jit (masks/colours static)."""
        out = []
        cur_ku, cur_kv = ku, kv
        last = len(self.levels) - 1
        for i, lv in enumerate(self.levels):
            cx, cy = self.coef / lv["dx"] ** 2, self.coef / lv["dy"] ** 2
            d = (1.0 + cx * (cur_ku + jnp.roll(cur_ku, 1, axis=-1)) + cy * (cur_kv + jnp.roll(cur_kv, 1, axis=-2)))
            d = jnp.where(lv["msk"] <= 0.0, 1.0, d)
            out.append(dict(dx=lv["dx"], dy=lv["dy"], ku=cur_ku, kv=cur_kv, msk=lv["msk"], dinv=1.0 / d, col=lv["col"]))
            if i == last:
                break          # the coarsest level has no child; an odd size would
                               # make the slices disagree in shape (docs/90 N20)
            cur_ku = 0.5 * (cur_ku[0::2, 1::2] + cur_ku[1::2, 1::2])
            cur_kv = 0.5 * (cur_kv[1::2, 0::2] + cur_kv[1::2, 1::2])
        return out

    def _finish(self, lv):
        cx, cy = self.coef / lv["dx"] ** 2, self.coef / lv["dy"] ** 2
        d = (1.0 + cx * (lv["ku"] + np.roll(lv["ku"], 1, axis=-1))
             + cy * (lv["kv"] + np.roll(lv["kv"], 1, axis=-2)))
        d = np.where(lv["msk"] <= 0.0, 1.0, d)
        jj, ii = np.meshgrid(np.arange(lv["ny"]), np.arange(lv["nx"]), indexing="ij")
        col = [((ii + jj) % 2 == c) & (lv["msk"] > 0.0) for c in (0, 1)]
        return dict(dx=lv["dx"], dy=lv["dy"], ku=jnp.asarray(lv["ku"]), kv=jnp.asarray(lv["kv"]),
                    msk=jnp.asarray(lv["msk"]), dinv=jnp.asarray(1.0 / d),
                    col=[jnp.asarray(c) for c in col])

    # -- PCG (Jacobi or 4-sweep RBGS preconditioner) ---------------------
    def _precondition(self, lv, r):
        if self.kind == "pcg_rbgs":
            z = jnp.zeros_like(r)
            for c in (0, 1, 1, 0):
                z = rbgs(lv, self.coef, z, r, c)
            return z
        return lv["dinv"] * r

    def solve(self, rhs, levels):
        """Returns (x, iterations, converged). Pure function of (rhs, levels)."""
        lv = levels[0]
        r = rhs * lv["msk"]
        norm_b = jnp.sqrt(_dot(r, r))
        tol = self.rtol * norm_b
        x = jnp.zeros_like(rhs)
        if self.kind == "multigrid":
            b = r

            def cond(st):
                return (st[2] < self.max_iter) & (st[3] > tol)

            def body(st):
                x, res, it, _ = st
                x = self._vcycle(0, x, b, levels)
                res = b - helm_apply(lv, self.coef, x)
                return (x, res, it + 1, jnp.sqrt(_dot(res, res)))

            x, _, it, resn = lax.while_loop(cond, body, (x, r, 0, norm_b))
            return x, it, resn <= tol

        z = self._precondition(lv, r)
        p = z
        rz = _dot(r, z)

        def cond(st):
            return (st[5] < self.max_iter) & (st[6] > tol)

        def body(st):
            x, r, p, z, rz, it, _ = st
            ap = helm_apply(lv, self.coef, p)
            alpha = rz / _dot(p, ap)
            x = x + alpha * p
            r = r - alpha * ap
            resn = jnp.sqrt(_dot(r, r))
            z = self._precondition(lv, r)
            rzn = _dot(r, z)
            p = z + (rzn / rz) * p
            return (x, r, p, z, rzn, it + 1, resn)

        x, _, _, _, _, it, resn = lax.while_loop(cond, body, (x, r, p, z, rz, 0, norm_b))
        return x, it, resn <= tol

    def _vcycle(self, l, x, b, levels):
        lv = levels[l]
        for _ in range(2):
            x = rbgs(lv, self.coef, x, b, 0)
            x = rbgs(lv, self.coef, x, b, 1)
        if l + 1 < len(levels):
            r = (b - helm_apply(lv, self.coef, x)) * lv["msk"]
            ny, nx = r.shape
            bc = 0.25 * (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2])
            xc = self._vcycle(l + 1, jnp.zeros_like(bc), bc, levels)
            x = x + jnp.repeat(jnp.repeat(xc, 2, axis=0), 2, axis=1) * lv["msk"]
        for _ in range(2):
            x = rbgs(lv, self.coef, x, b, 1)
            x = rbgs(lv, self.coef, x, b, 0)
        return x


# ------------------------------------------------------------------ stepper
@dataclass
class StepperJAX:
    cfg: dict[str, Any]
    h: np.ndarray
    mask: np.ndarray
    geo: dict = field(init=False)
    n_split: int = 0

    def __post_init__(self) -> None:
        c = self.cfg
        nx, ny, nz = int(c["nx"]), int(c["ny"]), int(c["nz"])
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dt = float(c["dt"])
        self.dx, self.dy = float(c["lx"]) / nx, float(c["ly"]) / ny
        self.scheme = c["scheme_name"]
        self.is_theta = self.scheme == "theta"
        self.is_split = self.scheme == "split_explicit"
        self.is_ts = c["tracers"] == "TS"
        self.do_adv = c["advection"] != "none"
        self.is_up3 = c["advection"] in ("up3", "up3_tvd")
        self.is_tvd = c["advection"] == "up3_tvd"
        self.is_tke = c.get("closure", "none") == "tke"
        self.tke_p = {k: float(c.get(k, dflt)) for k, dflt in (("c_k", 0.1), ("c_eps", 0.7), ("pr_t", 1.0),
                                                                 ("z_0", 0.1), ("e_min", 1e-6), ("l_min", 0.01),
                                                                 ("e_bb", 3.75), ("n2_min", 1e-8))}
        self.mxl = str(c.get("mxl", "integral"))
        if self.mxl not in ("integral", "recursive"):
            raise ValueError(f"mxl must be integral|recursive, got {self.mxl!r}")
        if c["vcoord"] != "zlevel":
            raise ValueError("JAX v0.5 backend: vcoord=zlevel only")
        if c["bc_x"] == "open" or c["bc_y"] == "open":
            raise ValueError("JAX v0.5 backend: periodic|closed walls only")
        g = build_geometry(self.h, self.mask, nz, c["face_rule"], float(c["min_partial"]),
                           c["bc_x"], c["bc_y"])
        thv = float(c["theta_v"])
        tri_m = diffusion_coeffs(float(c["nu"]), g["dz3"], g["mask3"], self.dt, thv,
                                 float(c["bottom_drag"]))
        tri_b = diffusion_coeffs(float(c["kappa"]), g["dz3"], g["mask3"], self.dt, thv, 0.0)
        q = thomas_np(*tri_m, np.ones((nz, ny, nx)))
        qu = np.minimum(q, np.roll(q, -1, axis=-1))
        qv = np.minimum(q, np.roll(q, -1, axis=-2))
        ku = np.sum(g["dz3u"] * qu, axis=0) * g["mask_u"]
        kv = np.sum(g["dz3v"] * qv, axis=0) * g["mask_v"]
        tau_u = np.full((ny, nx), float(c["tau_x"]) / float(c["rho0"])) * g["mask3u"][0]
        tau_v = np.full((ny, nx), float(c["tau_y"]) / float(c["rho0"])) * g["mask3v"][0]
        flux_t = np.full((ny, nx), float(c["q_heat"]) / (float(c["rho0"]) * float(c["cp"])))
        flux_s = np.full((ny, nx), float(c["q_salt"]))
        arrays = dict(g, tri_m=tri_m, tri_b=tri_b, q=q, ku=ku, kv=kv, tau_u=tau_u, tau_v=tau_v,
                      flux_t=flux_t, flux_s=flux_s, zero2=np.zeros((ny, nx)))
        arrays.pop("hmax")
        self.geo = {k: (tuple(jnp.asarray(a) for a in v) if isinstance(v, tuple) else jnp.asarray(v))
                    for k, v in arrays.items()}
        self.geo["dx"], self.geo["dy"] = self.dx, self.dy
        self.geo["nu3"] = jnp.full((nz + 1, ny, nx), float(c["nu"]))
        self.geo["kap3"] = jnp.full((nz + 1, ny, nx), float(c["kappa"]))
        self.solver = None
        self.levels = []
        if self.is_theta:
            coef = float(c["g"]) * float(c["theta"]) ** 2 * self.dt ** 2
            self.solver = HelmholtzJAX(nx, ny, self.dx, self.dy, coef, ku, kv, g["mask"],
                                       c["solver_kind"], float(c["rtol"]), int(c["max_iter"]))
            self.levels = self.solver.levels
        if self.is_split:
            cc = np.sqrt(float(c["g"]) * g["hmax"])
            dt_baro = 1.0 / (cc * np.sqrt(1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
            ns = int(c["n_split"])
            self.n_split = ns if ns > 0 else max(1, int(np.ceil(1.2 * self.dt / dt_baro)))
        self._step = jax.jit(self._step_impl)

    # ------------------------------------------------------------ one step
    def step(self, state: tuple) -> tuple:
        return self._step(state, self.geo, self.levels)

    def _step_impl(self, state, g, levels):
        u, v, b, eta, t, s, iters, fails, e, nu3, kap3 = state
        c = self.cfg
        dt, f, g_ = self.dt, float(c["f0"]), float(c["g"])
        th = float(c["theta"]) if self.is_theta else 0.0      # fb: full explicit gradient (N14)
        tc, thv = float(c["theta_cor"]), float(c["theta_v"])
        npic = int(c["n_picard"])
        m3, m3u, m3v = g["mask3"], g["mask3u"], g["mask3v"]
        dx, dy = self.dx, self.dy
        if self.is_tke:
            # spec S11.2: closure first, then every coefficient that depends on K
            ustar2 = jnp.full(eta.shape, np.hypot(float(c["tau_x"]), float(c["tau_y"])) / float(c["rho0"]))
            e, nu3, kap3 = tke_step(e, u, v, b, g["dz3"], m3, ustar2, dt, self.tke_p,
                                    float(c["nu"]), float(c["kappa"]), nu3, kap3, self.mxl)
            g = dict(g)
            g["nu3"], g["kap3"] = nu3, kap3
            g["tri_m"] = diffusion_coeffs_jnp(nu3, g["dz3"], m3, dt, thv, float(c["bottom_drag"]))
            g["tri_b"] = diffusion_coeffs_jnp(kap3, g["dz3"], m3, dt, thv, 0.0)
            q = thomas(*g["tri_m"], jnp.ones_like(u))
            g["q"] = q
            g["ku"] = jnp.sum(g["dz3u"] * jnp.minimum(q, jnp.roll(q, -1, axis=-1)), axis=0) * g["mask_u"]
            g["kv"] = jnp.sum(g["dz3v"] * jnp.minimum(q, jnp.roll(q, -1, axis=-2)), axis=0) * g["mask_v"]
            if self.is_theta:
                levels = self.solver.make_levels(g["ku"], g["kv"])
        nu, kappa = nu3, kap3

        pgx, pgy = pressure_gradients(b, g, c["pgf"] == "remove", bool(c["pgf_correction"]))
        av_prev_v = avg_v_to_u(v, m3v, m3u)
        av_prev_u = avg_u_to_v(u, m3u, m3v)
        dzu_diff = apply_diffusion(u, nu, g["dz3u"], g["tau_u"], float(c["bottom_drag"]), m3)
        dzv_diff = apply_diffusion(v, nu, g["dz3v"], g["tau_v"], float(c["bottom_drag"]), m3)
        w_old = w_from_transport(u, v, g)
        ex_u = jnp.zeros_like(u)
        ex_v = jnp.zeros_like(v)
        if self.is_up3:
            adv_u, adv_v = momentum_advection_up3(u, v, w_old, dx, dy, float(c["h0"]) / self.nz, m3u, m3v, self.is_tvd)
            ex_u = ex_u - adv_u * m3u
            ex_v = ex_v - adv_v * m3v
        elif self.do_adv:
            adv_u, adv_v = momentum_advection(u, v, w_old, dx, dy, float(c["h0"]) / self.nz)
            ex_u = ex_u - adv_u * m3u
            ex_v = ex_v - adv_v * m3v
        a_h = float(c["a_h"])
        if a_h > 0.0:
            ex_u = ex_u + a_h * laplacian(u, m3u, m3v, m3u, dx, dy)
            ex_v = ex_v + a_h * laplacian(v, m3u, m3v, m3v, dx, dy)
        top_u = thv * dt * g["tau_u"] * _safe(g["dz3u"][0])
        top_v = thv * dt * g["tau_v"] * _safe(g["dz3v"][0])
        stress_u = dt * ex_u + jnp.concatenate([top_u[None], jnp.zeros_like(u[1:])], axis=0)
        stress_v = dt * ex_v + jnp.concatenate([top_v[None], jnp.zeros_like(v[1:])], axis=0)

        state = (u, v, b, eta, t, s, iters, fails, e, nu3, kap3)
        if self.is_split:
            return self._step_split(state, g, pgx, pgy, av_prev_u, av_prev_v, dzu_diff, dzv_diff,
                                    stress_u, stress_v)

        gx_eta = gradx(eta, g["mask_u"], dx)[None] * m3u
        gy_eta = grady(eta, g["mask_v"], dy)[None] * m3v
        div_old = div(jnp.sum(g["dz3u"] * u, axis=0), jnp.sum(g["dz3v"] * v, axis=0), g["mask"], dx, dy)
        u_it, v_it, eta_new = u, v, eta
        for _ in range(npic):
            v_cor = (1.0 - tc) * av_prev_v + tc * avg_v_to_u(v_it, m3v, m3u)
            u_cor = (1.0 - tc) * av_prev_u + tc * avg_u_to_v(u_it, m3u, m3v)
            gu = u + dt * (f * v_cor + pgx) - g_ * dt * (1.0 - th) * gx_eta + (1.0 - thv) * dt * dzu_diff
            gv = v + dt * (-f * u_cor + pgy) - g_ * dt * (1.0 - th) * gy_eta + (1.0 - thv) * dt * dzv_diff
            gh_u = thomas(*g["tri_m"], gu + stress_u) * m3
            gh_v = thomas(*g["tri_m"], gv + stress_v) * m3
            if self.is_theta:
                div_g = div(jnp.sum(g["dz3u"] * gh_u, axis=0), jnp.sum(g["dz3v"] * gh_v, axis=0),
                            g["mask"], dx, dy)
                rhs = eta - dt * ((1.0 - th) * div_old + th * div_g)
                eta_new, it, ok = self.solver.solve(rhs, levels)
                iters = iters + it
                fails = fails + jnp.where(ok, 0, 1)
                u_it = (gh_u - g_ * dt * th * g["q"] * gradx(eta_new, g["mask_u"], dx)[None] * m3u) * m3
                v_it = (gh_v - g_ * dt * th * g["q"] * grady(eta_new, g["mask_v"], dy)[None] * m3v) * m3
            else:
                u_it, v_it = gh_u, gh_v
        if not self.is_theta:
            eta_new = eta - dt * div(jnp.sum(g["dz3u"] * u_it, axis=0),
                                     jnp.sum(g["dz3v"] * v_it, axis=0), g["mask"], dx, dy)
        w = w_from_transport(u_it, v_it, g)
        b, t, s = self._advance_tracers(b, t, s, u_it, v_it, w, g)
        return (u_it, v_it, b, eta_new, t, s, iters, fails, e, nu3, kap3)

    def _step_split(self, state, g, pgx, pgy, av_prev_u, av_prev_v, dzu_diff, dzv_diff,
                    stress_u, stress_v):
        u, v, b, eta, t, s, iters, fails, e, nu3, kap3 = state
        c = self.cfg
        dt, f, g_ = self.dt, float(c["f0"]), float(c["g"])
        tc, thv, npic = float(c["theta_cor"]), float(c["theta_v"]), int(c["n_picard"])
        m3, m3u, m3v = g["mask3"], g["mask3u"], g["mask3v"]
        dx, dy = self.dx, self.dy
        U0 = jnp.sum(g["dz3u"] * u, axis=0)
        V0 = jnp.sum(g["dz3v"] * v, axis=0)
        u_it, v_it = u, v
        for _ in range(npic):
            v_cor = (1.0 - tc) * av_prev_v + tc * avg_v_to_u(v_it, m3v, m3u)
            u_cor = (1.0 - tc) * av_prev_u + tc * avg_u_to_v(u_it, m3u, m3v)
            gu = u + dt * (f * v_cor + pgx) + (1.0 - thv) * dt * dzu_diff
            gv = v + dt * (-f * u_cor + pgy) + (1.0 - thv) * dt * dzv_diff
            gh_u = thomas(*g["tri_m"], gu + stress_u) * m3
            gh_v = thomas(*g["tri_m"], gv + stress_v) * m3
            u_it, v_it = gh_u, gh_v
        umean = jnp.sum(g["dz3u"] * gh_u, axis=0)
        vmean = jnp.sum(g["dz3v"] * gh_v, axis=0)
        mu2, mv2 = g["mask_u"], g["mask_v"]
        # Coriolis removed with the 2D transport operator (spec S8.3, N15)
        cor_x = f * ((1.0 - tc) * avg_v_to_u(V0, mv2, mu2) + tc * avg_v_to_u(vmean, mv2, mu2))
        cor_y = -f * ((1.0 - tc) * avg_u_to_v(U0, mu2, mv2) + tc * avg_u_to_v(umean, mu2, mv2))
        fx = (umean - U0) / dt - cor_x
        fy = (vmean - V0) / dt - cor_y

        ddt = dt / self.n_split

        def substep(k, st):
            U, V, e = st
            av_v, av_u = avg_v_to_u(V, mv2, mu2), avg_u_to_v(U, mu2, mv2)
            gxe, gye = gradx(e, mu2, dx), grady(e, mv2, dy)
            Ui, Vi = U, V
            for _ in range(npic):
                v_cor = (1.0 - tc) * av_v + tc * avg_v_to_u(Vi, mv2, mu2)
                u_cor = (1.0 - tc) * av_u + tc * avg_u_to_v(Ui, mu2, mv2)
                Ui = (U + ddt * (f * v_cor - g_ * g["ku"] * gxe + fx)) * mu2
                Vi = (V + ddt * (-f * u_cor - g_ * g["kv"] * gye + fy)) * mv2
            e = e - ddt * div(Ui, Vi, g["mask"], dx, dy)
            return (Ui, Vi, e)

        U, V, eta_new = lax.fori_loop(0, self.n_split, substep, (U0, V0, eta))
        u_new = (gh_u - umean * g["inv_hcu"] + U * g["inv_hcu"]) * m3u     # face masks (N15)
        v_new = (gh_v - vmean * g["inv_hcv"] + V * g["inv_hcv"]) * m3v
        w = w_from_transport(u_new, v_new, g)
        b, t, s = self._advance_tracers(b, t, s, u_new, v_new, w, g)
        return (u_new, v_new, b, eta_new, t, s, iters, fails, e, nu3, kap3)

    def _advance_tracer(self, cc, u, v, w, sflux, source, g):
        c = self.cfg
        dt, thv = self.dt, float(c["theta_v"])
        tend = source
        if self.is_up3:
            tend = tend - tracer_advection_up3(cc, u, v, w, g, self.is_tvd)
        elif self.do_adv:
            tend = tend - tracer_advection(cc, u, v, w, g, c["advection"] == "upwind1")
        k_h = float(c["k_h"])
        if k_h > 0.0:
            tend = tend + k_h * laplacian(cc, g["mask3u"], g["mask3v"], g["mask3"], self.dx, self.dy)
        rhs = cc + dt * tend + (1.0 - thv) * dt * apply_diffusion(cc, g["kap3"], g["dz3"],
                                                                  sflux, 0.0, g["mask3"])
        top = rhs[0] + thv * dt * sflux * _safe(g["dz3"][0])
        rhs = jnp.concatenate([top[None], rhs[1:]], axis=0)
        return thomas(*g["tri_b"], rhs) * g["mask3"]

    def _advance_tracers(self, b, t, s, u, v, w, g):
        c = self.cfg
        zero3 = jnp.zeros_like(b)
        if self.is_ts:
            t = self._advance_tracer(t, u, v, w, g["flux_t"], zero3, g)
            s = self._advance_tracer(s, u, v, w, g["flux_s"], zero3, g)
            b = buoyancy_eos(c["eos"], t, s, g["zc"], float(c["g"]), float(c["rho0"]),
                             float(c["alpha_t"]), float(c["beta_s"]), float(c["t0"]),
                             float(c["s0"])) * g["mask3"]
            return b, t, s
        source = -float(c["n2"]) * 0.5 * (w[:-1] + w[1:])
        b = self._advance_tracer(b, u, v, w, g["zero2"], source, g)
        return b, t, s
