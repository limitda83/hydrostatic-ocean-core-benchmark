#########################################################################
#  Module: eos                                                          #
#  Description: Equations of state at three levels of nonlinearity      #
#               (docs/03_discretization_spec.md S10.6). The EOS is the  #
#               only arithmetically dense kernel in a hydrostatic ocean #
#               core, so it is also the axis on which the roofline      #
#               balance moves in the GPU's favour (RQ7).                #
#  Pipeline: model3d_v05 -> eos                                         #
#########################################################################

from __future__ import annotations

import numpy as np

# Simplified EOS of Roquet et al. (2015), Ocean Modelling 90, 29-43,
# their equation (25) - "S-EOS", NEMO's ln_seos. Retains cabbeling
# (lambda1) and thermobaricity (mu1) with six coefficients.
SEOS = dict(a0=1.6550e-1, b0=7.6554e-1, lam1=5.9520e-2, lam2=7.4914e-4,
            mu1=1.4970e-4, mu2=1.1090e-5, nu_ts=2.4341e-3)

# polyTEOS10-bsq, Roquet et al. (2015) Table 4: 55-term polynomial in the
# reduced variables of their eq. (33). Coefficients are theirs verbatim.
R00, R01, R02, R03, R04, R05 = (4.6494977072e+01, -5.2099962525e+00,
                                2.2601900708e-01, 6.4326772569e-02,
                                1.5616995503e-02, -1.7243708991e-03)
EOS000 = 8.0189615746e+02
EOS100, EOS200, EOS300, EOS400, EOS500, EOS600 = (
    8.6672408165e+02, -1.7864682637e+03, 2.0375295546e+03,
    -1.2849161071e+03, 4.3227585684e+02, -6.0579916612e+01)
EOS010, EOS110, EOS210, EOS310, EOS410, EOS510 = (
    2.6010145068e+01, -6.5281885265e+01, 8.1770425108e+01,
    -5.6888046321e+01, 1.7681814114e+01, -1.9193502195e+00)
EOS020, EOS120, EOS220, EOS320, EOS420 = (
    -3.7074170417e+01, 6.1548258127e+01, -6.0362551501e+01,
    2.9130021253e+01, -5.4723692739e+00)
EOS030, EOS130, EOS230, EOS330 = (2.1661789529e+01, -3.3449108469e+01,
                                  1.9717078466e+01, -3.1742946532e+00)
EOS040, EOS140, EOS240 = (-8.3627885467e+00, 1.1311538584e+01,
                          -5.3563304045e+00)
EOS050, EOS150 = (5.4048723791e-01, 4.5111434961e-01)
EOS060 = -1.9098268277e-01
EOS001, EOS101, EOS201, EOS301, EOS401 = (
    1.9681925209e+01, -4.2549998214e+01, 5.0774768218e+01,
    -3.0938076334e+01, 6.6051753097e+00)
EOS011, EOS111, EOS211, EOS311 = (-1.3336301113e+01, -4.4870114575e+00,
                                  5.0042598061e+00, -6.5399043664e-01)
EOS021, EOS121, EOS221 = (6.7080479603e+00, 3.5063081279e+00,
                          -1.8795372996e+00)
EOS031, EOS131 = (-2.4649669534e+00, -5.5077101279e-01)
EOS041 = 5.5927935970e-01
EOS002, EOS102, EOS202 = (2.0660924175e+00, -4.9527603989e+00,
                          2.5019633803e+00)
EOS012, EOS112 = (2.0564311499e+00, -2.1311365518e-01)
EOS022 = -1.2419983026e+00
EOS003, EOS103 = (-2.3342758797e-02, -1.8507636718e-02)
EOS013 = 3.7969820455e-01

# Reduced variables of Roquet et al. (2015) eq. (33), in NEMO's spelling.
# The check value below fails by ~6 kg m-3 if rdeltaS is taken as 24: these
# two constants are the whole difference between right and plausible.
RDELTA_S = 32.0
R1_S0 = 0.875 / 35.16504
R1_T0 = 1.0 / 40.0
R1_Z0 = 1.0e-4
RHO0_BSQ = 1026.0     # Boussinesq reference density of polyTEOS10-bsq [kg m-3]


def _rho_teos10(CT: np.ndarray, SA: np.ndarray, z: np.ndarray) -> np.ndarray:
    """In-situ density from polyTEOS10-bsq (Roquet et al. 2015, eq. 33).

    CT: Conservative Temperature [degC], SA: Absolute Salinity [g/kg],
    z: depth, positive downward [m]. Verified against the published check
    value rho(SA=30, CT=10, z=1000) = 1027.45140 kg m-3 (V5-4).
    """
    zh = z * R1_Z0
    ss = np.sqrt((SA + RDELTA_S) * R1_S0)
    tt = CT * R1_T0

    # Vertical reference profile r0(z), their eq. (34).
    r0 = (((((R05 * zh + R04) * zh + R03) * zh + R02) * zh + R01) * zh
          + R00) * zh

    rz3 = EOS013 * tt + EOS103 * ss + EOS003
    rz2 = ((EOS022 * tt + EOS112 * ss + EOS012) * tt
           + (EOS202 * ss + EOS102) * ss + EOS002)
    rz1 = ((((EOS041 * tt + EOS131 * ss + EOS031) * tt
             + (EOS221 * ss + EOS121) * ss + EOS021) * tt
            + ((EOS311 * ss + EOS211) * ss + EOS111) * ss + EOS011) * tt
           + (((EOS401 * ss + EOS301) * ss + EOS201) * ss + EOS101) * ss
           + EOS001)
    rz0 = ((((((EOS060 * tt + EOS150 * ss + EOS050) * tt
               + (EOS240 * ss + EOS140) * ss + EOS040) * tt
              + ((EOS330 * ss + EOS230) * ss + EOS130) * ss + EOS030) * tt
             + (((EOS420 * ss + EOS320) * ss + EOS220) * ss + EOS120) * ss
             + EOS020) * tt
            + ((((EOS510 * ss + EOS410) * ss + EOS310) * ss + EOS210) * ss
               + EOS110) * ss + EOS010) * tt
           + (((((EOS600 * ss + EOS500) * ss + EOS400) * ss + EOS300) * ss
               + EOS200) * ss + EOS100) * ss + EOS000)
    return ((rz3 * zh + rz2) * zh + rz1) * zh + rz0 + r0


def _rho_prime_seos(T: np.ndarray, S: np.ndarray, z: np.ndarray,
                    rho0: float) -> np.ndarray:
    """Density anomaly from the simplified EOS, spec S10.6."""
    c = SEOS
    Ta, Sa = T - 10.0, S - 35.0
    # The coefficients are already in kg m-3, so no rho0 factor appears.
    return (-c["a0"] * (1.0 + 0.5 * c["lam1"] * Ta + c["mu1"] * z) * Ta
            + c["b0"] * (1.0 - 0.5 * c["lam2"] * Sa - c["mu2"] * z) * Sa
            - c["nu_ts"] * Ta * Sa)


def rho_prime(kind: str, T: np.ndarray, S: np.ndarray, z: np.ndarray,
              *, rho0: float = 1025.0, alpha_T: float = 2.0e-4,
              beta_S: float = 7.4e-4, T0: float = 10.0, S0: float = 35.0
              ) -> np.ndarray:
    """Density anomaly rho' [kg m-3] for the requested EOS level."""
    if kind == "linear":
        return rho0 * (-alpha_T * (T - T0) + beta_S * (S - S0))
    if kind == "seos":
        return _rho_prime_seos(T, S, z, rho0)
    if kind == "teos10":
        # rho0 here is the caller's Boussinesq reference; the polynomial's own
        # reference is RHO0_BSQ, so the anomaly is taken about the caller's.
        return _rho_teos10(T, S, z) - rho0
    raise ValueError(f"unknown eos '{kind}' (expected linear|seos|teos10)")


def buoyancy(kind: str, T: np.ndarray, S: np.ndarray, z: np.ndarray,
             *, g: float = 9.80616, rho0: float = 1025.0, **kw) -> np.ndarray:
    """b = -g rho' / rho0, the sign convention of spec S7.1."""
    return -g * rho_prime(kind, T, S, z, rho0=rho0, **kw) / rho0


# Nominal floating-point operations per cell, counted from the expressions
# above. Used to place each EOS level on the roofline (docs/13, RQ7).
FLOPS_PER_CELL = {"linear": 4, "seos": 15, "teos10": 71}
