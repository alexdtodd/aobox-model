# ocean.py

import numpy as np
import constants
import coupler


def volume_thermocline(D):
    """Volume of T (thermocline/tropical) ocean box."""
    return constants.AREA_T_OCEAN * D


def ocean_density(temp_ocean, salinity=constants.S0):
    """Ocean density from linear equation of state.

    Parameters
    ----------
    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    salinity : ndarray, optional
        Ocean box [T, N, D, S] salinity in PSU.

    Returns
    -------
    density : ndarray
        Ocean box [T, N, D, S] density in kg m^-3.

    """
    density = constants.RHO0 * (1 - constants.ALPHA * (temp_ocean
                                                       - constants.T0)
                                + constants.BETA * (salinity - constants.S0))
    return density


def ocean_circulation(temp_ocean, salinity, D, parms):
    """Ocean circulation components.

    Parameters
    ----------
    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    salinity : ndarray, optional
        Ocean box [T, N, D, S] salinity in PSU.

    D : float
        T (thermocline/tropical) ocean box depth in m.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    Returns
    -------
    psi : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.

    """
    # Unpack parms
    tau, kappav, kappagm = parms[:3]

    # tropics -> north and north -> deep
    rhot, rhon = ocean_density(temp_ocean, salinity)[:2]
    gp = constants.GRAVITY*(rhon - rhot)/constants.RHO0
    qn = gp*(D**2)/(2.0*constants.CORIOLIS_N)

    # deep -> tropics
    qu = kappav*constants.AREA_T_OCEAN/D

    # deep -> south and south -> tropics
    q_Ek = tau*Lx/(rho0*abs(fs))
    q_eddy = kappagm*D*Lx/Ly
    qs = q_Ek - q_eddy

    # total circulation
    psi = np.array([qu, qn, qs])
    return psi
