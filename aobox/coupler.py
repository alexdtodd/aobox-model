# coupler.py

import numpy as np
import constants
import atmosphere

# General constants
atm_inds = constants.ATMOS_INDS


def surface_latent_heat_flux(humidity, temp_ocean, parms):
    """Surface latent heat flux.

    Return downwards surface latent heat flux in W from atmopshere to ocean
    boxes.

    Parameters
    ----------
    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    Returns
    -------
    lhf : ndarray
        Downwards latent heat flux in W.

    """
    # Unpack parms
    tau_q = parms[-1]
    lambda_q = 1/(constants.SEC_PER_DAY*tau_q)

    ocean_qsat = atmosphere.saturation_specific_humidity(temp_ocean[atm_inds])
    delta_humidity = humidity - ocean_qsat
    lhf = (constants.RHO_ATMOS * constants.VOLUME_ATMOS * lambda_q
           * constants.WATER_EVAPORATION_LATENT_HEAT * delta_humidity)

    return lhf


def surface_sensible_heat_flux(temp_atmos, temp_ocean, parms):
    """Surface sensible heat flux.

    Return downwards surface sensible heat flux in W from atmopshere to ocean
    boxes.

    Parameters
    ----------
    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    Returns
    -------
    shf : ndarray
        Downwards sensible heat flux in W.

    """
    # Unpack parms
    tau_theta = parms[4]
    lambda_theta = 1/(constants.SEC_PER_DAY*tau_theta)

    delta_theta = temp_atmos - temp_ocean[atm_inds]
    shf = (constants.RHO_ATMOS * constants.VOLUME_ATMOS * lambda_theta
           * constants.ATMOS_SPECIFIC_HEAT_CAPACITY * delta_theta)

    return shf
