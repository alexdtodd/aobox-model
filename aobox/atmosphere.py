# atmosphere.py

import numpy as np
import constants
import coupler


def specific_humidity_to_mass(humidity):
    """Convert atmosphere specific humidity to water mass."""
    mass = constants.MASS_ATMOS*(humidity/(1 - humidity))
    return mass


def mass_to_specific_humidity(mass):
    """Convert atmosphere specific humidity to water mass."""
    q = mass/(constants.MASS_ATMOS + mass)
    return q


def saturation_specific_humidity(temp):
    """Saturation specific humidity.

    Parameters
    ----------
    temp : ndarray
        Atmosphere/ocean box temperatures in degC.

    Returns
    -------
    qsat : ndarray
        Atmosphere saturation specific humidity in kg kg^-1.

    """
    temp_kelvin = temp + constants.DEGC_TO_KELVIN

    # Linearised Clausius-Clapeyron equation
    frac = (2.0*(0.622*constants.SATURATION_VAPOUR_PRESSURE /
                 constants.SURFACE_PRESSURE))
    qsat = (frac*np.exp(-1.0*constants.WATER_EVAPORATION_LATENT_HEAT /
                        (constants.WATER_GAS_CONSTANT*(temp_kelvin))))
    return qsat


def precipitation_flux(humidity, temp_atmos):
    """Calculate precipitation flux.

    Following Sayag et al. (2004), cap atmosphere relative humidity at
    constants.MAX_RELATIVE_HUMIDITY and precipitate any remaining freshwater
    in one constants.TIME_STEP out of the atmosphere.

    Parameters
    ----------
    humidity : ndarray
        Atmosphere box [T, N, S] specific humidity in kg kg^-1.

    temp_atmos : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    Returns
    -------
    precip : ndarray
        Precipitation in kg m^-2 s^-1.

    """
    # Maximum specific humidity
    max_humidity = (constants.MAX_RELATIVE_HUMIDITY *
                    saturation_specific_humidity(temp_atmos))

    # Precipitation
    delta_humidity = humidity - max_humidity
    delta_humidity[delta_humidity < 0.0] = 0.0

    delta_mass = specific_humidity_to_mass(delta_humidity)
    precip = delta_mass/(constants.TIME_STEP*constants.AREA_ATMOS)

    return precip


def delta_temp_atmos_delta_t(temp_atmos, temp_ocean, humidity, parms,
                             toa_flux=None, temp_atmos_star=None, Qtoap=0.0):
    """Atmosphere box temperature tendency.

    Atmosphere box temperature tendency in degC s^-1 from applying either
    TOA restorting or TOA fluxes.

    Parameters
    ----------
    temp_atmos : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    toa_flux : ndarray, optional
        Prescribed top of atmopshere (TOA) fluxes [T, N, S] in W m^-2.

    temp_atmos_star : ndarray, optional
        Atmosphere box [T, N, S] restoring temperatures in degC.

    Qtoap : ndarray, optional
        Perturbation to be applied to TOA fluxes in W m^-2.

    Returns
    -------
    tend : ndarray
        Atmosphere box [T, N, S] temperature tendency in degC s^-1.

    flux_comps : ndarray
        Tendency components, [T, N, S] * [surf, mix, TOA], in W m^-2.

    toa_flux : ndarray
        TOA column of flux_comps, in W m^-2.

    """

    # Unpack parms
    v_m, lambda_LW = parms[-2:]

    # convert W to K s-1
    heat_to_temp = (1.0
                    / (constants.ATMOS_SPECIFIC_HEAT_CAPACITY
                       * constants.RHO_ATMOS * constants.VOLUME_ATMOS))

    # Heat from TOA flux
    if temp_atmos_star is None:
        toa = constants.AREA_ATMOS*(toa_flux + Qtoap)

    else:
        toa = constants.TOA_RELAX*(temp_atmos_star - temp_atmos)/heat_to_temp
        toa_flux = toa/constants.AREA_ATMOS

    # Heat from LW feedback
    net_toa = (toa - lambda_LW * (temp_atmos + constants.DEGC_TO_KELVIN)
               * constants.AREA_ATMOS)

    # Heat from meridional mixing
    nmix = (constants.N_ATMOS_BOUNDARY*v_m*(temp_atmos[0] - temp_atmos[1])
            * constants.ATMOS_SPECIFIC_HEAT_CAPACITY * constants.RHO_ATMOS)

    smix = (constants.S_ATMOS_BOUNDARY*v_m*(temp_atmos[0] - temp_atmos[2])
            * constants.ATMOS_SPECIFIC_HEAT_CAPACITY * constants.RHO_ATMOS)

    mix = np.array([-1*(nmix + smix), nmix, smix])

    # Heat from surface sensible flux
    oce_shf = coupler.surface_sensible_heat_flux(temp_atmos, temp_ocean, parms)

    # Heat from precipitation
    pre_lhf = (precipitation_flux(humidity, temp_atmos) * constants.AREA_ATMOS
               * constants.WATER_EVAPORATION_LATENT_HEAT)

    # Total heat tendency
    comps = np.array([-1.0*oce_shf, mix, pre_lhf, net_toa]).T
    total = np.sum(comps, axis=1)

    # Convert heat tend to temp tend
    tend = total*heat_to_temp
    flux_comps = comps/constants.AREA_ATMOS[:, None]
    return tend, flux_comps, toa_flux


def delta_humidity_delta_t(temp_atmos, temp_ocean, humidity, parms):
    """Atmosphere box humidity tendency.

    Parameters
    ----------
    temp_atmos : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    Returns
    -------
    total_tend : ndarray
        Atmosphere box [T, N, S] humidity tendency in kg kg^-1 s^-1.

    """
    # Unpack parms
    v_m = parms[-2]

    # Components
    oce_lhf = -1*coupler.surface_latent_heat_flux(humidity, temp_ocean, parms)
    oce_lhf_mass_tend = oce_lhf/constants.WATER_EVAPORATION_LATENT_HEAT

    pre_mass_tend = (precipitation_flux(humidity, temp_atmos)
                     * constants.AREA_ATMOS)

    nmix = (constants.N_ATMOS_BOUNDARY * v_m * humidity[0]
            * constants.RHO_ATMOS)

    smix = (constants.S_ATMOS_BOUNDARY * v_m * humidity[0]
            * constants.RHO_ATMOS)

    mix_mass_tend = np.array([-1*(nmix + smix), nmix, smix])

    total_tend = ((oce_lhf_mass_tend - pre_mass_tend + mix_mass_tend)
                  / constants.MASS_ATMOS)

    return total_tend
