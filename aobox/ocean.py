# ocean.py

import numpy as np
import constants
import coupler
import atmosphere


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
    q_Ek = tau*constants.S_OCEAN_X/(constants.RHO0*abs(constants.CORIOLIS_S))
    q_eddy = kappagm*D*constants.S_OCEAN_X/constants.S_OCEAN_Y
    qs = q_Ek - q_eddy

    # total circulation
    psi = np.array([qu, qn, qs])
    return psi


def delta_D_delta_t(D, q):
    """Tropical ocean box depth tendency.

    Parameters
    ----------
    D : float
        T (thermocline/tropical) ocean box depth in m.

    q : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.

    Returns
    -------
    tend : ndarray
        Tropical ocean box depth tendency in m s^-1.
    """
    tend = (1.0/constants.AREA_T_OCEAN)*(q[2] + q[0] - q[1])
    return tend


def delta_temp_volume_ocean_dt(psi, temp_ocean, D, temp_atmos, humidity, parms,
                               TanmTon_ti=None, psiamoc_ti=None,
                               TotmTon_ti=None):
    """Ocean box volume and temperature tendency.

    Ocean box volume and temperature tendency in degC m^3 s^-1.

    Parameters
    ----------
    psi : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    D : float
        T (thermocline/tropical) ocean box depth in m.

    temp_atmos : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    TanmTon_ti : float, optional
        Atmosphere minus ocean N box temperature difference in degC.

    psiamoc_ti : float, optional
        AMOC circulation component in m^3 s^-1.

    TotmTon_ti : float, optional
        T minus N ocean box temperature difference in degC.

    Returns
    -------
    total : ndarray
        Ocean box [T, N, D, S] temperature tendency in degC s^-1.
    ocn_flux_comps : ndarray
        Tendency components, [T, N, D, S] * [atm, adv], in W m^-2.

    """
    # Heat inc. due to flux from atmosphere
    atm_shf = coupler.surface_sensible_heat_flux(temp_atmos, temp_ocean, parms)
    atm_lhf = coupler.surface_latent_heat_flux(humidity, temp_ocean, parms)
    atm = ((atm_shf + atm_lhf)
           / (constants.OCEAN_SPECIFIC_HEAT_CAPACITY * constants.RHO0))

    # Heat inc. in north ocean
    TotmTon = temp_ocean[0] - temp_ocean[1]

    if TotmTon_ti is not None:
        TotmTon = TotmTon_ti

    if psiamoc_ti is not None:
        psi[1] = psiamoc_ti

    n_comps = np.array([atm[1], psi[1]*TotmTon])

    # Heat inc. in south ocean
    s_comps = np.array([atm[2], psi[2]*(temp_ocean[2] - temp_ocean[3])])

    # Heat inc. in tropical box
    t_comps = np.array([atm[0], psi[2]*temp_ocean[3] - psi[1]*temp_ocean[0]
                        + psi[0]*temp_ocean[2]])

    # Heat inc. in deep ocean
    d_comps = np.array([0.0, psi[1]*temp_ocean[1]
                        - (psi[0] + psi[2])*temp_ocean[2]])

    # Total heat increments
    comps = np.array([t_comps, n_comps, d_comps, s_comps])
    ocn_flux_comps = comps*(constants.OCEAN_SPECIFIC_HEAT_CAPACITY
                            * constants.RHO0/constants.AREA_OCEAN[:, None])
    total = np.sum(comps, axis=1)
    return total, ocn_flux_comps


def delta_salinity_volume_ocean_delta_t(psi, salinity, D, humidity,
                                        temp_ocean, temp_atmos,
                                        parms, psiamoc_ti=None):
    """Ocean box volume and salinity tendency.

    Ocean box volume and salinity tendency in PSU m^3 s^-1.

    Parameters
    ----------
    psi : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.

    salinity : ndarray
        Ocean box [T, N, D, S] salnity in PSU.

    D : float
        T (thermocline/tropical) ocean box depth in m.

    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    psiamoc_ti : float, optional
        AMOC circulation component in m^3 s^-1.

    Returns
    -------
    total : ndarray
        Ocean box [T, N, D, S] salinity tendency in PSU s^-1.

    ocn_flux_comps : ndarray
        Tendency components, [T, N, D, S] * [atm, adv], in PSU s^-1.

    """
    # Unpack parms
    # v_m = parms[4]

    surf_volume = np.array([volume_thermocline(D), constants.VOLUME_N_OCEAN,
                            constants.VOLUME_S_OCEAN])
    deep_volume = constants.VOLUME_TOTAL_OCEAN - np.sum(surf_volume)

    surf_area = np.array([constants.AREA_T_OCEAN, constants.AREA_N_OCEAN,
                          constants.AREA_S_OCEAN])

    atm_lhf = coupler.surface_latent_heat_flux(humidity, temp_ocean, parms)
    evap_volume = atm_lhf/(constants.WATER_EVAPORATION_LATENT_HEAT
                           * constants.RHO_FRESH)
    prec_volume = (atmosphere.precipitation_flux(humidity, temp_atmos)
                   * surf_area / constants.RHO_FRESH)

    fwf = constants.S0 * (prec_volume + evap_volume) / surf_volume

    # fixed circulation
    if psiamoc_ti is not None:
        psi[1] = psiamoc_ti

    # Salt increments
    n_comps = np.array([fwf[1], psi[1]*(salinity[0] - salinity[1])])
    s_comps = np.array([fwf[2], psi[2]*(salinity[2] - salinity[3])])
    t_comps = np.array([fwf[0], psi[2]*salinity[3] - psi[1]*salinity[0]
                        + psi[0]*salinity[2]])
    d_comps = np.array([0.0, psi[1]*salinity[1]
                        - (psi[0] + psi[2])*salinity[2]])

    # Total salt increments
    comps = np.array([t_comps, n_comps, d_comps, s_comps])
    Vocean = np.array([surf_volume[0], surf_volume[1], deep_volume,
                       surf_volume[2]])
    ocn_flux_comps = comps/Vocean[:, None]
    total = np.sum(comps, axis=1)
    return total, ocn_flux_comps
