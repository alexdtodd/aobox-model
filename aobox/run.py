# run.py

import numpy as np
import atmosphere
import ocean
import constants

deltat = constants.TIME_STEP


def ocean_step(temp_ocean, salinity, D, parms, temp_atmos, humidity):
    #               TanmTon_ti, psiamoc_ti, TotmTon_ti):
    """Integrate ocean model forward one time step.

    Parameters
    ----------
    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    salnity : ndarray
        Ocean box [T, N, D, S] salinity in PSU

    D : float
        T (thermocline/tropical) ocean box depth in m.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    temp_atmos : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    TanmTon_ti : float
        Atmosphere minus ocean N box temperature difference in degC.
    psiamoc_ti : float
        AMOC circulation component in m^3 s^-1.
    TotmTon_ti : float
        T minus N ocean box temperature difference in degC.

    Returns
    -------
    Tostep : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    Dstep : float
        T ocean box depth in m.
    qstep : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.
    dTcomps : ndarray
        Tendency components, [T, N, D, S]*[surf, adv], in W m^-2.

    """
    # Circulation
    psistep = ocean.ocean_circulation(temp_ocean, salinity, D, parms)

    # Pycnocline depth
    Dstep = D + ocean.delta_D_delta_t(D, psistep)*deltat

    surf_volume = np.array([ocean.volume_thermocline(Dstep),
                            constants.VOLUME_N_OCEAN,
                            constants.VOLUME_S_OCEAN])
    deep_volume = constants.VOLUME_TOTAL_OCEAN - np.sum(surf_volume)
    Vocean = np.array([surf_volume[0], surf_volume[1], deep_volume,
                       surf_volume[2]])

    # Temperature:
    dToVo = ocean.delta_temp_volume_ocean_dt(psistep, temp_ocean, Dstep,
                                             temp_atmos, humidity, parms)
    # , TanmTon_ti, psiamoc_ti, TotmTon_ti)
    Tostep = temp_ocean + (dToVo[0]*deltat)/Vocean
    dTcomps = dToVo[1]

    # Salinity
    dSVo = ocean.delta_salinity_volume_ocean_delta_t(psistep, salinity, Dstep,
                                                     humidity, temp_ocean,
                                                     temp_atmos, parms)
    # , psiamoc_ti)
    Sstep = salinity + dSVo[0]*deltat/Vocean
    dScomps = dSVo[1]

    return Tostep, Sstep, Dstep, psistep, dTcomps, dScomps


def atmos_step(temp_atmos, temp_ocean, humidity, parms, temp_atmos_star,
               toa_flux, Qtoap):
    """Integrate atmosphere model forward one time step.

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

    Tar : ndarray, optional
        Atmosphere box [T, N, S] restoring temperatures in degC.

    toa_flux : ndarray, optional
        Prescribed top of atmopshere (TOA) fluxes [T, N, S] in W m^-2.

    Qtoap : ndarray, optional
        Perturbation to be applied to TOA fluxes in W m^-2.

    Returns
    -------
    Tastep : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    dTcomps : ndarray
        Tendency components, [T, N, S]*[surf, vert, TOA], in W m^-2.

    toa_flux_step : ndarray
        Diagnosed TOA fluxes [T, N, S] in W m^-2.
    """
    # Increment atmosphere
    dTa = atmosphere.delta_temp_atmos_delta_t(temp_atmos, temp_ocean, humidity,
                                              parms, toa_flux, temp_atmos_star,
                                              Qtoap)
    Tastep = temp_atmos + dTa[0]*deltat
    dTcomps = dTa[1]
    toa_flux_step = dTa[2]

    dqa = atmosphere.delta_humidity_delta_t(temp_atmos, temp_ocean, humidity,
                                            parms)
    qastep = humidity + dqa
    
    return Tastep, qastep, dTcomps, toa_flux_step


def forward_step(temp_atmos, humidity, temp_ocean, salinity, D,
                 parms, temp_atmos_star, toa_flux, Qtoap):
    """Integrate coupled model forward one time step.

    Parameters
    ----------
    temp_atmos : ndarray
        Atmosphere box [T, N, S] temperatures in degC.

    humidity : ndarray
        Atmosphere box [T, N, S] humidity in kg kg^-1.

    temp_ocean : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    salinity : ndarray
        Ocean box [T, N, D, S] salinity in PSU

    D : float
        T (thermocline/tropical) ocean box depth in m.

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    temp_atmos_star : ndarray, optional
        Atmosphere box [T, N, S] restoring temperatures in degC.

    toa_flux : ndarray, optional
        Prescribed top of atmopshere (TOA) fluxes [T, N, S] in W m^-2.

    Qtoap : ndarray, optional
        Perturbation to be applied to TOA fluxes in W m^-2.

    TanmTon_ti : float
        Atmosphere minus ocean N box temperature difference in degC.
    psiamoc_ti : float
        AMOC circulation component in m^3 s^-1.
    TotmTon_ti : float
        T minus N ocean box temperature difference in degC.

    Returns
    -------
    prog_vars : tuple
        Prognostic state: Ta, qa, To, S, and D at end of time step.

    diag_vars : tuple
        Diagnostics: circulation, atmos tendency, ocean tendency components.
    toa_flux_step : ndarray
        Diagnosed TOA fluxes [T, N, S] in W m^-2.
    """

    # Ocean step
    ocean_step_out = ocean_step(temp_ocean, salinity, D, parms, temp_atmos,
                                humidity)
    Tostep, Sstep, Dstep, psistep, dTocomps, dScomps = ocean_step_out

    # Atmosphere steps
    atmos_step_out = atmos_step(temp_atmos, temp_ocean, humidity, parms,
                                temp_atmos_star, toa_flux, Qtoap)
    Tastep, qastep, dTacomps, toa_flux_step = atmos_step_out

    # Output
    prog_vars = (Tastep, qastep, Tostep, Sstep, Dstep)
    diag_vars = (psistep, dTacomps, dTocomps, dScomps)

    return prog_vars, diag_vars, toa_flux_step


def run(init=None,
        parms=None,
        forcing_type="spinup",
        nyear=100,
        Qtoap=0.0,
        temp_atmos_star=None):
    """Run the atmosphere-ocean box model.

    Parameters
    ----------
    init : ndarray
        Initial conditions - [atmosphere temperatures, atmosphere humidity,
        ocean temperatures, tropical ocean depth, TOA fluxes].

    parms : ndarray
        Parameters [tau, kappa_v, kappa_GM, tau_q, tau_theta, v_m, lambda_LW].

    forcing_type : str, optional
        One of 'spinup', 'step', 'linear' or 'circulation';
        - 'spinup': spin up integration with TOA restoring to Tar,
        - 'step': instantaneous perturbation (Qtoap) applied to the TOA fluxes,
        - 'linear': linearly increasing perturbation (0 at t=0 to Qtoap at
          t=nyear) applied to the TOA fluxes,
        - 'circulation': prescribed ocean circulation at every time step.
    nyear : int, optional
        Number of years to run (1 year == 360 days, 1 day == 8.64e4 seconds)
    Qtoap : float or ndarray, optional
        Perturbation to TOA fluxes in W m^-2.
    Tar : ndarray, optional
        Atmosphere box [T, N, S] restoring temperatures in degC.
    TanmTon_star : ndarray, optional
        Array of atmosphere minus ocean N box temperature difference in degC.
    psiamoc_star : ndarray, optional
        Array of AMOC circulation component in m^3 s^-1.
    TotmTon_star : ndarray, optional
        Array of T minus N ocean box temperature difference in degC.

    Returns
    -------
    diagnostics : dict
        Diagnostics from the atmosphere-ocean box model run, notably:
        - 'Ta': atmosphere box temperature [T, N, S] time series.
        - 'To': ocean box temperature [T, N, D, S] time series.
        - 'D': tropical ocean box depth.
        - 'final_state': prognostic variables at end of run, to use as <init>
          in a new run.

    """
    # Default initial conditions
    if forcing_type == "spinup" and init is None:
        init = np.array([25.0, 5.0, 5.0,
                         0.03, 0.01, 0.01,
                         25.0, 5.0, 5.0, 5.0,
                         35.0, 35.0, 35.0, 35.0,
                         350.0, 0.0, 0.0, 0.0])
        temp_atmos_star = np.array([25.0, 5.0, 5.0])

    # Default parameters
    if parms is None:
        parms = np.array([1e-1, 2e-5, 2e3, 45.0, 45.0, 0.5, 1.5])

    # Model steps
    nstep = int(nyear*31104000.0/deltat)
    nsave = nstep

    # Unpack initial/boundary conditions
    Ta, qa, To, S, D = init[:3], init[3:6], init[6:10], init[10:14], init[14]
    toa_flux = init[15:]

    # Output arrays
    Ta_out, To_out = np.zeros((3, nsave)), np.zeros((4, nsave))
    qa_out = np.zeros((3, nsave))
    S_out = np.zeros((4, nsave))
    D_out = np.zeros((1, nsave))
    psistep_out = np.zeros((3, nsave))
    toa_flux_out = np.zeros((3, nsave))
    
    dTa_out, dTo_out = np.zeros((3, 4, nsave)), np.zeros((4, 2, nsave))
    dS_out = np.zeros((4, 2, nsave))
    TanmTon_out, psiamoc_out = np.zeros(nsave), np.zeros(nsave)
    TotmTon_out = np.zeros(nsave)

    # Iterate over each time step
    for ti in range(nstep):
        #print(Ta, qa, To)

        # Fixed conditions
        #psiamoc_ti = None if psiamoc_star is None else psiamoc_star[ti]
        #TanmTon_ti = None if TanmTon_star is None else TanmTon_star[ti]
        #TotmTon_ti = None if TotmTon_star is None else TotmTon_star[ti]

        # External forcing controls
        forcing = {"spinup": (np.zeros(3), np.zeros(3), temp_atmos_star),
                   "step": (np.array(Qtoap), toa_flux, None),
                   "linear": ((ti/float(nstep - 1))*np.array(Qtoap), toa_flux,
                              None)}[forcing_type]
        Qtoap_ti, toa_flux, temp_atmos_star = forcing

        # Step box model forward
        prog_vars, diag_vars, toa_flux_step = forward_step(Ta, qa, To, S, D,
                                                           parms,
                                                           temp_atmos_star,
                                                           toa_flux,
                                                           Qtoap_ti)
        Ta, qa, To, S, D = prog_vars
        psistep, dTa, dTo, dS = diag_vars

        # Store the output
        Ta_out[:, ti] = Ta
        qa_out[:, ti] = qa
        To_out[:, ti] = To
        S_out[:, ti] = S
        D_out[0, ti] = D
        psistep_out[:, ti] = psistep
        dTa_out[:, :, ti] = dTa
        dTo_out[:, :, ti] = dTo
        dS_out[:, :, ti] = dS
        TanmTon_out[ti] = Ta[1] - To[1]
        psiamoc_out[ti] = psistep[1]
        TotmTon_out[ti] = To[0] - To[1]
        toa_flux_out[:, ti] = toa_flux_step
        #print(toa_flux_step)

    # Output diagnostics
    final_state = np.concatenate((Ta, qa, To, S, [D], toa_flux_step))

    # Calculate annual means
    diagnostics = {"final_state": final_state, "parm": parms,
                   "Ta": Ta_out, "qa": qa_out, "To": To_out, "D": D_out,
                   "S": S_out, "dS_out": dS_out,
                   "psistep": psistep_out, "dTa": dTa_out, "dTo": dTo_out,
                   "TanmTon": TanmTon_out, "psiamoc": psiamoc_out,
                   "TotmTon": TotmTon_out,
                   "Qsurf": dTo_out[:, 0], "Psi": psistep_out*1e-6,
                   "TOA": toa_flux_out}
    return diagnostics
