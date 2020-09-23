# model.py

import numpy as np

# General model constants
deltat = 8.64e4   # Model time step [s]
day_sec = 8.64e4  # Seconds per day [s day^-1]
m3_to_Sv = 1e-6   # Sv per m^3 [Sv m^-3]

# Ocean constants
rhoa = 1.225      # atmosphere reference density [kg m^-3]
rho0 = 1027.5     # ocean reference density [kg m^-3]
cpa = 1e3         # atmosphere specific heat capacity [J K^-1 kg^-1]
cp = 4.2e3        # ocean specific heat capacity [J K^-1 kg^-1]
S0 = 35.0         # reference salinity [g kg^-1]
T0 = 5.0          # reference temperature [degC]
alpha = 2e-4      # thermal expansion coefficient [degC^-1]
beta = 8e-4       # haline contraction coefficient [g^-1 kg]
g = 9.8           # acceleration due to gravity [m s^-2]
fn = 1e-4         # Coriolis parameter at north/tropical box interface [s^-1]
fs = -1.0*fn      # Coriolis parameter at south/tropical box interface [s^-1]

# Atmosphere constants
a = 6.371e6                       # Earth radius [m]
omega = (2*np.pi)/day_sec         # Earth rotation rate [s^-1]
phin = np.arcsin(fn/(2.0*omega))  # Latitude at f = fn [rad]
lambda_r = 1.0/(5.0*day_sec)      # TOA relaxation time scale [s^-1]

# Ocean dimensions
A = 2.6e14                          # air-sea interface area of T box [m^2]
Lx = 3e7                            # zonal extent of S ocean box [m]
Ly = 1e6                            # meridional extent of S ocean box [m]
Vn = 3e15                           # volume of N ocean box [m^3]
Vs = 9e15                           # volume of S ocean box [m^3]
Vtotal = 1.2e18                     # total volume of all ocean boxes [m^3]
h = 5e2                             # mixed layer depth [m]
hn = h                              # north ocean box depth [m]
hs = h                              # south ocean box depth [m]
An = Vn/h                           # air-sea interface area of N box [m^2]
As = Vs/h                           # air-sea interface area of S box [m^2]
Aocean = np.array([A, An, A + An + As, As])  # Ocean box areas

# Atmosphere dimensions
Aan = 2*np.pi*(1-np.sin(phin))*a**2    # N atmosphere box horizontal area [m^2]
Aas = Aan                              # S atmopshere box horizontal area [m^2]
Aglobe = 4.0*np.pi*a**2                # Earth horizontal area [m^2]
Aatmos = np.array([Aglobe - 2*Aan, Aan, Aas])  # Atmosphere box areas

hatmos = 1.5e4                          # atmosphere box depth [m]
Vatmos = Aatmos*hatmos                  # atmosphere box volumes [m^3]
nbound = hatmos*2*np.pi*a*np.cos(phin)  # interface between N and T box [m^2]
sbound = nbound                         # interface between S and T box [m^2]
atm_inds = np.array([0, 1, 3])


# Atmosphere functions
def dTadt(Ta, To, parms, Tar=None, toa_flux=None, Qtoap=0.0):
    """Atmosphere box temperature tendency.

    Atmosphere box temperature tendency in degC s^-1 from applying either
    TOA restorting or TOA fluxes.

    Parameters
    ----------
    Ta : ndarray
        Atmosphere box [T, N, S] temperatures in degC.
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
    Tar : ndarray, optional
        Atmosphere box [T, N, S] restoring temperatures in degC.
    toa_flux : ndarray, optional
        Prescribed top of atmopshere (TOA) fluxes [T, N, S] in W m^-2.
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
    tau_s, v_m, lambda_LW = parms[3:]
    lambda_s = 1/(day_sec*tau_s)

    # convert J s-1 to K s-1
    heat_to_temp = 1.0/(cpa*rhoa*Vatmos)

    # Heat from TOA flux
    if Tar is None:
        toa = Aatmos*(toa_flux + Qtoap)

    else:
        toa = lambda_r*(Tar - Ta)/heat_to_temp
        toa_flux = toa/Aatmos

    # Heat from LW feedback
    net_toa = toa - lambda_LW*Ta*Aatmos

    # Heat from meridional mixing
    nmix = nbound*v_m*(Ta[0] - Ta[1])*cpa*rhoa
    smix = sbound*v_m*(Ta[0] - Ta[2])*cpa*rhoa
    mix = np.array([-1*(nmix + smix), nmix, smix])

    # Heat from surface flux
    atm_inds = np.array([0, 1, 3])
    oce = lambda_s*rhoa*cpa*Vatmos*(To[atm_inds] - Ta)

    # Total heat tendency
    comps = np.array([oce, mix, net_toa]).T
    total = np.sum(comps, axis=1)

    # Convert heat tend to temp tend
    tend = total*heat_to_temp
    flux_comps = comps/Aatmos
    return tend, flux_comps, toa_flux


# Ocean functions
def Vt(D):
    """Volume of T (thermocline/tropical) ocean box."""
    return A*D


def rho(To, S=S0):
    """Ocean density from linear equation of state.

    Parameters
    ----------
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    S : ndarray, optional
        Ocean box [T, N, D, S] salinity in PSU.

    Returns
    -------
    density : ndarray
        Ocean box [T, N, D, S] density in kg m^-3.

    """
    density = rho0*(1 - alpha*(To - T0) + beta*(S - S0))
    return density


def q(To, D, parms):
    """Ocean circulation components.

    Parameters
    ----------
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.

    D : float
        T (thermocline/tropical) ocean box depth in m.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].

    Returns
    -------
    psi : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.

    """
    # Unpack parms
    tau, kappav, kappagm = parms[:3]

    # tropics -> north and north -> deep
    rhot, rhon = rho(To)[:2]
    gp = g*(rhon - rhot)/rho0
    qn = gp*(D**2)/(2.0*fn)

    # deep -> tropics
    qu = kappav*A/D

    # deep -> south and south -> tropics
    q_Ek = tau*Lx/(rho0*abs(fs))
    q_eddy = kappagm*D*Lx/Ly
    qs = q_Ek - q_eddy

    # total circulation
    psi = np.array([qu, qn, qs])
    return psi


def dDdt(D, q):
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
    tend = (1.0/A)*(q[2] + q[0] - q[1])
    return tend


def deltaToVo(q, To, D, Ta, parms, TanmTon_ti=None, psiamoc_ti=None,
              TotmTon_ti=None):
    """Ocean box volume and temperature tendency.

    Ocean box volume and temperature tendency in degC m^3 s^-1.

    Parameters
    ----------
    q : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    D : float
        T (thermocline/tropical) ocean box depth in m.
    Ta : ndarray
        Atmosphere box [T, N, S] temperatures in degC.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
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
    # Unpack parms
    tau_s = parms[3]
    lambda_s = 1/(day_sec*tau_s)

    # Heat inc. due to flux from atmosphere
    TamTo = Ta - To[atm_inds]

    if TanmTon_ti is not None:
        TamTo[1] = TanmTon_ti

    atm = lambda_s*rhoa*cpa*hatmos*TamTo*(Aocean[atm_inds]/(cp*rho0))

    # Heat inc. in north ocean
    TotmTon = To[0] - To[1]

    if TotmTon_ti is not None:
        TotmTon = TotmTon_ti

    if psiamoc_ti is not None:
        q[1] = psiamoc_ti

    n_comps = np.array([atm[1], q[1]*TotmTon])

    # Heat inc. in south ocean
    s_comps = np.array([atm[2], q[2]*(To[2] - To[3])])

    # Heat inc. in tropical box
    t_comps = np.array([atm[0], q[2]*To[3] - q[1]*To[0] + q[0]*To[2]])

    # Heat inc. in deep ocean
    d_comps = np.array([0.0, q[1]*To[1] - (q[0] + q[2])*To[2]])

    # Total heat increments
    comps = np.array([t_comps, n_comps, d_comps, s_comps])
    ocn_flux_comps = comps*(cp*rho0/Aocean[:, None])
    total = np.sum(comps, axis=1)
    return total, ocn_flux_comps


def deltaSVo(q, S, D, parms, psiamoc_ti=None):
    """Ocean box volume and salinity tendency.

    Ocean box volume and salinity tendency in PSU m^3 s^-1.

    Parameters
    ----------
    q : ndarray
        Ocean circulation components [up, AMOC, south] in m^3 s^-1.
    S : ndarray
        Ocean box [T, N, D, S] salnity in PSU.
    D : float
        T (thermocline/tropical) ocean box depth in m.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
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

    # atm
    E = 0.25e6  # <- possibly scale by v_m ?
    atm = np.array([-2.0*E*S0, E*S0, E*S0])

    # fixed circulation
    if psiamoc_ti is not None:
        q[1] = psiamoc_ti

    # Salt increments
    n_comps = np.array([atm[1], q[1]*(S[0] - S[1])])
    s_comps = np.array([atm[2], q[2]*(S[2] - S[3])])
    t_comps = np.array([atm[0], q[2]*S[3] - q[1]*S[0] + q[0]*S[2]])
    d_comps = np.array([0.0, q[1]*S[1] - (q[0] + q[2])*S[2]])

    # Total salt increments
    comps = np.array([t_comps, n_comps, d_comps, s_comps])
    Vocean = np.array([Vt(D), Vn, Vtotal - (Vn + Vs + Vt(D)), Vs])
    ocn_flux_comps = comps/Vocean
    total = np.sum(comps, axis=1)
    return total, ocn_flux_comps


# Forward model
def ocean_step(To, S, D, parms, Ta, TanmTon_ti, psiamoc_ti, TotmTon_ti):
    """Integrate ocean model forward one time step.

    Parameters
    ----------
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    S : ndarray
        Ocean box [T, N, D, S] salinity in PSU
    D : float
        T (thermocline/tropical) ocean box depth in m.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
    Ta : ndarray
        Atmosphere box [T, N, S] temperatures in degC.
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
    qstep = q(To, D, parms)

    # Pycnocline depth
    Dstep = D + dDdt(D, qstep)*deltat
    Vocean = np.array([Vt(Dstep), Vn, Vtotal - (Vn + Vs + Vt(Dstep)), Vs])

    # Total heat:
    dToVo = deltaToVo(qstep, To, Dstep, Ta, parms, TanmTon_ti, psiamoc_ti,
                      TotmTon_ti)
    Tostep = To + dToVo[0]*deltat/Vocean
    dTcomps = dToVo[1]

    # Salinity
    dSVo = deltaSVo(qstep, S, Dstep, parms, psiamoc_ti)
    Sstep = S + dSVo[0]*deltat/Vocean
    dScomps = dSVo[1]

    return Tostep, Sstep, Dstep, qstep, dTcomps, dScomps


def atmos_step(Ta, To, parms, Tar, toa_flux, Qtoap):
    """Integrate atmosphere model forward one time step.

    Parameters
    ----------
    Ta : ndarray
        Atmosphere box [T, N, S] temperatures in degC.
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
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
    dTa = dTadt(Ta, To, parms, Tar, toa_flux, Qtoap)
    Tastep = Ta + dTa[0]*deltat
    dTcomps = dTa[1]
    toa_flux_step = dTa[2]

    return Tastep, dTcomps, toa_flux_step


def forward_step(Ta, To, S, D, parms, Tar, toa_flux, Qtoap,
                 TanmTon_ti, psiamoc_ti, TotmTon_ti):
    """Integrate coupled model forward one time step.

    Parameters
    ----------
    Ta : ndarray
        Atmosphere box [T, N, S] temperatures in degC.
    To : ndarray
        Ocean box [T, N, D, S] temperatures in degC.
    S : ndarray
        Ocean box [T, N, D, S] salinity in PSU
    D : float
        T (thermocline/tropical) ocean box depth in m.
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
    Tar : ndarray, optional
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
        Prognostic state: Ta, To, D, at end of time step.
    diag_vars : tuple
        Diagnostics: circulation, atmos tendency, ocean tendency components.
    toa_flux_step : ndarray
        Diagnosed TOA fluxes [T, N, S] in W m^-2.
    """

    # Ocean step
    Tostep, Dstep, Sstep, qstep, dTocomps, dScomps = ocean_step(To, S, D,
                                                                parms,
                                                                Ta, TanmTon_ti,
                                                                psiamoc_ti,
                                                                TotmTon_ti)
    # Atmosphere steps
    Tastep, dTacomps, toa_flux_step = atmos_step(Ta, To, parms, Tar, toa_flux,
                                                 Qtoap)
    # Output
    prog_vars = (Tastep, Tostep, Sstep, Dstep)
    diag_vars = (qstep, dTacomps, dTocomps, dScomps)

    return prog_vars, diag_vars, toa_flux_step


def run(init,
        parms,
        forcing_type="spinup",
        nyear=1000,
        Qtoap=0.0,
        Tar=None,
        TanmTon_star=None,
        psiamoc_star=None,
        TotmTon_star=None):
    """Run the atmosphere-ocean box model.

    Parameters
    ----------
    init : ndarray
        Initial conditions - [atmosphere temperatures, ocean temperatures,
        tropical ocean depth, TOA fluxes].
    parms : ndarray
        Parameters in the form [tau, kappa_v, kappa_GM, tau_s, v_m, lambda_LW].
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
        init = np.array([25.0, 5.0, 5.0, 25.0, 5.0, 5.0, 5.0,
                         35.0, 35.0, 35.0, 35.0,
                         350.0, 0.0, 0.0, 0.0])
        Tar = np.array([25.0, -10.0, 5.0])

    # Default parameters
    if parms is None:
        parms = np.array([1e-1, 2e-5, 2e3, 45.0, 0.5, 1.5])

    # Model steps
    nstep = int(nyear*31104000.0/deltat)
    nsave = nstep

    # Unpack initial/boundary conditions
    Ta, To, S, D = init[:3], init[3:7], init[7:11], init[11]
    toa_flux = init[12:]

    # Output arrays
    Ta_out, To_out = np.zeros((3, nsave)), np.zeros((4, nsave))
    S_out = np.zeros((4, nsave))
    D_out = np.zeros((1, nsave))
    qstep_out = np.zeros((3, nsave))
    dTa_out, dTo_out = np.zeros((3, 3, nsave)), np.zeros((4, 2, nsave))
    dS_out = np.zeros((4, 2, nsave))
    TanmTon_out, psiamoc_out = np.zeros(nsave), np.zeros(nsave)
    TotmTon_out = np.zeros(nsave)

    # Iterate over each time step
    for ti in range(nstep):

        # Fixed conditions
        psiamoc_ti = None if psiamoc_star is None else psiamoc_star[ti]
        TanmTon_ti = None if TanmTon_star is None else TanmTon_star[ti]
        TotmTon_ti = None if TotmTon_star is None else TotmTon_star[ti]

        # External forcing controls
        forcing = {"spinup": (np.zeros(3), np.zeros(3), Tar),
                   "step": (np.array(Qtoap), toa_flux, None),
                   "linear": ((ti/float(nstep - 1))*np.array(Qtoap), toa_flux,
                              None)}[forcing_type]
        Qtoap_ti, toa_flux, Tar = forcing

        # Step box model forward
        prog_vars, diag_vars, toa_flux_step = forward_step(Ta, To, S, D, parms,
                                                           Tar, toa_flux,
                                                           Qtoap_ti,
                                                           TanmTon_ti,
                                                           psiamoc_ti,
                                                           TotmTon_ti)
        Ta, To, S, D = prog_vars
        qstep, dTa, dTo, dS = diag_vars

        # Store the output
        Ta_out[:, ti] = Ta
        To_out[:, ti] = To
        S_out[:, ti] = S
        D_out[0, ti] = D
        qstep_out[:, ti] = qstep
        dTa_out[:, :, ti] = dTa
        dTo_out[:, :, ti] = dTo
        dS_out[:, :, ti] = dS
        TanmTon_out[ti] = Ta[1] - To[1]
        psiamoc_out[ti] = qstep[1]
        TotmTon_out[ti] = To[0] - To[1]

    # Output diagnostics
    final_state = np.concatenate((Ta, To, [D], toa_flux_step))

    # Calculate annual means
    diagnostics = {"final_state": final_state, "parm": parms,
                   "Ta": Ta_out, "To": To_out, "D": D_out,
                   "qstep": qstep_out, "dTa": dTa_out, "dTo": dTo_out,
                   "TanmTon": TanmTon_out, "psiamoc": psiamoc_out,
                   "TotmTon": TotmTon_out,
                   "Qsurf": -1.0*dTa_out[1, 0], "Psi": psiamoc_out*1e-6}
    return diagnostics
