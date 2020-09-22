import numpy as np
import matplotlib.pyplot as plt

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
Aocean = np.array([A, An, A + An + As, As])

# Atmosphere dimensions
Aan = 2*np.pi*(1-np.sin(phin))*a**2    # N atmosphere box horizontal area [m^2]
Aas = Aan                              # S atmopshere box horizontal area [m^2]
Aglobe = 4.0*np.pi*a**2                # Earth horizontal area [m^2]
Aatmos = np.array([Aglobe - 2*Aan, Aan, Aas])

hatmos = 1.5e4                          # atmosphere box depth [m]
Vatmos = Aatmos*hatmos                  # atmosphere box volumes [m^3]
nbound = hatmos*2*np.pi*a*np.cos(phin)  # interface between N and T box [m^2]
sbound = nbound                         # interface between S and T box [m^2]
atm_inds = np.array([0, 1, 3])


# Atmosphere functions
def dTadt(Ta, To, parms, Tar=None, toa_flux=None, Qtoap=0.0):
    """
    Atmosphere box temperature tendency in degC s^-1 using either
    TOA restorting or TOA fluxes.

    Args:
        Ta (numpy.array): Atmosphere box [T, N, S] temperatures in degC.
        To (numpy.array): Ocean box [T, N, D, S] temperatures in degC.
        parms (numpy.array): Parameters in the form [tau, kappa_v, kappa_GM,
    tau_s, $v_m$, lambda_LW].

    Kwargs:
        Tar (numpy.array): Atmosphere box [tropical, north, south] restoring temperatures in degC.
        toa_flux (numpy.array): Prescribed top of atmopshere (TOA) fluxes [tropical, north, south] in W m^-2.
        Qtoap (float or numpy.array): Perturbation to TOA fluxes in W m^-2.
        
    Returns:
        tend (numpy.array): Atmosphere box [tropical, north, south] temperature tendency in degC s^-1.
        flux_comps (numpy.array): Tendency components, [tropical, north, south]*[surf, mix, TOA], in W m^-2.
        toa_flux (numpy.array): TOA column of flux_comps, in W m^-2.
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
    return A*D                  


def rho(To, S=S0):
    """
    Ocean density from linear equation of state.
    
    Args:
        To (float or numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        
    Kwargs:
        S (float or numpy.array): Ocean box [tropical, north, deep, south] salinity in g kg^-1, optional.
        
    Returns:
        density (float or numpy.array): Ocean box [tropical, north, deep, south] density in kg m^-3.
    """
    density = rho0*(1 - alpha*(To - T0) + beta*(S - S0))
    return density
    
def q(To, D, parms):
    """
    Ocean circulation components.
    
    Args:
        To (float or numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        D (float): Tropical ocean box depth in m.
        parms (numpy.array): Parameters in the form [$\tau$, $\kappa_{v}$, $\kappa_{GM}$,
                             $\tau_s$, $v_m$, $\lambda_{LW}$].
    Returns:
        psi (numpy.array): Ocean circulation components [up, AMOC, south] in m^3 s^-1.
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
    """
    Tropical ocean box depth tendency.
    
    Args:
        D (float): Tropical ocean box depth in m.
        q (numpy.array): Ocean circulation components [up, AMOC, south] in m^3 s^-1.
        
    Returns:
        tend (float): Tropical ocean box depth tendency in m s^-1.
    """
    tend = (1.0/A)*(q[2] + q[0] - q[1])
    return tend

def deltaToVo(q, To, D, Ta, parms, TanmTon_ti=None, psiamoc_ti=None, TotmTon_ti=None):
    """
    Atmosphere box temperature tendency in degC s^-1 using either TOA restorting or TOA fluxes.
    
    Args:
        q (numpy.array): Ocean circulation components [up, AMOC, south] in m^3 s^-1.
        To (numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        D (float): Tropical ocean box depth in m.
        Ta (numpy.array): Atmosphere box [tropical, north, south] temperatures in degC.
        parms (numpy.array): Parameters in the form [$\tau$, $\kappa_{v}$, $\kappa_{GM}$,
                             $\tau_s$, $v_m$, $\lambda_{LW}$].
        To_ti (numpy.array): Prescribed ocean temperatures [tropical, north, deep, south] in degC.
        
    Returns:
        total (numpy.array): Ocean box [tropical, north, deep, south] temperature*volume tendency in degC m^3 s^-1.
        flux_comps (numpy.array): Tendency components, [tropical, north, deep, south]*[surf, horiz, vert], in W m^-2.
    """
    # Unpack parms
    tau_s = parms[3]
    lambda_s = 1/(day_sec*tau_s)
    Voce = np.array([Vt(D), Vn, Vs])
    hoce = np.array([D, hn, hs])
    
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
    flux_comps = comps*(cp*rho0/Aocean[:,None])
    total = np.sum(comps, axis=1)
    return total, flux_comps


# Forward model
def ocean_step(To, D, parms, Ta, TanmTon_ti, psiamoc_ti, TotmTon_ti):
    """
    Integrate ocean model forward one time step.
    
    Args:
        To (numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        D (float): Tropical ocean box depth in m.
        parms (numpy.array): Parameters in the form [$\tau$, $\kappa_{v}$, $\kappa_{GM}$,
                             $\tau_s$, $v_m$, $\lambda_{LW}$].
        Ta (numpy.array): Atmosphere box [tropical, north, south] temperatures in degC.
        psi (numpy.array): Ocean circulation components [up, AMOC, south] in m^3 s^-1.
        To_ti (numpy.array): Prescribed ocean temperatures [tropical, north, deep, south] in degC.
        
    Returns:
        Tostep (numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        Dstep (float): Tropical ocean box depth in m.
        qstep (numpy.array): Ocean circulation components [up, AMOC, south] in m^3 s^-1.
        dTcomps (numpy.array): Tendency components, [tropical, north, deep, south]*[surf, horiz, vert], in W m^-2.
    """
    # Circulation
    qstep = q(To, D, parms)
    
    # Pycnocline depth
    Dstep = D + dDdt(D, qstep)*deltat
    Vocean = np.array([Vt(Dstep), Vn, Vtotal - (Vn + Vs + Vt(Dstep)), Vs])
    hocean = np.array([Dstep, hn, Vocean[2]/A, hs])

    # Total heat:
    dToVo = deltaToVo(qstep, To, Dstep, Ta, parms, TanmTon_ti, psiamoc_ti, TotmTon_ti)
    Tostep = To + dToVo[0]*deltat/Vocean
    dTcomps = dToVo[1]#*cp*rho0*hocean[:,None]/Vocean[:,None]
    
    return Tostep, Dstep, qstep, dTcomps

def atmos_step(Ta, To, parms, Tar, toa_flux, Qtoap):
    """
    Integrate atmosphere model forward one time step.
    
    Args:
        Ta (numpy.array): Atmosphere box [tropical, north, south] temperatures in degC.
        To (numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        parms (numpy.array): Parameters in the form [$\tau$, $\kappa_{v}$, $\kappa_{GM}$,
                             $\tau_s$, $v_m$, $\lambda_{LW}$].
        Tar (numpy.array): Atmosphere box [tropical, north, south] restoring temperatures in degC.
        toa_flux (numpy.array): Prescribed top of atmopshere (TOA) fluxes [tropical, north, south] in W m^-2.
        Qtoap (float or numpy.array): Perturbation to TOA fluxes in W m^-2.
        
    Returns:
        Tastep (numpy.array): Atmosphere box [tropical, north, south] temperatures in degC.
        dTcomps (numpy.array): Tendency components, [tropical, north, deep, south]*[surf, horiz, vert], in W m^-2.
        toa_flux_step (numpy.array): Diagnosed TOA fluxes [tropical, north, south] in W m^-2.
    """
    # Increment atmosphere
    dTa = dTadt(Ta, To, parms, Tar, toa_flux, Qtoap)
    Tastep = Ta + dTa[0]*deltat
    dTcomps = dTa[1]
    toa_flux_step = dTa[2]

    return Tastep, dTcomps, toa_flux_step


def forward_step(Ta, To, D, parms, Tar, toa_flux, Qtoap, TanmTon_ti, psiamoc_ti, TotmTon_ti):
    """
    Integrate coupled model forward one time step.
    
    Args:
        Ta (numpy.array): Atmosphere box [tropical, north, south] temperatures in degC.
        To (numpy.array): Ocean box [tropical, north, deep, south] temperatures in degC.
        D (float): Tropical ocean box depth in m.
        parms (numpy.array): Parameters in the form [$\tau$, $\kappa_{v}$, $\kappa_{GM}$,
                             $\tau_s$, $v_m$, $\lambda_{LW}$].
        Tar (numpy.array): Atmosphere box [tropical, north, south] restoring temperatures in degC.
        toa_flux (numpy.array): Prescribed top of atmopshere (TOA) fluxes [tropical, north, south] in W m^-2.
        Qtoap (float or numpy.array): Perturbation to TOA fluxes in W m^-2.
        psi_ti (numpy.array): Prescribed ocean circulation components [up, AMOC, south] in m^3 s^-1.
        To_ti (numpy.array): Prescribed ocean temperatures [tropical, north, deep, south] in degC.
        
    Returns:
        prog_vars (tuple): Prognostic state, [Ta, To, D], at end of time step.
        diag_vars (tuple): Diagnostics, [circulation, atmosphere tendency, ocean tendency] components, at end of time step.
        toa_flux_step (numpy.array): Diagnosed TOA fluxes [tropical, north, south] in W m^-2.
    """
    
    # Ocean step
    Tostep, Dstep, qstep, dTocomps = ocean_step(To, D, parms, Ta, TanmTon_ti, psiamoc_ti, TotmTon_ti)
    
    # Atmosphere steps
    Tastep, dTacomps, toa_flux_step = atmos_step(Ta, To, parms, Tar, toa_flux, Qtoap)
    
    # Output
    prog_vars = (Tastep, Tostep, Dstep)
    diag_vars = (qstep, dTacomps, dTocomps)
    
    return prog_vars, diag_vars, toa_flux_step


def run(init,
        parms,
        forcing_type="spinup",
        nyear=1000,
        Qtoap=0.0,
        Tar=None,
        psi_star=None,
        To_star=None,
        freq=None,
        TanmTon_star=None,
        psiamoc_star=None,
        TotmTon_star=None):
    """
    Run the atmosphere-ocean box model.
    
    Args:
        init (numpy.array): Initial conditions - [atmosphere temperatures, ocean temperatures, tropical ocean depth, TOA fluxes].
        parms (numpy.array): Parameters - [$\tau$, $\kappa_{v}$, $\kappa_{GM}$, $\tau_s$, $v_m$, $\lambda_{LW}$].
        
    Kwargs:
        forcing_type (str): One of 'spinup', 'step', 'linear' or 'circulation';
                            - 'spinup': spin up integration with TOA restoring to Tar,
                            - 'step': instantaneous perturbation (Qtoap) applied to the TOA fluxes,
                            - 'linear': linearly increasing perturbation (0 at t=0 to Qtoap at t=nyear) applied to the TOA fluxes,
                            - 'circulation': prescribed ocean circulation at every time step.
        nyear (int): Number of years to run the model (1 year == 360 days, 1 day == 8.64e4 seconds)
        Qtoap (float or numpy.array): Perturbation to TOA fluxes in W m^-2.
        Tar (numpy.array): Atmosphere box [tropical, north, south] restoring temperatures in degC.
        psi (numpy.array): Ocean circulation components [time]*[up, AMOC, south] in m^3 s^-1.
        freq (float): time frequency with which to save diagnostic time series (default = nyear/10).
        
    Returns:
        diagnostics (dict): Diagnostics from the atmosphere-ocean box model run, of particular interest are:
                            - 'psi': ocean circulation components at every time step, to use in a 'circulation' run.
                            - 'Ta': atmosphere box temperature [tropical, north, south] time series.
                            - 'To': ocean box temperature [tropical, north, deep, south] time series.
                            - 'D': tropical ocean box depth.
                            - 'Qsurf': atmosphere to ocean (downwards) heat flux [tropical, north, south] time series.
                            - 'final_state': prognostic variables at end of run, to use as init in a new run.
    """
    
    # Initial conditions
    if forcing_type == "spinup" and init is None:
        init = np.array([25.0, 5.0, 5.0, 25.0, 5.0, 5.0, 5.0, 350.0, 0.0, 0.0, 0.0])
        Tar = np.array([25.0, -10.0, 5.0])
        
    if parms is None:
        parms = np.array([1e-1, 2e-5, 2e3, 45.0, 0.5, 1.5])
    
    # Run controls
    if freq is None:
        freq = int(360*nyear*0.1)
        
    nstep = int(nyear*31104000.0/deltat)
    nsave = int(nstep/float(freq))
    
    # Unpack initial/boundary conditions
    Ta, To, D, toa_flux = init[:3], init[3:7], init[7], init[8:]
    
    # Output arrays
    if True:#forcing_type != "spinup":
        Ta_out, To_out, D_out = np.zeros((3, nstep)), np.zeros((4, nstep)), np.zeros((1, nstep))
        qstep_out, dTa_out, dTo_out = np.zeros((3, nstep)), np.zeros((3, 3, nstep)), np.zeros((4, 2, nstep))
        TanmTon_out, psiamoc_out, TotmTon_out = np.zeros(nstep), np.zeros(nstep), np.zeros(nstep)
    
    # Iterate over time steps
    for ti in range(nstep):
        
        # Fixed conditions
        psiamoc_ti = None if psiamoc_star is None else psiamoc_star[ti]
        TanmTon_ti = None if TanmTon_star is None else TanmTon_star[ti]
        TotmTon_ti = None if TotmTon_star is None else TotmTon_star[ti]
        
        # External forcing controls
        forcing = {"spinup":(np.zeros(3), np.zeros(3), Tar),
                   "step":(np.array(Qtoap), toa_flux, None),
                   "linear":((ti/float(nstep - 1))*np.array(Qtoap), toa_flux, None)}[forcing_type]
        Qtoap_ti, toa_flux, Tar = forcing
        
        # Step box model forward
        prog_vars, diag_vars, toa_flux_step = forward_step(Ta, To, D, parms,
                                                           Tar, toa_flux, Qtoap_ti,
                                                           TanmTon_ti, psiamoc_ti, TotmTon_ti)
        Ta, To, D = prog_vars
        qstep, dTa, dTo = diag_vars
        
        # Store the output
        if True:#forcing_type != "spinup":
            Ta_out[:,ti] = Ta
            To_out[:,ti] = To
            D_out[0,ti] = D
            qstep_out[:,ti] = qstep
            dTa_out[:,:,ti] = dTa
            dTo_out[:,:,ti] = dTo
            TanmTon_out[ti] = Ta[1] - To[1]
            psiamoc_out[ti] = qstep[1]
            TotmTon_out[ti] = To[0] - To[1]
    
    # Output diagnostics
    final_state = np.concatenate((Ta, To, [D], toa_flux_step))
    
    # Calculate annual means
    #if forcing_type != "spinup":
    special_diagnostics = {"final_state":final_state, "qstep":qstep_out,
                           "Qsurf":-1.0*dTa_out[:,0], "parm":parms}
    diagnostics = {"final_state":final_state, "Ta":Ta_out, "To":To_out, "D":D_out,
                   "qstep":qstep_out, "dTa":dTa_out, "dTo":dTo_out,
                   "TanmTon":TanmTon_out, "psiamoc":psiamoc_out, "TotmTon":TotmTon_out,
                   "parm":parms, "Qsurf":-1.0*dTa_out[1,0], "Psi":psiamoc_out*1e-6}
    #else:
    #    diagnostics = {"final_state":final_state}
    
    return diagnostics


# Diagnostics plot
def dashboard(diags):
    """
    Make a dashboard plot showing the time series of key variables from an AO box model run.
    """
    
    # Colors
    colors3 = plt.get_cmap("RdYlBu")(np.linspace(0, 1, 3))
    circ_colors = plt.get_cmap("viridis")(np.linspace(0, 1, 3))
    circ_labels = ["up", "AMOC", "south"]
    
    colors = {3:colors3,
              4:[colors3[0], colors3[1], "k", colors3[2]]}
    boxes = {3:["tropics", "north", "south"],
             4:["tropics", "north", "deep", "south"]}
    
    diag_names = [["Ta", "To", "AOQsurf"],
                  ["Qtoa", "Qmix", "OAQsurf"],
                  ["circ"]]
    units = ["degC", "W m$^{-2}$", "Sv"]
    nt = diags["psi"].shape[1]
    time = np.linspace(0, nt/360, 10)
    #time = np.arange(diags[diag_names[0][0]].shape[-1])
    
    fig, axes = plt.subplots(3, 3, figsize=(6, 4), dpi=150, sharex=True,
                             tight_layout=True)#, sharey="row")
    for ri, dnames in enumerate(diag_names):
        for ci, dname in enumerate(dnames):
            ax = axes[ri,ci]
            plt.sca(ax)
            
            data = diags[dname]
            nbox = data.shape[0]
            
            for bi, box in enumerate(boxes[nbox]):
                plt.plot(time, data[bi],
                         color=colors[nbox][bi] if dname != "circ" else circ_colors[bi],
                         label=box if dname != "circ" else circ_labels[bi])
            plt.ylabel("{} [{}]".format(dname, units[ri]))
            
            if dname == "To":
                h2, l2 = ax.get_legend_handles_labels()
            elif dname == "circ":
                h1, l1 = ax.get_legend_handles_labels()
                
            
        if ri == 2:
            plt.xlabel("time [yr]")
            
    ax1 = axes[-1,-2]
    ax1.legend(h1, l1)
    ax1.axis("off")
    
    ax2 = axes[0,-1]
    #ax2.legend(h2, l2)
    #ax2.axis("off")
    
    ax3 = axes[-1,-1]
    ax3.legend(h2, l2)
    ax3.axis("off")
    return None
    


