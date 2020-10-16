# constants.py

import numpy as np

# General model constants
TIME_STEP = 8.64e4
SEC_PER_DAY = 8.64e4

# Dimension constants
GRAVITY = 9.8
CORIOLIS_N = 1e-4
CORIOLIS_S = -1.0*CORIOLIS_N

# Atmosphere constants
EARTH_RADIUS = 6.371e6
EARTH_ROTATION = (2*np.pi)/SEC_PER_DAY
LATITUDE_N = np.arcsin(CORIOLIS_N/(2.0*EARTH_ROTATION))
TOA_RELAX = 1.0/(2.0*SEC_PER_DAY)
SATURATION_VAPOUR_PRESSURE = 2.525e11
SURFACE_PRESSURE = 1.0135e5
WATER_EVAPORATION_LATENT_HEAT = 2.501e6
WATER_GAS_CONSTANT = 461.5
ATMOS_SPECIFIC_HEAT_CAPACITY = 1e3
MAX_RELATIVE_HUMIDITY = 0.77
RHO_FRESH = 1e3
RHO_ATMOS = 1.225
DEGC_TO_KELVIN = 273.15

# Ocean constants
RHO0 = 1027.5     # ocean reference density [kg m^-3]
S0 = 35.0         # reference salinity [g kg^-1]
T0 = 5.0          # reference temperature [degC]
ALPHA = 2e-4      # thermal expansion coefficient [degC^-1]
BETA = 8e-4       # haline contraction coefficient [g^-1 kg]
GRAVITY = 9.8           # acceleration due to gravity [m s^-2]

# Ocean dimensions
AREA_T_OCEAN = 2.6e14
S_OCEAN_X = 3e7
S_OCEAN_Y = 1e6
VOLUME_N_OCEAN = 3e15
VOLUME_S_OCEAN = 9e15
VOLUME_TOTAL_OCEAN = 1.2e18
EXTRATROPICAL_MIXED_LAYER_DEPTH = 5e2
OCEAN_SPECIFIC_HEAT_CAPACITY = 4.2e3
DEPTH_N_OCEAN = EXTRATROPICAL_MIXED_LAYER_DEPTH
DEPTH_S_OCEAN = EXTRATROPICAL_MIXED_LAYER_DEPTH
AREA_N_OCEAN = VOLUME_N_OCEAN/DEPTH_N_OCEAN
AREA_S_OCEAN = VOLUME_S_OCEAN/DEPTH_S_OCEAN

AREA_OCEAN = np.array([AREA_T_OCEAN, AREA_N_OCEAN,
                       AREA_T_OCEAN + AREA_N_OCEAN + AREA_S_OCEAN,
                       AREA_S_OCEAN])

# Atmosphere dimensions
AREA_N_ATMOS = 2*np.pi*(1-np.sin(LATITUDE_N))*EARTH_RADIUS**2
AREA_S_ATMOS = AREA_N_ATMOS
AREA_GLOBE = 4.0*np.pi*EARTH_RADIUS**2

AREA_ATMOS = np.array([AREA_GLOBE - (AREA_N_ATMOS + AREA_S_ATMOS),
                       AREA_N_ATMOS, AREA_S_ATMOS])

DEPTH_ATMOS = 1e4
VOLUME_ATMOS = AREA_ATMOS*DEPTH_ATMOS
MASS_ATMOS = VOLUME_ATMOS*RHO_ATMOS
N_ATMOS_BOUNDARY = DEPTH_ATMOS*2*np.pi*EARTH_RADIUS*np.cos(LATITUDE_N)
S_ATMOS_BOUNDARY = N_ATMOS_BOUNDARY
ATMOS_INDS = np.array([0, 1, 3])