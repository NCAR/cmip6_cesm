import xarray as xr
import numpy as np
import dask
import functools
from distributed import Client

client = Client()

def read_file_nc(file):
    ds = xr.open_dataset(file,decode_times=False,chunks={'time':1,'z_t':1})
    return ds

def calc_pressure(ds):
    pressure = 0.059808 * (np.exp(-0.025 * ds.z_t) - 1.0) + 0.100766 * ds.z_t + 2.28405e-7 * (ds.z_t**2.0)
    return pressure

def calc_eos_num (pressure, salinity, temperature):
    #constants
    mwjfnp0s0t0 =  9.99843699e+2
    mwjfnp0s0t1 =  7.35212840
    mwjfnp0s0t2 = -5.45928211e-2
    mwjfnp0s0t3 =  3.98476704e-4
    mwjfnp0s1t0 =  2.96938239
    mwjfnp0s1t1 = -7.23268813e-3
    mwjfnp0s2t0 =  2.12382341e-3
    mwjfnp1s0t0 =  1.04004591e-2
    mwjfnp1s0t2 =  1.03970529e-7
    mwjfnp1s1t0 =  5.18761880e-6
    mwjfnp2s0t0 = -3.24041825e-8
    mwjfnp2s0t2 = -1.23869360e-11
    
    ## *** first calculate numerator of MWJF density [P_1(S,T,p)]
    mwjfnums0t0 = mwjfnp0s0t0 + pressure * (mwjfnp1s0t0 + pressure * mwjfnp2s0t0)
    mwjfnums0t1 = mwjfnp0s0t1
    mwjfnums0t2 = mwjfnp0s0t2 + pressure * (mwjfnp1s0t2 + pressure * mwjfnp2s0t2)
    mwjfnums0t3 = mwjfnp0s0t3
    mwjfnums1t0 = mwjfnp0s1t0 + pressure * mwjfnp1s1t0
    mwjfnums1t1 = mwjfnp0s1t1
    mwjfnums2t0 = mwjfnp0s2t0

    numerator = mwjfnums0t0 + temperature  *  (mwjfnums0t1 + temperature  *  (mwjfnums0t2 + \
    mwjfnums0t3  *  temperature)) + salinity  *  (mwjfnums1t0 +              \
    mwjfnums1t1  *  temperature + mwjfnums2t0  *  salinity)
    
    return numerator

def calc_eos_den(pressure,salinity,temperature):
    #constants
    mwjfdp0s0t0 =  1.0
    mwjfdp0s0t1 =  7.28606739e-3
    mwjfdp0s0t2 = -4.60835542e-5
    mwjfdp0s0t3 =  3.68390573e-7
    mwjfdp0s0t4 =  1.80809186e-10
    mwjfdp0s1t0 =  2.14691708e-3
    mwjfdp0s1t1 = -9.27062484e-6
    mwjfdp0s1t3 = -1.78343643e-10
    mwjfdp0sqt0 =  4.76534122e-6
    mwjfdp0sqt2 =  1.63410736e-9
    mwjfdp1s0t0 =  5.30848875e-6
    mwjfdp2s0t3 = -3.03175128e-16
    mwjfdp3s0t1 = -1.27934137e-17
    
    salinity_sqroot = np.sqrt(salinity)
    
    ## *** now calculate denominator of MWJF density [P_2(S,T,p)]
    mwjfdens0t0 = mwjfdp0s0t0 + pressure * mwjfdp1s0t0
    mwjfdens0t1 = mwjfdp0s0t1 + (pressure ** 3)  *  mwjfdp3s0t1
    mwjfdens0t2 = mwjfdp0s0t2
    mwjfdens0t3 = mwjfdp0s0t3 + (pressure ** 2)  *  mwjfdp2s0t3
    mwjfdens0t4 = mwjfdp0s0t4
    mwjfdens1t0 = mwjfdp0s1t0
    mwjfdens1t1 = mwjfdp0s1t1
    mwjfdens1t3 = mwjfdp0s1t3
    mwjfdensqt0 = mwjfdp0sqt0
    mwjfdensqt2 = mwjfdp0sqt2
    
    denomenator = mwjfdens0t0 + temperature  *  (mwjfdens0t1 + temperature  *  (mwjfdens0t2 +       \
    temperature  *  (mwjfdens0t3 + mwjfdens0t4  *  temperature))) +                   \
    salinity  *  (mwjfdens1t0 + temperature  *  (mwjfdens1t1 + temperature * temperature * mwjfdens1t3)+ \
    salinity_sqroot  *  (mwjfdensqt0 + temperature * temperature * mwjfdensqt2))
    
    return denomenator

def eos(ds_salt, ds_temp):
    '''
    compute density as a function of
      SALT (salinity, psu),
      TEMP (potential temperature, degree C) and
      DEPTH (meters).
      PRESS_IN:  logical: DEPTH is passed in as dbar

      output RHOFULL is in kg/m**3

     McDougall, Wright, Jackett, and Feistel EOS
     test value : rho = 1.033213387 for
     S = 35.0 PSU, theta = 20.0 C, pressure = 2000.0 dbars

     Julia Kent: Daskified EOS code takes the salt and temeperature time sequences as xarray datasets

    '''

    ## type conversion
    #typeRHO    = typeof(RHOFULL)
    #copy_VarAtts(TEMP,RHOFULL)
    ##copy_VarAtts(TEMP,DRHODT)
    ##copy_VarAtts(TEMP,DRHODS)
    #RHOFULL@long_name = "\rho_{sw}"
    #RHOFULL@units      = "kg m**{-3}"

    ##DRHODT@long_name = "Thermal expansion coefficient"
    ##DRHODT@units     = "kg m**{-3} **{\circ}C**{-1}"
    ##DRHODS@long_name = "Haline contraction coefficient"
    ##DRHODS@units     = "kg m**{-3} (psu)**{-1}"

    #------------------------------------------------------------
    # enforce min/max values
    #------------------------------------------------------------

    tmin =  -2.0 ; tmax = 999.0
    smin =   0.0 ; smax = 999.0

    check_temp_vals=lambda x: x if x>tmin and x<tmax else np.nan
    check_temp_vals_vec=np.vectorize(check_temp_vals)

    check_salinity_vals=lambda x: x if x>smin and x<smax else np.nan
    check_salinity_vals_vec=np.vectorize(check_salinity_vals)

    temperature = xr.apply_ufunc(check_temp_vals_vec,ds_temp.TEMP,dask='parallelized',output_dtypes=[ds_temp.TEMP.dtype])
    salinity = xr.apply_ufunc(check_salinity_vals_vec,ds_salt.SALT,dask='parallelized',output_dtypes=[ds_salt.SALT.dtype])

    #------------------------------------------------------------
    # compute pressure
    #------------------------------------------------------------
    
    pressure = calc_pressure(ds_salt)

    #------------------------------------------------------------
    # compute density
    #------------------------------------------------------------
    numerator = calc_eos_num(pressure,salinity,temperature)
    denomenator = calc_eos_den(pressure,salinity,temperature)

    RHOFULL = numerator/denomenator

    return RHOFULL