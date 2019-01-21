import numpy as np

def pressure(depth_in):
  ''' convert depth in meters to pressure in bars'''
  depth = np.double(depth_in)
  return 0.059808 * ( np.exp( -0.025 * depth ) - 1.0 ) \
  + 0.100766 * depth + 2.28405e-7 * ( depth ** 2.0 )


def eos(SALT,TEMP,DEPTH,PRESS_IN=False):
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
    # MWJF EOS coefficients
    #------------------------------------------------------------

    ## *** these constants will be used to construct the numerator
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

    ## *** these constants will be used to construct the denominator
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

    #------------------------------------------------------------
    # enforce min/max values
    #------------------------------------------------------------

    tmin =  -2.0
    tmax = 999.0
    smin =   0.0
    smax = 999.0

    old_err_settings = np.seterr(invalid='ignore')
    
    TQ = np.double(TEMP)
    TQ = np.where(TQ < tmin, tmin, TQ)
    TQ = np.where(TQ > tmax, tmax, TQ)

    SQ = np.double(SALT)
    SQ = np.where(SQ < smin, smin, SQ)
    SQ = np.where(SQ > smax, smax, SQ)

    np.seterr(**old_err_settings)

    #------------------------------------------------------------
    # compute pressure
    #------------------------------------------------------------
    if not PRESS_IN:
        p = 10.0 * pressure(DEPTH) # dbar
    else:
        p = DEPTH ## dbar

    #------------------------------------------------------------
    # compute density
    #------------------------------------------------------------
    SQR = np.sqrt(SQ)

    ## *** first calculate numerator of MWJF density [P_1(S,T,p)]
    mwjfnums0t0 = mwjfnp0s0t0 + p * (mwjfnp1s0t0 + p * mwjfnp2s0t0)
    mwjfnums0t1 = mwjfnp0s0t1
    mwjfnums0t2 = mwjfnp0s0t2 + p * (mwjfnp1s0t2 + p * mwjfnp2s0t2)
    mwjfnums0t3 = mwjfnp0s0t3
    mwjfnums1t0 = mwjfnp0s1t0 + p * mwjfnp1s1t0
    mwjfnums1t1 = mwjfnp0s1t1
    mwjfnums2t0 = mwjfnp0s2t0

    WORK1 = mwjfnums0t0 + TQ  *  (mwjfnums0t1 + TQ  *  (mwjfnums0t2 + \
    mwjfnums0t3  *  TQ)) + SQ  *  (mwjfnums1t0 +              \
    mwjfnums1t1  *  TQ + mwjfnums2t0  *  SQ)

    ## *** now calculate denominator of MWJF density [P_2(S,T,p)]
    mwjfdens0t0 = mwjfdp0s0t0 + p * mwjfdp1s0t0
    mwjfdens0t1 = mwjfdp0s0t1 + (p ** 3)  *  mwjfdp3s0t1
    mwjfdens0t2 = mwjfdp0s0t2
    mwjfdens0t3 = mwjfdp0s0t3 + (p ** 2)  *  mwjfdp2s0t3
    mwjfdens0t4 = mwjfdp0s0t4
    mwjfdens1t0 = mwjfdp0s1t0
    mwjfdens1t1 = mwjfdp0s1t1
    mwjfdens1t3 = mwjfdp0s1t3
    mwjfdensqt0 = mwjfdp0sqt0
    mwjfdensqt2 = mwjfdp0sqt2

    WORK2 = mwjfdens0t0 + TQ  *  (mwjfdens0t1 + TQ  *  (mwjfdens0t2 +       \
    TQ  *  (mwjfdens0t3 + mwjfdens0t4  *  TQ))) +                   \
    SQ  *  (mwjfdens1t0 + TQ  *  (mwjfdens1t1 + TQ * TQ * mwjfdens1t3)+ \
    SQR  *  (mwjfdensqt0 + TQ * TQ * mwjfdensqt2))

    DENOMK = 1.0 / WORK2

    RHOFULL = WORK1  *  DENOMK

    return RHOFULL

    #------------------------------------------------------------
    # dRHOdS/dRHOdT
    #------------------------------------------------------------
    if False:
        ## dRHOdT
        WORK3 = mwjfnums0t1 + TQ  *  (2.0 * mwjfnums0t2 +           \
        3.0 * mwjfnums0t3  *  TQ) + mwjfnums1t1  *  SQ

        WORK4 = mwjfdens0t1 + SQ  *  mwjfdens1t1 +                     \
        TQ  *  (2.0 * (mwjfdens0t2 + SQ * SQR * mwjfdensqt2) +      \
        TQ  *  (3.0 * (mwjfdens0t3 + SQ  *  mwjfdens1t3) +          \
        TQ  *   4.0 * mwjfdens0t4))

        DRHODT = (WORK3 - WORK1 * DENOMK * WORK4) * DENOMK

        ## dRHOdS
        WORK3 = mwjfnums1t0 + mwjfnums1t1  *  TQ + 2.0 * mwjfnums2t0  *  SQ

        WORK4 = mwjfdens1t0 +                                   \
        TQ  *  (mwjfdens1t1 + TQ * TQ * mwjfdens1t3) +          \
        1.5 * SQR * (mwjfdensqt0 + TQ * TQ * mwjfdensqt2)

        DRHODS = (WORK3 - WORK1 * DENOMK * WORK4) * DENOMK  *  1000.0
