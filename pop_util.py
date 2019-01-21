'''Utilities for analysis of the POP ocean model.'''
from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import yaml

import logging

import numpy as np
import xarray as xr

import esmlab.statistics as gt

def fix_encoding(da, orig_encoding):
    CP_ENCODING = ['_FillValue', 'missing_value', 'dtype']
    new_encoding = {k: v for k, v in orig_encoding.items() if k in CP_ENCODING}
    if '_FillValue' not in orig_encoding:
        new_encoding['_FillValue'] = None
        new_encoding['missing_value'] = None
    if da.dtype == 'int64':
        new_encoding['dtype'] = 'int32'

    da.encoding = new_encoding
    return da


def region_mask_diagnostics(pop_hist_ds):
    TLAT = pop_hist_ds.TLAT
    TLONG = pop_hist_ds.TLONG
    KMT = pop_hist_ds.KMT
    REGION_MASK = pop_hist_ds.REGION_MASK

    M = xr.DataArray(np.ones(REGION_MASK.shape),dims=('nlat','nlon'))
    region_defs = OrderedDict([
        ('Global', M.where(REGION_MASK > 0) ),
        ('90S-44S', M.where((-90.0 < TLAT) & (TLAT <= -44.0)) ),
        ('Atl_44S-18S', M.where((-44.0 < TLAT) & (TLAT <= -18.0) & ((292.0 <= TLONG) | (TLONG < 20.0))) ),
        ('Pac_44S-18S', M.where((-44.0 < TLAT) & (TLAT <= -18.0) & ((147.0 <= TLONG) & (TLONG < 292.0))) ),
        ('Ind_44S-18S', M.where((-44.0 < TLAT) & (TLAT <= -18.0) & ((20.0  <= TLONG) & (TLONG < 147.0))) ),
        ('Atl_18S-18N', M.where((-18.0 < TLAT) & (TLAT <=  18.0) & (REGION_MASK == 6)) ),
        ('Pac_18S-18N', M.where((-18.0 < TLAT) & (TLAT <=  18.0) & (REGION_MASK == 2)) ),
        ('Ind_18S-30N', M.where((-18.0 < TLAT) & (TLAT <=  30.0) & (REGION_MASK == 3)) ),
        ('Atl_18N-49N', M.where(( 18.0 < TLAT) & (TLAT <=  49.0) & (REGION_MASK == 6)) ),
        ('Pac_18N-49N', M.where(( 18.0 < TLAT) & (TLAT <=  49.0) & (REGION_MASK == 2)) ),
        ('Atl_49N-66N', M.where(( 49.0 < TLAT) & (TLAT <=  66.0) & (REGION_MASK >= 6) & (REGION_MASK <= 10)) ),
        ('Pac_49N-66N', M.where(( 49.0 < TLAT) & (TLAT <=  66.0) & (REGION_MASK == 2)) ),
        ('Arc_66N-90N', M.where(( 66.0 < TLAT) & (REGION_MASK >= 2) & (REGION_MASK <= 10)) )])

    mask3d = xr.DataArray(np.zeros(((len(region_defs),)+REGION_MASK.shape)),
                        dims=('region','nlat','nlon'),
                        coords={'region':list(region_defs.keys())})

    for i,mask_logic in enumerate(region_defs.values()):
        mask3d.values[i,:,:] = mask_logic.fillna(0.)
    mask3d = mask3d.where(KMT>0)

    return mask3d

def flux_weighted_area_sum_convert_units(da,category):
    attrs = da.attrs.copy()
    encoding = da.encoding

    if category == 'Cflux':
        #-- nmolC/cm^2/s to PgC/y
        da = da * (1.0e-15*12.0*1.0e-9)*(365.0*86400.0)
        attrs['units'] = 'Pg C yr$^{-1}$'

    elif category == 'Nflux':
        #-- nmolN/cm^2/s to TgN/y
        da = da * (1.0e-12*14.0*1.0e-9)*(365.0*86400.0)
        attrs['units'] = 'Tg N yr$^{-1}$'

    elif category == 'Siflux':
        #-- nmolSi/cm^2/s to TmolSi/y
        da = da * (1.0e-12*1.0e-9)*(365.0*86400.0)
        attrs['units'] = 'Tmol Si yr$^{-1}$'

    elif category == 'Feflux':
        #-- mmolFe/m^2/s to GmolFe/y
        da = da * (1.0e-16)*(365.0*86400.0)
        attrs['units'] = 'Gmol Fe yr$^{-1}$'
    else:
        raise ValueError(f'Category unknown: {category}')

    da.attrs = attrs
    da = fix_encoding(da, encoding)

    return da

def derived_var(varname, ds):
    '''Return DataArray for derived variable.'''

    if varname == 'ACC':
        return acc_from_bsf_gx1v7(ds)[varname]
    elif varname == 'sigma_theta':
        return derive_var_sigma_theta_from_PD(ds)[varname]
    elif varname == 'pCFC11':
        return derive_var_pCFC12(ds)[varname]
    elif varname == 'pCFC12':
        return derive_var_pCFC12(ds)[varname]
    elif varname == 'MLD':
        return derive_var_MLD(ds)[varname]
    else:
        raise ValueError('derived_var: Unknown varname: {0}'.format(varname))


def require_variables(ds, req_var):
    missing_var_error = False
    for v in req_var:
        if v not in ds:
            logging.error('ERROR: Missing required variable: %s'%v)
            missing_var_error = True
    if missing_var_error:
        raise ValueError('Variables missing')

def derive_var_pCFC11(ds, drop_derivedfrom_vars=True):
    from solubility import calc_cfc11sol

    require_variables(ds,['CFC11','TEMP','SALT'])

    ds['pCFC11'] = ds['CFC11'] * 1e-9 / calc_cfc11sol(ds.SALT,ds.TEMP)
    ds.pCFC11.attrs['long_name'] = 'pCFC-11'
    ds.pCFC11.attrs['units'] = 'patm'
    if 'coordinates' in ds.TEMP.attrs:
        ds.pCFC11.attrs['coordinates'] = ds.TEMP.attrs['coordinates']
    ds.pCFC11.encoding = ds.TEMP.encoding

    if drop_derivedfrom_vars:
        ds = ds.drop(['CFC11','TEMP','SALT'])

    return ds

def derive_var_pCFC12(ds, drop_derivedfrom_vars=True):
    from solubility import calc_cfc12sol

    require_variables(ds,['CFC12','TEMP','SALT'])

    ds['pCFC12'] = ds['CFC12'] * 1e-9 / calc_cfc12sol(ds['SALT'],ds['TEMP'])
    ds.pCFC12.attrs['long_name'] = 'pCFC-12'
    ds.pCFC12.attrs['units'] = 'patm'
    if 'coordinates' in ds.TEMP.attrs:
        ds.pCFC12.attrs['coordinates'] = ds.TEMP.attrs['coordinates']
    ds.pCFC12.encoding = ds.TEMP.encoding

    if drop_derivedfrom_vars:
        ds = ds.drop(['CFC12','TEMP','SALT'])

    return ds

def derive_var_sigma_theta_from_PD(ds, drop_derivedfrom_vars=True):
    require_variables(ds,['PD'])

    attrs = ds.PD.attrs.copy()
    encoding = ds.PD.encoding

    ds['sigma_theta'] = (ds.PD - 1.) * 1000.

    ds.sigma_theta.attrs = attrs
    ds.sigma_theta.attrs['long_name'] = '$\sigma_{theta}$'
    ds.sigma_theta.attrs['units'] = 'kg m$^{-3}'
    ds.sigma_theta.encoding = encoding

    if drop_derivedfrom_vars:
        ds = ds.drop(['PD'])

    return ds

def derive_var_MLD(ds, Dsigma=0.03, use_PD = False):
    '''Return the mixed layer depth computed from potential density criterion.
    '''
    import popeos
    require_variables(ds,['TEMP','SALT','z_t','KMT'])

    attrs = ds.TEMP.attrs.copy()
    encoding = ds.TEMP.encoding

    if use_PD:
        RHO = (ds.PD - 1.) * 1000.
    else:
        RHO = popeos.eos(SALT=ds.SALT,TEMP=ds.TEMP,DEPTH=0.,PRESS_IN=False)
        RHO = RHO - 1000.

    KMT = ds.KMT.values.astype(int)
    z_t = ds.z_t.values * 1e-2 # convert from cm to m

    nl, nk, nj, ni = ds.SALT.shape
    MLD = np.ones((nl, nj, ni)) * np.nan

    for j in range(0,nj):
        for i in range(0,ni):
            kmt = KMT[j,i]
            if kmt == 0: continue
            for l in range(0,nl):
                rho = RHO[l,0:kmt,j,i]
                z = z_t[0:kmt]

                rho, I = np.unique(rho, return_index=True)
                z = z[I]
                if len(I) >= 3:
                    MLD[l,j,i] = np.interp(rho[0]+Dsigma, rho, z)

    ds = ds.drop(['TEMP','SALT'])
    ds = ds.load()
    ds['MLD'] = xr.DataArray(MLD,
                             dims=('time','nlat','nlon'),
                             coords={'time':ds.time},
                             attrs=attrs,
                             encoding=encoding)

    ds.MLD.attrs['long_name'] = 'MLD'
    ds.MLD.attrs['definition'] = f'$\Delta\sigma = {Dsigma:0.3f}$'
    ds.MLD.attrs['units'] = 'm'

    return ds

def acc_from_bsf_gx1v7(ds):
    require_variables(ds,['BSF'])

    BSF = ds.BSF

    if BSF.shape[1] != 384 or BSF.shape[2] != 320:
        raise ValueError('acc_from_bsf_gx1v7: Unexpected dims for BSF:\n{0}'.format(BSF))
    if BSF.name != 'BSF':
        raise ValueError('acc_from_bsf_gx1v7: Unexpected DataArray input:\n{0}'.format(BSF))

    attrs = BSF.attrs.copy()
    encoding = BSF.encoding

    i_antarc = 291
    j_antarc = 19
    i_s_amer = 291
    j_s_amer = 47

    ds['ACC'] = BSF[:,j_antarc,i_antarc] - BSF[:,j_s_amer,i_s_amer]

    for att in ['coordinates']:
        if att in attrs:
            del attrs[att]

    ds.ACC.attrs = attrs
    ds.ACC.attrs['long_name'] = 'ACC transport'
    ds.ACC.attrs['units'] = 'Sv'

    ds.ACC.encoding = fix_encoding(ds.ACC,encoding)

    return ds


def make_masked_area_and_vol(ds, rmask3d, ignore_ssh=True):

    nk = len(ds.z_t)
    nj = ds.KMT.shape[0]
    ni = ds.KMT.shape[1]

    #-- make 3D array of 0:km
    MASK = (xr.DataArray(np.arange(0,len(ds.z_t)),dims=('z_t')) *
            xr.DataArray(np.ones((nk,nj,ni)),dims=('z_t','nlat','nlon'),
                         coords={'z_t':ds.z_t}))
    MASK = MASK.where(MASK <= ds.KMT-1)
    MASK.values = np.where(MASK.notnull(),1.,0.)

    if not ignore_ssh:
        raise ValueError('SSH addition not implemented yet')
        if 'SSH' not in ds:
            raise ValueError('Require SSH to compute inventories: not found.')
        MASKED_SSH_VOL = ds.SSH * ds.TAREA

    MASKED_VOL = ds.dz * ds.TAREA * MASK * rmask3d
    MASKED_VOL.attrs['units'] = 'cm^3'
    MASKED_VOL.attrs['long_name'] = 'masked volume'

    MASKED_AREA = MASK * ds.TAREA * rmask3d
    MASKED_AREA.attrs['units'] = 'cm^2'
    MASKED_AREA.attrs['long_name'] = 'masked area'

    return MASKED_AREA, MASKED_VOL

def make_regional_timeseries(ds):

    with open('pop_variable_defs.yml') as f:
        pop_variable_defs = yaml.load(f)

    # TODO: MASKED_AREA, MASKED_VOL could be precomputed and passed in...
    rmask3d = region_mask_diagnostics(ds.isel(time=0))
    MASKED_AREA, MASKED_VOL = make_masked_area_and_vol(ds.isel(time=0), rmask3d, ignore_ssh=True)

    dso = xr.Dataset()

    # copy time bounds
    if 'bounds' in ds['time'].attrs:
        tb_name = ds['time'].attrs['bounds']
    elif 'time_bounds' in ds:
        tb_name = 'time_bounds'

    dso[tb_name] = ds[tb_name]

    dz_150m = ds.dz[0:15].rename({'z_t': 'z_t_150m'})

    variable_list = [v for v in ds.variables if 'time' in ds[v].dims]

    for varname in variable_list:

        if varname not in pop_variable_defs:
            logging.warning(f'Missing definition of variable {varname}')
            continue

        info = pop_variable_defs[varname]

        #-- get the variable
        klevel = 0

        #-- variable is a derived sum
        if 'derived_sum' in info:
            var = 0.
            for v in info['derived_sum']:
                var += ds[v]
            var.name = varname
            var.encoding = ds[info['derived_sum'][0]].encoding

        #-- variable is derived by indexing "z_t"
        elif 'derived_zindex' in info:
            varname_from = info['derived_zindex'][0]
            zlevel = info['derived_zindex'][1] * 100.
            klevel = np.abs(ds.z_t.values-zlevel).argmin()
            var = ds[varname_from].isel(z_t=klevel).drop(['z_t'])
            var.name = varname

        #-- variable is derived via "derived_var"
        elif 'derived' in info:
            var = derived_var(varname, ds)

        #-- variable is read directly from file
        else:
            var = ds[varname]

        #-- perform operations
        if 'operations' in info:
            for op in info['operations']:
                if op == 'vertical_integral':
                    if var.dims[1] == 'z_t_150m':
                        var = gt.weighted_sum(var, weights=dz_150m,
                                              dim='z_t_150m')
                    elif var.dims[1] == 'z_t':
                        var = gt.weighted_sum(var, weights=ds.dz,
                                              dim='z_t')
                else:
                    raise ValueError('Unknown operation: {0}'.format(op))

        #-- compute averages
        if info['category'] == 'tracer':
            # add global inventory
            dso[varname+'_avg'] = gt.weighted_mean(var,
                                                   weights=MASKED_VOL.isel(region=0,drop=True),
                                                   dim=['z_t','nlat','nlon'],
                                                   apply_nan_mask=False)
            dso[varname] = gt.weighted_mean(var,
                                            weights = MASKED_AREA,
                                            dim=['nlat','nlon'],
                                            apply_nan_mask=False)

        elif info['category'] in ['Cflux','Nflux','Siflux','Feflux']:
            var = gt.weighted_sum(var,
                                  weights = MASKED_AREA.isel(z_t=klevel).drop(['z_t']),
                                  dim=['nlat','nlon'])
            dso[varname] = flux_weighted_area_sum_convert_units(var,info['category'])

        elif info['category'] in ['diagave']:
            dso[varname] = gt.weighted_mean(var,
                                            weights = MASKED_AREA.isel(z_t=klevel).drop(['z_t']),
                                            dim=['nlat','nlon'],
                                            apply_nan_mask=False)

        elif info['category'] in ['transport']:
            dso[varname] = var

        else:
            raise ValueError('Unknown variable category {0}'.format(info['category']))

    return dso
