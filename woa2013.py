import os
from subprocess import call
import xarray as xr
import numpy as np

woapth = '/glade/work/mclong/woa2013v2'
woa_names = {'T':'t',
             'S':'s',
             'TEMP':'t',
             'SALT':'s',
             'NO3':'n',
             'O2':'o',
             'O2sat':'O',
             'AOU':'A',
             'SiO3':'i',
             'PO4':'p'}

mlperl_2_mmolm3 = 1.e6 / 1.e3 / 22.3916

long_names = {'NO3':'Nitrate',
              'O2':'Oxygen',
              'O2sat':'Oxygen saturation',
              'AOU':'AOU',
              'SiO3':'Silicic acid',
              'PO4':'Phosphate',
              'S':'Salinity',
              'T':'Temperature',
              'SALT':'Salinity',
              'TEMP':'Temperature',
             }

#----------------------------------------------------
#--- function
#----------------------------------------------------

def time_freq_code_code(freq):
    # 13: jfm, 14: amp, 15: jas, 16: ond

    if freq == 'ann':
        return ['00']
    elif freq == 'mon':
        return [f'{m:02d}' for m in range(1,13)]
    elif freq == 'jfm':
        return ['13']
    elif freq == 'amp':
        return ['14']
    elif freq == 'jas':
        return ['15']
    elif freq == 'ond':
        return ['16']

#----------------------------------------------------
#--- function
#----------------------------------------------------

def get_file_list(varname,freq='ann',grid='1x1d'):
      
    v = woa_names[varname]
    
    if grid == '1x1d':
        res_code = '01'
    
    elif grid == '0.25x0.25d':
        res_code = '04'
        
    elif grid == 'POP_gx1v7':
        res_code = 'gx1v7'
    
    if v in ['t','s']:
        files = [f'woa13_decav_{v}{code}_{res_code}v2.nc'for code in time_freq_code_code(freq)]
    elif v in ['o','p','n','i','O','A']:
        files = [f'woa13_all_{v}{code}_{res_code}.nc' for code in time_freq_code_code(freq)]
    else:
        raise ValueError('no file template defined')
        
    return [os.path.join(woapth,grid,f) for f in files]

#----------------------------------------------------
#--- function
#----------------------------------------------------

def open_dataset(varname_list,freq='ann',grid='1x1d'):
    if not isinstance(varname_list,list):
        varname_list = [varname_list]       
    
    ds = xr.Dataset()    
    for varname in varname_list:
        v = woa_names[varname]
        
        files_in = get_file_list(varname,freq=freq,grid=grid)
        dsi = xr.open_mfdataset(files_in,decode_times=False)

        dsi.rename({f'{v}_an':varname},inplace=True)        
        dsi = dsi.drop([k for k in dsi.variables if f'{v}_' in k])
        
        
        if varname in ['O2','AOU','O2sat']:
            dsi[varname] = dsi[varname] * mlperl_2_mmolm3
            dsi[varname].attrs['units'] = 'mmol m$^{-3}$'
        
        if dsi[varname].attrs['units'] == 'micromoles_per_liter':
            dsi[varname].attrs['units'] = 'mmol m$^{-3}$'
        dsi[varname].attrs['long_name'] = long_names[varname]
        
        if ds.variables:
            ds = xr.merge((ds,dsi))
        else:
            ds = dsi
    return ds
