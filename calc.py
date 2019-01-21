#! /usr/bin/env python
from __future__ import absolute_import, division, print_function
import xarray as xr
import numpy as np
import cftime
from datetime import datetime

xr_open_ds = {'chunks' : {'time':1},
              'decode_coords' : False,
              'decode_times' : False,
              'data_vars' : 'minimal'}

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def _list_to_indexer(index_list):
    '''
    .. function:: _list_to_indexer(index_list)

    Convert string formatted as: dimname,start[,stop[,stride]]
    to index (for the case where only 'start' is provided)
    or indexing object (slice).

    :param index_list: index list as passed in from
                       --isel dimname:start,stop,stride

    :returns: dict -- {dimname: indexing object}
    '''

    if len(index_list) == 1:
        return index_list[0]
    elif len(index_list) == 2:
        return slice(index_list[0],index_list[1])
    elif len(index_list) == 3:
        return slice(index_list[0],index_list[1],index_list[2])
    else:
        raise ValueError('illformed dimension subset')


def time_bound_var(ds):
    tb_name = ''
    if 'bounds' in ds['time'].attrs:
        tb_name = ds['time'].attrs['bounds']
    elif 'time_bound' in ds:
        tb_name = 'time_bound'
    else:
        raise ValueError('No time_bound variable found')
    tb_dim = ds[tb_name].dims[1]
    return tb_name,tb_dim

def fix_time(ds, year_offset=None):

    tb_name,tb_dim = time_bound_var(ds)
    time_values = ds[tb_name].mean(tb_dim)

    if np.isnan(year_offset):
        year_offset = None

    if year_offset is not None:
        time_values += cftime.date2num(datetime(int(year_offset),1,1),
                                       ds.time.attrs['units'],
                                       ds.time.attrs['calendar'])

    time = cftime.num2date(time_values,
                           units = ds.time.attrs['units'],
                           calendar = ds.time.attrs['calendar'])

    ds.time.values = xr.CFTimeIndex(time)
    
    return ds

def unfix_time(ds):
    if ds.time.dtype == 'O':
        time = cftime.date2num(ds.time,
                               units = ds.time.attrs['units'],
                               calendar = ds.time.attrs['calendar'])
        ds.time.values = time
    return ds

def save_attrs(ds):
    '''Return dictionary of attrs for each variable in dataset.'''
    return {v:ds[v].attrs for v in ds.variables}

def save_encoding(ds):
    '''Return dictionary of encoding for each variable in dataset.'''
    return {v:{key:val for key,val in ds[v].encoding.items()}
            for v in ds.variables}

def fix_attrs(ds,attrs):
    for v in ds.variables:
        if v in attrs: ds[v].attrs = attrs[v]
    return ds

def fix_encoding(ds,orig_encoding):
    '''Reattach pre-saved encodings for each variable in a dataset,
       including some modifications to avoid putting _FillValue where it
       didn't previously exist and avoid using dtype = int64, which breaks some
       other tools using netCDF.
    '''
    copy_encodings = ['_FillValue', 'missing_value', 'dtype']

    new_encoding = {}
    for v in ds.variables:
        if v in orig_encoding:
            new_encoding_v = {key: val for key, val in orig_encoding[v].items()
                              if key in copy_encodings}
        else:
            new_encoding_v = {}

        if '_FillValue' not in orig_encoding[v]:
            new_encoding_v['_FillValue'] = None
            new_encoding_v['missing_value'] = None

        if ds[v].dtype == 'int64':
            new_encoding_v['dtype'] = 'int32'

        ds[v].encoding = new_encoding_v

    return ds

def compute_mon_climatology(dsm):
    '''Compute a monthly climatology'''

    tb_name,tb_dim = time_bound_var(dsm)

    static_vars = [v for v in dsm.variables if 'time' not in dsm[v].dims]
    variables = [v for v in dsm.variables if 'time' in dsm[v].dims and v not in ['time']]

    # save attrs
    attrs = save_attrs(dsm)
    encoding = save_encoding(dsm)

    #-- compute climatology
    ds = dsm.drop(static_vars).groupby('time.month').mean('time').rename({'month':'time'})

    #-- put static_vars back
    ds = xr.merge((ds,dsm.drop([v for v in dsm.variables if v not in static_vars])))

    ds['month'] = ds.time.copy()
    attrs['month'] = {'long_name':'Month','units':'month'}
    encoding['month'] = {'dtype':'int32', '_FillValue': None}

    ds['month_bounds'] = ds[tb_name] - ds[tb_name][0,0]
    ds.time.values = ds.month_bounds.mean(tb_dim).values
    encoding['month_bounds'] = {'dtype':'float', '_FillValue': None}
    ds = ds.drop(tb_name)
    attrs['time'] = {'long_name':'time',
                     'units':'days since 0001-01-01 00:00:00',
                     'calendar': attrs['time']['calendar'],
                     'bounds':'month_bounds'}
    del encoding['time']
    encoding['time'] = {'dtype':'float', '_FillValue': None}

    # put the attributes and encoding back
    ds = fix_attrs(ds,attrs)
    ds = fix_encoding(ds,encoding)

    return ds

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def compute_mon_anomaly(dsm):
    '''Compute a monthly anomaly'''

    tb_name,tb_dim = time_bound_var(dsm)

    static_vars = [v for v in dsm.variables if 'time' not in dsm[v].dims]
    variables = [v for v in dsm.variables if 'time' in dsm[v].dims and v not in ['time',tb_name]]

    # save attrs
    attrs = save_attrs(dsm)
    encoding = save_encoding(dsm)

    if len(dsm.time)%12 != 0:
        raise ValueError('Time axis not evenly divisible by 12!')

    #-- compute anomaly
    ds = dsm.drop(static_vars).groupby('time.month') - dsm.drop(static_vars).groupby('time.month').mean('time')
    ds.reset_coords('month',inplace=True)

    #-- put static_vars back
    ds = xr.merge((ds,dsm.drop([v for v in dsm.variables if v not in static_vars])))
    attrs['month'] = {'long_name':'Month'}

    # put the attributes and encoding back
    ds = fix_attrs(ds,attrs)
    ds = fix_encoding(ds,encoding)

    return ds

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def compute_ann_mean(dsm):
    '''Compute an annual mean'''

    tb_name,tb_dim = time_bound_var(dsm)

    static_vars = [v for v in dsm.variables if 'time' not in dsm[v].dims]
    variables = [v for v in dsm.variables if 'time' in dsm[v].dims and v not in ['time',tb_name]]

    # save attrs
    attrs = save_attrs(dsm)
    encoding = save_encoding(dsm)

    #-- compute weights
    dt = dsm[tb_name].diff(dim=tb_dim)[:,0]
    wgt = dt.groupby('time.year')/dt.groupby('time.year').sum()
    np.testing.assert_allclose(wgt.groupby('time.year').sum(),1.)

    # groupby.sum() does not seem to handle missing values correctly: yields 0 not nan
    # the groupby.mean() does return nans, so create a mask of valid values for each variable
    valid = {v : dsm[v].groupby('time.year').mean(dim='time').notnull().rename({'year':'time'}) for v in variables}
    ones = dsm.drop(static_vars).where(dsm.isnull()).fillna(1.).where(dsm.notnull()).fillna(0.)

    # compute the annual means
    ds = (dsm.drop(static_vars) * wgt).groupby('time.year').sum('time').rename({'year':'time'},inplace=True)
    ones_out = (ones * wgt).groupby('time.year').sum('time').rename({'year':'time'},inplace=True)
    ones_out = ones_out.where(ones_out>0.)

    # renormalize to appropriately account for missing values
    ds = ds / ones_out

    # put the grid variables back
    ds = xr.merge((ds,dsm.drop([v for v in dsm.variables if v not in static_vars])))

    # apply the valid-values mask
    for v in variables:
        ds[v] = ds[v].where(valid[v])

    # reset time attrs
    ds.time.attrs = {'long_name': 'year'}
    ds = ds.drop(tb_name)

    # put the attributes and encoding back
    ds = fix_attrs(ds,attrs)
    ds = fix_encoding(ds,encoding)

    return ds

#-------------------------------------------------------------------------------
#-- main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    from subprocess import call

    import argparse
    import sys

    #---------------------------------------------------------------------------
    #-- parse args
    p = argparse.ArgumentParser(description='Process timeseries files.')

    p.add_argument('file_in',
                   type = lambda kv: kv.split(','))

    p.add_argument('file_out',
                   type = str)
    p.add_argument('--op', dest = 'operation',
                   required = True,
                   help = 'Specify operation')

    p.add_argument('-v', dest = 'variable_list',
                   default = [],
                   required = False,
                   help = 'variable list')

    p.add_argument('-x', dest = 'invert_var_selction',
                   action = 'store_true',
                   required = False,
                   help = 'invert variable list')

    p.add_argument('-O', dest = 'overwrite',
                   required = False,
                   action ='store_true',
                   help = 'overwrite')

    p.add_argument('--verbose', dest = 'verbose',
                   required = False,
                   action ='store_true',
                   help = 'Verbose')

    p.add_argument('--isel', dest = 'isel',
                   required = False,
                   default=[],
                   action = 'append',
                   help = 'subsetting mechanism "isel"')

    p.add_argument('--pbs-cluster', dest = 'pbs_cluster',
                   required = False,
                   action ='store_true',
                   help = 'do PBS cluster')

    p.add_argument('--pbs-spec', dest = 'pbs_spec',
                   type = lambda csv: {kv.split('=')[0]:kv.split('=')[1] for kv in csv.split(',')},
                   required = False,
                   default = {},
                   help = 'PBS cluster specifications')

    args = p.parse_args()

    #-- if the user has supplied a spec, assume pbs_cluster=True
    if args.pbs_spec:
        args.pbs_cluster = True

    #---------------------------------------------------------------------------
    #-- check output file existence and intentional overwrite
    if os.path.exists(args.file_out):
        if args.overwrite:
            call(['rm','-rfv',args.file_out])
        else:
            raise ValueError(f'{args.file_out} exists.  Use -O to overwrite.')

    #---------------------------------------------------------------------------
    #-- determine output format
    ext = os.path.splitext(args.file_out)[1]
    if ext == '.nc':
        write_output = lambda ds: ds.to_netcdf(args.file_out,unlimited_dims=['time'])
    elif ext == '.zarr':
        write_output = lambda ds: ds.to_zarr(args.file_out)
    else:
        raise ValueError('Unknown output file extension: {ext}')

    #---------------------------------------------------------------------------
    #-- set the operator
    if args.operation == 'annmean':
        operator = compute_ann_mean

    elif args.operation == 'monclim':
        operator = compute_mon_climatology

    elif args.operation == 'monanom':
        operator = compute_mon_anomaly
    else:
        raise ValueError(f'Unknown operation {args.operation}')

    #---------------------------------------------------------------------------
    #-- parse index
    isel = {}
    for dim_index in args.isel:
        dim = dim_index.split(',')[0]
        isel[dim] = _list_to_indexer([int(i) for i in dim_index.split(',')[1:]])

    #---------------------------------------------------------------------------
    #-- report args
    if args.verbose:
        print(f'\n{__file__}')
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        print()

    #---------------------------------------------------------------------------
    #-- spin up dask cluster?
    if args.pbs_cluster:
        queue = args.pbs_spec.pop('queue','regular')
        project = args.pbs_spec.pop('project','NCGD0011')
        walltime = args.pbs_spec.pop('walltime','04:00:00')
        n_nodes = args.pbs_spec.pop('n_nodes',4)

        if args.pbs_spec:
            raise ValueError(f'Unknown fields in pbs_spec: {args.pbs_spec.keys()}')

        from dask.distributed import Client
        from dask_jobqueue import PBSCluster

        USER = os.environ['USER']

        cluster = PBSCluster(queue = queue,
                             cores = 36,
                             processes = 9,
                             memory = '100GB',
                             project = project,
                             walltime = walltime,
                             local_directory=f'/glade/scratch/{USER}/dask-tmp')
        client = Client(cluster)
        cluster.scale(9*n_nodes)

    #---------------------------------------------------------------------------
    #-- read the input dataset
    ds = xr.open_mfdataset(args.file_in,**xr_open_ds)
    if isel:
        ds = ds.isel(**isel)

    if args.variable_list:
        if args.invert_var_selction:
            drop_vars = [v for v in ds.variables if v in args.variable_list]
        else:
            drop_vars = [v for v in ds.variables if v not in args.variable_list]
        ds = ds.drop(drop_vars)

    if args.verbose:
        print('\ninput dateset:')
        ds.info()

    #---------------------------------------------------------------------------
    #-- compute
    if args.verbose:
        print(f'\ncomputing {args.operation}')

    dso = operator(ds)

    if args.verbose:
        print('\noutput dateset:')
        dso.info()

    #---------------------------------------------------------------------------
    #-- write output
    if args.verbose:
        print(f'\nwriting {args.file_out}')

    write_output(dso)

    #---------------------------------------------------------------------------
    #-- wrap up
    if args.pbs_cluster:
        cluster.close()

    if args.verbose:
        print('\ndone.')
