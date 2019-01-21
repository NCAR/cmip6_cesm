from __future__ import absolute_import, division, print_function

import os
from subprocess import check_call
import logging
import importlib

import tempfile

import yaml

from datetime import datetime

import numpy as np
import dask
import xarray as xr
import cftime

import calc
import data_catalog

#-- settings (move to config.yml or similar)
USER = os.environ['USER']
file_format = 'zarr'

project_name = os.path.basename(os.getcwd())
dirout = f'/glade/scratch/{USER}/calcs/{project_name}'
if not os.path.exists(dirout):
    check_call(['mkdir','-p',dirout])

tmpdir = f'{dirout}/work'
if not os.path.exists(tmpdir):
    check_call(['mkdir','-p',tmpdir])

logging.basicConfig(level=logging.INFO)

#-------------------------------------------------------------------------------
#-- methods
#-------------------------------------------------------------------------------

def pop_calc_zonal_mean(file_in):
    '''
    compute zonal mean of POP field
    in lieau of wrapping klindsay's zon_avg program so as to operate on
    an `xarray` dataset: write to file, compute, read back.
    '''
    za = '/glade/u/home/klindsay/bin/za'

    fid,file_out = tempfile.mkstemp(dir=tmpdir,
                                    prefix='za-',
                                    suffix='.nc')

    rmask_file = '/glade/work/mclong/grids/PacAtlInd_REGION_MASK_gx1v6.nc'
    #print(' '.join([za,'-O','-rmask_file',rmask_file,'-o',file_out,file_in]))
    check_call([za,'-O','-rmask_file',rmask_file,'-o',file_out,file_in])

    return file_out

class derived_var_definition():
    def __init__(self,methods,vars_dependent,methods_kwargs=None):

        if not isinstance(methods,list):
            methods = [methods]

        self.methods = methods
        self.vars_dependent = vars_dependent
        if methods_kwargs is None:
            methods_kwargs = [{} for m in methods]
        self.methods_kwargs = methods_kwargs


class yaml_operator(yaml.YAMLObject):
    '''A wrapper used for defining callable functions in YAML.

    For example:
    !operator
    module: esmlab.climatology
    function: compute_mon_climatology
    kwargs: {}

    '''

    yaml_tag = u'!operator'

    def __init__(self, module, function, kwargs={}):
        '''Initialize attributes'''
        self.module = module
        self.func = function
        self.kwargs = kwargs

    def __repr__(self):
        '''Return string represention.'''
        return getattr(importlib.import_module(self.module),
                       self.function).__repr__()

    def __call__(self, val):
        '''Call the function!'''
        return getattr(importlib.import_module(self.module),
                       self.function)(val, **self.kwargs)


class process_data_source(object):
    '''Class to support preprocessing operations.'''


    def __init__(self, analysis_name, clobber=False, **query_kwargs):
        importlib.reload(calc)
        importlib.reload(pop_util)

        #-- parse query: hardwired now for certain fields
        self.experiment = query_kwargs['experiment']
        self.variable = query_kwargs.pop('variable')

        # get the analysis definition
        self.analysis_name = analysis_name
        with open('analysis_definitions.yml') as f:
            analysis_defs = yaml.load(f)

        analysis = analysis_defs[analysis_name]

        self.operators = analysis.pop('operators', [lambda ds: ds])

        self.sel_kwargs = analysis.pop('sel_kwargs', {})
        self.isel_kwargs = analysis.pop('isel_kwargs', {})

        self.derived_var_def = analysis.pop('derived_var_def', None)
        self.file_format = analysis.pop('file_format', 'nc')

        #-- do some checking and rationalizing
        if self.file_format not in ['nc','zarr']:
            raise ValueError(f'Unknown file format: {self.file_format}')

        if self.derived_var_def is not None:
            #if not isinstance(self.derived_var_def,project.derived_var_definition):
            #    raise ValueError('Expecting type derived_var_definition')
            self.operators = self.derived_var_def.methods + self.operators
            self.operators_kwargs = self.derived_var_def.methods_kwargs + self.operators_kwargs

        #-- set some attrs
        self.dirout = os.path.join(dirout, 'processed_collections')

        #-- pull specified dataset from catalog
        ensembles = data_catalog.find_in_index(**query_kwargs).ensemble.unique()
        self.n_members = len(ensembles)

        self.cache_locations = []
        self.input = [] # if the cached_locations are present,
                        # then this list will be empty in the returned
                        # object. Could be that the orig files are gone,
                        # (off disk) but the cache remains.

        for ens_i in ensembles:

            file_out = '.'.join([self.experiment,
                                 '%03d'%ens_i,
                                 self.analysis_name,
                                 self.variable,
                                 self.file_format])

            file_out = os.path.join(self.dirout,file_out)
            self.cache_locations.append(file_out)

            if os.path.exists(file_out) and clobber:
                check_call(['rm','-fr',file_out]) # zarr files are directories


            if not os.path.exists(file_out):

                if self.derived_var_def is None:
                    data_desc = data_catalog.get_entries(ensemble=ens_i,
                                                         variable=self.variable,
                                                         **query_kwargs)
                    n_files = len(data_desc['files'])

                else:
                    data_desc = [
                        data_catalog.get_entries(ensemble=ens_i, variable=v,
                                                 **query_kwargs)
                        for v in self.derived_var_def.vars_dependent]

                    n_files = len(data_desc[0]['files'])

                if n_files > 0:
                    self._process(file_out, data_desc)

                else:
                    self.cache_locations.pop(-1)
                    logging.warning(f'No data to generate {file_out}.')

                self.input.append(data_desc)

    def __repr__(self):
        '''Return compact string represention of self.'''
        ens_str = '000'
        if self.n_members > 1:
            ens_str = f'000-{self.n_members:03d}'

        return '.'.join([self.experiment,
                         ens_str,
                         self.analysis_name,
                         self.variable])

    def load(self, **kwargs):
        '''Load the cached data.'''

        # QUESTION: whats the right thing to do if there are no files?
        #           some datasets might not have some variables
        if not self.cache_locations:
            return xr.Dataset()

        option = kwargs.pop('option',None)
        if option not in [None, 'za']:
            raise ValueError(f'Unrecognized option: {option}')

        if option == 'za' and self.file_format == 'zarr':
            raise ValueError(f'File format = zarr is incompatible with za')

        ds_list = []
        for f in self.cache_locations:
            # NOTE: this is probably not the right way to do this
            if option == 'za':
                f = pop_calc_zonal_mean(f)

            ds_list.append(self._open_cached_dataset(f))

        return xr.concat(ds_list,
                         dim='ens',
                         data_vars=[self.variable])

    def _process(self, file_out, data_input):
        '''Apply a preprocessing workflow to specified datasets and save a
           cached file.'''

        files_input = data_input['files']

        # if files_in is a 2D list, merge the files
        if isinstance(files_input[0],list):
            year_offset = data_input['year_offset'][0][0]
            dsi = xr.Dataset()
            for v, f in zip(self.derived_var_def.vars_dependent, files_input):
                dsi = xr.merge((dsi,xr.open_mfdataset(f,
                                                      decode_times=False,
                                                      decode_coords=False,
                                                      data_vars=[v],
                                                      chunks={'time':1})))
        else: # concat with time
            year_offset = data_input['year_offset'][0]
            dsi = xr.open_mfdataset(files_input,
                                    decode_times=False,
                                    decode_coords=False,
                                    #data_vars=[self.variable],
                                    chunks={'time': 1})


        dso = calc.fix_time(dsi, year_offset=year_offset)

        if self.sel_kwargs:
            logging.info(f'Applying sel_kwargs: {self.sel_kwargs}')
            dso = dso.sel(**self.sel_kwargs)

        if self.isel_kwargs:
            logging.info(f'Applying isel_kwargs: {self.isel_kwargs}')
            dso = dso.isel(**self.isel_kwargs)

        for op in self.operators:
            logging.info(f'Applying operator: {op}')
            dso = op(dso)

        dso = calc.unfix_time(dso)

        #dso.to_netcdf(file_out,unlimited_dims=['time'])
        self._write_output(dso, file_out)

        dsi.close()

    def _open_cached_dataset(self,filename):
        '''Open a dataset using appropriate method.'''

        if self.file_format == 'nc':
            ds = xr.open_mfdataset(filename,decode_times=False,
                                     decode_coords=False,
                                     data_vars=[self.variable],
                                     chunks={'time':1})
        elif self.file_format == 'zarr':
            ds = xr.open_zarr(filename,decode_times=False,
                                decode_coords=False)

        #-- fix time?
        return ds

    def _write_output(self, ds, file_out):
        '''Function to write output:
           - add file-level attrs
           - switch method based on file extension
           '''

        if not os.path.exists(self.dirout):
            logging.info(f'creating {self.dirout}')
            os.makedirs(self.dirout)

        if os.path.exists(file_out):
            logging.info(f'removing old {file_out}')
            check_call(['rm','-fr',file_out]) # zarr files are directories

        dsattrs = {
            'history': f'created by {USER} on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            }
        for k,v in self.__dict__.items():
            dsattrs[k] = repr(v)
        ds.attrs.update(dsattrs)

        if self.file_format == 'nc':
            logging.info(f'writing {file_out}')
            ds.to_netcdf(file_out)

        elif self.file_format  == 'zarr':
            logging.info(f'writing {file_out}')
            ds.to_zarr(file_out)