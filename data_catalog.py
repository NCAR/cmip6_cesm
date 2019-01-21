#! /usr/bin/env python
import os
import re
import pandas as pd
import numpy as np
import xarray as xr
import yaml

import logging

logging.basicConfig(level=logging.INFO)

libdir = './lib_data_catalog'
active_database_file_name = None


def set_catalog(catalog_name):
    '''Point to a catalog database file.'''
    # TODO: this is not threadsafe...whole thing should be a class
    global active_database_file_name
    active_database_file_name = f'{libdir}/{catalog_name}.csv'
    print(f'active catalog: {catalog_name}')


def get_files(**kwargs):
    '''return files according to requested data.

    '''
    df_subset = find_in_index(**kwargs)

    return df_subset.files.tolist()


def get_entries(**kwargs):
    '''Return a dictionary with all entries of query as lists.'''
    df_subset = find_in_index(**kwargs)
    return {key: df_subset[key].tolist() for key in df_subset}


def find_in_index(**kwargs):
    '''Return subset of database according to requested data.
    '''
    if active_database_file_name is None:
        raise ValueError('no catalog set.')

    df = pd.read_csv(active_database_file_name, index_col=0)

    for key in kwargs.keys():
        if key not in df.columns:
            raise ValueError(f'"{key}" is not a column')

    query = np.ones(len(df), dtype=bool)
    for key, val in kwargs.items():
        if isinstance(val,list):
            query_i = np.zeros(len(df), dtype=bool)
            for vali in val:
                query_i = query_i | (df[key] == vali)
        else:
            query_i = (df[key] == val)
        query = query & query_i

    return df.loc[query].sort_values(by=['sequence_order', 'files'],
                                     ascending=True)


def build_catalog(collection_input_file, clobber=False):
    '''Generate a catalog from input file.'''

    #-- open the input file
    with open(collection_input_file) as f:
        collections = yaml.load(f)

    #-- open the collection definition file
    for catalog_name, collection in collections.items():

        collection_type = collection['type']

        # add check for existence
        with open(f'{libdir}/{collection_type}_definitions.yml') as f:
            catalog_definition = yaml.load(f)

        catalog_columns = catalog_definition['catalog_columns']
        for req_col in ['files', 'sequence_order']:
            if not req_col in catalog_columns:
                raise ValueError(f'missing required column: {req_col}')

        set_catalog(catalog_name)
        if os.path.isfile(active_database_file_name) and not clobber:
            continue

        #-- build the catalog
        if collection_type.lower() == 'cesm':
            build_method = _build_catalog_cesm

        build_method(collection, catalog_columns, catalog_definition)


def _extract_cesm_date_str(filename):
    '''Extract a datastr from file name.'''

    # must be in order of longer to shorter strings
    # TODO: make this function return a date object as well as string
    datestrs = [r'\d{12}Z-\d{12}Z',
                r'\d{10}Z-\d{10}Z',
                r'\d{8}-\d{8}',
                r'\d{6}-\d{6}',
                r'\d{4}-\d{4}']

    for datestr in datestrs:
        match = re.compile(datestr).findall(filename)
        if match:
            return match[0]

    raise ValueError(f'unable to match date string: {filename}')


def _extract_cesm_varname_str(filename, case, component, component_streams):
    '''Extract varname from case.stream.varname.datestr.nc file pattern.'''

    # define lists of stream strings
    datestr = _extract_cesm_date_str(filename)

    # loop over stream strings (order matters!)
    for stream in sorted(component_streams[component], key=lambda s: len(s),
                         reverse=True):

        # search for case.stream part of filename
        case_stream = '.'.join([case, stream])
        s = filename.find(case_stream)

        if s >= 0: # got a match
            # get varname.datestr.nc part of filename
            l = len(case_stream)
            varname_datestr_nc = filename[s+l+1:]
            varname = varname_datestr_nc[:varname_datestr_nc.find('.')]

            # assert expected pattern
            datestr_nc = varname_datestr_nc[
                varname_datestr_nc.find(f'.{varname}.')+len(varname)+2:]

            # ensure that file name conforms to expectation
            assert datestr_nc == f'{datestr}.nc', (f'Filename: {filename} does'
                                                   ' not conform to expected'
                                                   ' pattern')

            return varname

def _cesm_filename_parts(filename, component, component_streams):
    '''Extract each part of case.stream.variable.datestr.nc file pattern.'''

    # define lists of stream strings
    datestr = _extract_cesm_date_str(filename)

    # loop over stream strings (order matters!)
    for stream in sorted(component_streams[component], key=lambda s: len(s),
                         reverse=True):

        # search for case.stream part of filename
        s = filename.find(stream)

        if s >= 0: # got a match
            # get varname.datestr.nc part of filename
            case = filename[0:s-1]
            l = len(stream)
            variable_datestr_nc = filename[s+l+1:]
            variable = variable_datestr_nc[:variable_datestr_nc.find('.')]

            # assert expected pattern
            datestr_nc = variable_datestr_nc[
                variable_datestr_nc.find(f'.{variable}.')+len(variable)+2:]

            # ensure that file name conforms to expectation
            if datestr_nc != f'{datestr}.nc':
                logging.warning(f'Filename: {filename} does'
                                ' not conform to expected'
                                ' pattern')
                return
            return {'case': case, 'stream': stream, 'variable': variable,
                    'datestr': datestr}

def _build_catalog_cesm(collection, catalog_columns, catalog_definition):
    '''Build a CESM data catalog.'''

    component_streams = catalog_definition['component_streams']

    replacements = {}
    if 'replacements' in catalog_definition:
        replacements = catalog_definition['replacements']

    df = pd.DataFrame(columns=catalog_columns)

    for experiment, ensembles in collection['data_sources'].items():

        entry = {'experiment': experiment}

        for ens, d_attrs in enumerate(ensembles):

            root_dir = d_attrs['root_dir']
            case = d_attrs['case']
            path_structure = d_attrs['path_structure']
            component_attrs = d_attrs['component_attrs']
            entry.update({key: val for key, val in d_attrs.items()
                               if key in catalog_columns})

            if 'ensemble' not in d_attrs:
                entry.update({'ensemble': ens})
                                
            if 'sequence_order' not in d_attrs:
                entry.update({'sequence_order': 0})

            path_structure_split = path_structure.split('/')
            index_comp = path_structure_split.index('component') - len(path_structure_split)
            index_freq = path_structure_split.index('freq') - len(path_structure_split)

            w = os.walk(os.path.join(root_dir))

            for root, dirs, files in w:

                if not files:
                    continue

                sfiles = sorted([f for f in files if f.endswith('.nc')])
                if not sfiles:
                    continue

                # expecting: path/to/here/component/proc/tseries/freq/*.nc
                # in some cases, for instance, there will be files in
                # component/hist, but we don't want those
                root_split = root.split('/')
                if 'proc' not in root_split and 'tseries' not in root_split:
                    continue

                component = root_split[index_comp]
                freq = root_split[index_freq]

                if component not in component_streams:
                    print(root_split)
                    raise ValueError(f'unknown component: {component}')
                entry.update({'component': component, 'freq': freq})

                if component in component_attrs:
                    entry.update({key: val for key, val in
                                  component_attrs[component].items()
                                  if key in catalog_columns})
                fs = []
                for f in sfiles:
                    fileparts = _cesm_filename_parts(f, component,
                                                     component_streams)

                    if fileparts is None:
                        continue
                    if fileparts['case'] != case:
                        continue

                    #variable = _extract_cesm_varname_str(f, case, component,
                    #                                     component_streams)
                    entry.update({'variable': fileparts['variable'],
                                  'freq': freq,
                                  'date_range': fileparts['datestr'].split('-'),
                                  'file_basename': f,
                                  'files': os.path.join(root,f)})

                    completed_entry = dict(entry)
                    for key, replist in replacements.items():
                        if key in completed_entry:
                            for old_new in replist:
                                if completed_entry[key] == old_new[0]:
                                    completed_entry[key] = old_new[1]

                    fs.append(completed_entry)

                if fs:
                    temp_df = pd.DataFrame(fs)
                else:
                    temp_df = pd.DataFrame(columns=df.columns)

                df = pd.concat([temp_df, df], ignore_index=True, sort=False)

    df.to_csv(active_database_file_name)
