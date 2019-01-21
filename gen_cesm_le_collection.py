#! /usr/bin/env python

from itertools import chain
import yaml

entry = {'case': 'b.e11.B1850C5CN.f09_g16.005',
         'ensemble': 5,
         'root_dir': '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE',
         'path_structure': 'component/proc/tseries/freq/variable',
         'year_offset': 1448,
         'component_attrs': {'ocn': {'grid': 'POP_gx1v6'}}}

collection = {'CTRL': [entry]}

# 3-8 have no BGC fields
# 33 has zeros for O2 in March 1938; 25 is missing a DENITRIF file
exclude_ens = [ii for ii in range(3,9)]+[33,25]


scenario_case = {'20C': 'b.e11.B20TRC5CNBDRD.f09_g16',
                 'RCP85': 'b.e11.BRCP85C5CNBDRD.f09_g16',
                 'RCP45': 'b.e11.BRCP45C5CNBDRD.f09_g16'}

for scenario, case in scenario_case.items():
    collection[scenario] = []
    if scenario == '20C':
        sequence_order = 0
    else:
        sequence_order = 1

    if scenario in ['20C', 'RCP85']:
        ens = ['%03d'%i for i in chain(range(1,36),range(101,106))]
    elif scenario in ['RCP45']:
        ens = ['%03d'%i for i in range(1,16)]


    for e in ens:

        if int(e) in exclude_ens:
            has_ocean_bgc = False
        else:
            has_ocean_bgc = True

        entry = {'case': f'{case}.{e}',
                 'ensemble': int(e),
                 'sequence_order': sequence_order,
                 'has_ocean_bgc': has_ocean_bgc,
                 'root_dir': '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE',
                 'path_structure': 'component/proc/tseries/freq/variable',
                 'component_attrs': {'ocn': {'grid': 'POP_gx1v6'}}}
        collection[scenario].append(entry)

collections = {'cesm1_le': {'type': 'cesm',
                            'data_sources': collection}}

with open('collection_cesm1_le.yml', 'w') as f:
    yaml.dump(collections, f)
