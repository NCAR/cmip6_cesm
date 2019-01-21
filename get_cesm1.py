#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
from subprocess import check_call, Popen, PIPE


USER = os.environ['USER']
file_format = 'zarr'

project_name = os.path.basename(os.getcwd())
dirout = f'/glade/scratch/{USER}/calcs/{project_name}'

case = 'b40.20th.1deg.bdrd.001'
loc_path = f'{dirout}/cesm1/{case}/ocn/proc/tseries/monthly'
hsi_path = f'/CCSM/csm/{case}/ocn/proc/tseries/monthly'

if not os.path.exists(loc_path):
    check_call(['mkdir','-p',loc_path])

variable_list = ['TEMP','SALT','CFC11','CFC12',
                 'NO3','PO4','SiO3','DIC','O2','Fe',
                 'diatChl','spChl','diazChl',
                 'DENITRIF','NITRIF','diaz_Nfix',
                 'IRON_FLUX','FG_CO2','PD','KAPPA_THIC','KAPPA_ISOP','MOC']

#datestr = '199001-199912'
datestr = '??????-??????'
for v in variable_list:
    rfile_patt = f'{hsi_path}/{case}.pop.h.{v}.{datestr}.nc'
    p = Popen(['hsi','ls -1 '+rfile_patt],stdin=None,stdout=PIPE,stderr=PIPE)
    stdout, stderr = p.communicate()
    rfiles = stderr.decode('UTF-8').strip().split('\n')[1:]

    for rfile in rfiles:
        lfile = rfile.replace(hsi_path,loc_path)
        if not os.path.exists(lfile):
            check_call(['hsi','cget '+lfile+' : '+rfile])
