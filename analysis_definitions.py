import esmlab
import pop_regional_means as pop

analysis_defs = {
    'clm-1990s': {
        'operators': esmlab.climatology.compute_mon_climatology,
        'sel_kwargs': {'time': slice('1990-01-01', '1999-12-31')},
        'file_format': 'nc'},
    'ann-rgn': {
        'operators': [pop.make_regional_timeseries,
                      esmlab.climatology.compute_ann_climatology],
        'operators_kwargs': [{}, {}],
        'file_format': 'zarr'}
    }
