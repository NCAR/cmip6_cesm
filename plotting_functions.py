import os
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import plottools as pt

def plot_cfc_bias_section(lat,z_t,bias,left_str,center_str,right_str,plot_name=''):
    
    lo = -90.
    hi = 90.
    dc = 5.
    cnlevels = np.arange(lo,hi+dc,dc)


    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.51,0.7,0.35])    
    ax2 = fig.add_axes([0.1,0.1,0.7,0.4])    
    mesh = ax1.contourf(lat, z_t*1e-2, bias[:,:],
                        levels=cnlevels, 
                        cmap = cmocean.cm.delta,
                        extend='both')

    mesh2 = ax2.contourf(lat, z_t*1e-2, bias[:,:],
                         levels=cnlevels, 
                         cmap = cmocean.cm.delta,
                         extend='both')
    ax1.set_ylim([1000.,0.])
    ax2.set_ylim([5500.,1000.])

    ax1.set_xlim([-78,70])
    ax2.set_xlim([-78,70])

    ax1.set_xticklabels([])
    ax1.set_yticklabels(np.arange(0,1000,200))
    ax1.minorticks_on()
    ax1.xaxis.set_ticks_position('top')

    ax2.minorticks_on()
    ax2.set_xlabel('Latitude')
    ax2.xaxis.set_ticks_position('bottom')

    ax2.set_ylabel('Depth [m]')
    ax2.yaxis.set_label_coords(-0.12, 1.05)

    cbaxes = fig.add_axes([0.85, 0.2, 0.03, 0.6]) 
    cb = plt.colorbar(mesh, cax = cbaxes, drawedges=True)  

    cb.ax.set_title('patm',y=1.04)

    h = ax1.set_title(left_str,loc='left')
    h = ax1.set_title(center_str,loc='center')
    h = ax1.set_title(right_str,loc='right')    
    if plot_name:
        plt.savefig(plot_name,dpi=300,bbox_inches='tight')
        
        
def contour(ax,field):
    if field == 'slp_mean':
        dP = 4.
        levels = np.arange(952,1048+dP,dP)
        cmap = cmocean.cm.curl
    elif field == 'slp_std':
        dP = 1.
        levels = np.arange(0.5,17+dP,dP)
        cmap = cmocean.cm.curl
        
    lon = np.concatenate((ds.longitude.values,[359.99999]))
    data = add_cyclic_point(ds[field].values)

    cf = ax.contourf(lon, ds.latitude, data,
                 levels = levels,
                 cmap = cmap,
                 extend = 'both',
                 transform=ccrs.PlateCarree())
    zorder = 0
    for contour_level in cf.collections:
        contour_level.set_zorder(zorder)
        contour_level.set_rasterized(True)
        
    cs = ax.contour(lon, ds.latitude, data,
                    levels = levels,
                    colors='k',
                    linewidths = 0.5,
                    transform=ccrs.PlateCarree(),
                    zorder=len(cf.collections)+10)
    cs.levels = [contour_label_format(val) for val in cs.levels]
    fmt = '%r'
    lb = plt.clabel(cs, fontsize=8, inline=True, fmt=fmt)
    
    return cf


def moc_plot(moc,left_str,center_str,plot_name=''):
    fig = plt.figure(figsize=(8,6))
    lo = -40.
    hi = 40.
    dc = 4.
    cnlevels = np.arange(lo,hi+dc,dc)

    lat = moc.lat_aux_grid
    z = moc.moc_z * 1e-2

    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.51,0.7,0.35])    
    ax2 = fig.add_axes([0.1,0.1,0.7,0.4])   

    ax1.set_facecolor('gainsboro')
    ax2.set_facecolor('gainsboro')

    cs = ax1.contour(lat, z, moc,
                       colors='k',
                       linewidths=1,
                       levels=cnlevels)

    
    cs.levels = [pt.contour_label_format(val) for val in cs.levels]
    fmt = '%r'
    lb = plt.clabel(cs, fontsize=8, inline=True, fmt=fmt)
    
    mesh = ax1.contourf(lat, z, moc,
                    levels = cnlevels,
                    cmap = 'PRGn',
                    extend='both',zorder=-1000)
    
    cs2 = ax2.contour(lat, z, moc,
                      colors='k',
                      linewidths=1,
                      levels=cnlevels)

    cs2.levels = [pt.contour_label_format(val) for val in cs2.levels]
    fmt = '%r'
    lb = plt.clabel(cs2, fontsize=8, inline=True, fmt=fmt)
    
    
    mesh2 = ax2.contourf(lat, z, moc,
                         levels = cnlevels,
                         cmap = 'PRGn',
                         extend='both',zorder=-1000)
    
    ax1.set_ylim([1000.,0.])
    ax2.set_ylim([5500.,1000.])

    ax1.set_xlim([-90,90])
    ax2.set_xlim([-90,90])

    ax1.set_xticklabels([])
    ax1.set_yticklabels(np.arange(0,1000,200))
    ax1.minorticks_on()
    ax1.xaxis.set_ticks_position('top')

    ax2.minorticks_on()
    ax2.set_xlabel('Latitude')
    ax2.xaxis.set_ticks_position('bottom')

    ax2.set_ylabel('Depth [m]')
    ax2.yaxis.set_label_coords(-0.12, 1.05)

    h = ax1.set_title(left_str,loc='left')
    h = ax1.set_title(center_str,loc='center')
    #h = ax1.set_title(right_str,loc='right')    

    if plot_name:
        plt.savefig(plot_name,dpi=300,bbox_inches='tight')