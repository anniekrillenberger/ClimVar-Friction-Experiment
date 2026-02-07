#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_geopotential_height_graph(fileName, experimentName, pressureLevel):
    # load data from netCDF file
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(fileName, engine='netcdf4', decode_times=time_coder)
    zg = ds['zg']

    # overview of file data
    print("\n" + "="*60)
    print("Dataset Summary:")
    print(zg)
    print(f"Time range: {zg.time.values[0]} to {zg.time.values[-1]}")
    print(f"Pressure levels: {zg.lev.values/100} hPa")
    print(f"Spatial resolution: {len(zg.lat)} lat x {len(zg.lon)} lon")
    print("="*60 + "\n")

    # calculate the average across all 5400 time steps for each combination of (lev, lat, lon)
    zg_mean = zg.mean(dim='time')
    # select a single pressure level 
    zg_at_level = zg_mean.sel(lev=pressureLevel, method='nearest')

    # plot!
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    contour = ax.contourf(zg_at_level.lon, zg_at_level.lat, zg_at_level, 
                        levels=20, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
    contour_lines = ax.contour(zg_at_level.lon, zg_at_level.lat, zg_at_level,
                            levels=10, colors='black', linewidths=0.5, 
                            alpha=0.4, transform=ccrs.PlateCarree())
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.0f')
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                        pad=0.05, shrink=0.8)
    cbar.set_label('Geopotential Height [m]', fontsize=11)
    plt.title(f'{experimentName}: Time-averaged Geopotential Height at {pressureLevel/100:.0f} hPa', 
            fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fileName}_zg_contour_{pressureLevel/100:.0f}hPa.png', 
            dpi=300, bbox_inches='tight')
    plt.show()

    return

create_geopotential_height_graph('code_156-lessfric5',  'Less Friction: LEV = 5',   10000)
create_geopotential_height_graph('code_156-lessfric10', 'Less Friction: LEV = 10',  10000)
create_geopotential_height_graph('code_156-lessfric15', 'Less Friction: LEV = 15',  10000)
create_geopotential_height_graph('code_156-morefric5',  'More Friction: LEV = 5',   10000)
create_geopotential_height_graph('code_156-morefric10', 'More Friction: LEV = 10',  10000)
create_geopotential_height_graph('code_156-morefric15', 'More Friction: LEV = 15',  10000)
create_geopotential_height_graph('code_156-ref5',       'Reference: LEV = 5',       10000)
create_geopotential_height_graph('code_156-ref10',      'Reference: LEV = 10',      10000)
create_geopotential_height_graph('code_156-ref15',      'Reference: LEV = 15',      10000)
