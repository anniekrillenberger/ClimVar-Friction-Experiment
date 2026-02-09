#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def create_geopotential_height_map(fileName, experimentName, pressureLevel):
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
    levels = np.arange(60, 135, 1)
    contour = ax.contourf(zg_at_level.lon, zg_at_level.lat, zg_at_level, vmin=60, vmax=135,
                          levels=levels, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
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


def create_geopotential_height_difference_map(refFile, expFile, title, pressureLevel, lev):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    dsRef = xr.open_dataset(refFile, engine='netcdf4', decode_times=time_coder)
    dsExp = xr.open_dataset(expFile, engine='netcdf4', decode_times=time_coder)

    zgRef = dsRef['zg']
    zgExp = dsExp['zg']

    # Time mean
    zgRef_mean = zgRef.mean(dim='time')
    zgExp_mean = zgExp.mean(dim='time')

    # Select pressure level
    zgRef_at_level = zgRef_mean.sel(lev=pressureLevel, method='nearest')
    zgExp_at_level = zgExp_mean.sel(lev=pressureLevel, method='nearest')

    # Difference: experiment - reference
    zg_diff = zgExp_at_level - zgRef_at_level

    # Symmetric color range around 0
    # max_abs = np.nanmax(np.abs(zg_diff.values))
    vmin = float(zg_diff.min())
    vmax = float(zg_diff.max())
    levels = np.linspace(vmin, vmax, 50)

    # Plot
    plt.figure(figsize=(8, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    contour = ax.contourf(
        zg_diff.lon, zg_diff.lat, zg_diff,
        levels=levels,
        cmap='RdBu_r',
        extend='both',
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
    cbar.set_label('Δ Geopotential Height [m]', fontsize=11)

    plt.title(f'{title} (LEV = {lev}): Δzg at {pressureLevel/100:.0f} hPa',
              fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'diff_{expFile}_{refFile}_{pressureLevel/100:.0f}hPa.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    dsRef.close()
    dsExp.close()



def create_geopotential_height_graph(file1, file2, file3, label1, label2, label3, lev, 
                                     pressureLevel, ymax, totalYears=15):
    plt.figure(figsize=(14, 6))
    
    files = [file1, file2, file3]
    labels = [label1, label2, label3]
    colors = ['blue', 'red', 'green']
    all_data = []
    
    for fname, label, color in zip(files, labels, colors):
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(fname, engine='netcdf4', decode_times=time_coder)
        zg = ds['zg']
        
        zg_at_level = zg.sel(lev=pressureLevel, method='nearest')
        zg_global_mean = zg_at_level.mean(dim=['lat', 'lon'])
        
        # Convert time to years for x-axis
        total_timesteps = len(zg_global_mean)
        time_in_years = np.linspace(0, totalYears, total_timesteps)
        
        plt.plot(time_in_years, zg_global_mean, linewidth=1.5, label=label, color=color, alpha=0.8)
        
        # Store final equilibrated value
        spinup_idx = int(total_timesteps / totalYears)
        equilibrated_mean = zg_global_mean.isel(time=slice(spinup_idx, None)).mean().values
        all_data.append({
            'label': label,
            'equilibrated_mean': equilibrated_mean,
            'final_value': zg_global_mean.isel(time=-1).values
        })

        ds.close()
    
    # Formatting
    plt.xlabel('Time [years]', fontsize=13)
    plt.ylabel('Global Mean Geopotential Height [m]', fontsize=13)
    plt.title(f'Geopotential Height Time Evolution at {pressureLevel/100:.0f} hPa, LEV = {lev}', 
                fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(top=ymax)
    plt.tight_layout()
    plt.savefig(f'zg_comparison_{pressureLevel/100:.0f}hPa_LEV{lev}.png', dpi=300)
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print(f"Statistics at {pressureLevel/100:.0f} hPa")
    print("="*70)
    print(f"{'Experiment':<25} {'Equilibrated Mean (m)':<25} {'Final Value (m)':<20}")
    print("-"*70)
    for data in all_data:
        print(f"{data['label']:<25} {data['equilibrated_mean']:>22.2f} {data['final_value']:>17.2f}")
    print("="*70)
    
    # Calculate differences
    print("\nDifferences from Reference:")
    ref_mean = all_data[0]['equilibrated_mean']
    for data in all_data[1:]:
        diff = data['equilibrated_mean'] - ref_mean
        percent = (diff / ref_mean) * 100
        print(f"  {data['label']}: {diff:+.2f} m ({percent:+.3f}%)")
    print()
    return

'''
create_geopotential_height_map('code_156-lessfric5',  'Less Friction: LEV = 5',   100000)
create_geopotential_height_map('code_156-lessfric10', 'Less Friction: LEV = 10',  100000)
create_geopotential_height_map('code_156-lessfric15', 'Less Friction: LEV = 15',  100000)
create_geopotential_height_map('code_156-morefric5',  'More Friction: LEV = 5',   100000)
create_geopotential_height_map('code_156-morefric10', 'More Friction: LEV = 10',  100000)
create_geopotential_height_map('code_156-morefric15', 'More Friction: LEV = 15',  100000)
create_geopotential_height_map('code_156-ref5',       'Reference: LEV = 5',       100000)
create_geopotential_height_map('code_156-ref10',      'Reference: LEV = 10',      100000)
create_geopotential_height_map('code_156-ref15',      'Reference: LEV = 15',      100000)

create_geopotential_height_difference_map('code_156-ref15', 'code_156-morefric15', 'More Friction Difference', 100000, 15)
create_geopotential_height_difference_map('code_156-ref15', 'code_156-lessfric15', 'Less Friction Difference', 100000, 15)

create_geopotential_height_graph('code_156-ref15', 'code_156-lessfric15', 'code_156-morefric15',
                                 'Reference', 'Less Friction', 'More Friction', 15, 50000, 5470)
create_geopotential_height_graph('code_156-ref15', 'code_156-lessfric15', 'code_156-morefric15',
                                 'Reference', 'Less Friction', 'More Friction', 15, 85000, 1410)
create_geopotential_height_graph('code_156-ref15', 'code_156-lessfric15', 'code_156-morefric15',
                                 'Reference', 'Less Friction', 'More Friction', 15, 100000, 105)
                                 '''


create_geopotential_height_difference_map('code_156-ref15', 'code_156-morefric15', 'More Friction Difference', 50000, 15)
create_geopotential_height_difference_map('code_156-ref15', 'code_156-lessfric15', 'Less Friction Difference', 50000, 15)
