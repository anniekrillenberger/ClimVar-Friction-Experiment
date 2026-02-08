#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def create_air_temperature_map(fileName, experimentName, pressureLevel):
    # load data from netCDF file
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(fileName, engine='netcdf4', decode_times=time_coder)
    print(ds)
    ta = ds['ta']

    # overview of file data
    print("\n" + "="*60)
    print("Dataset Summary:")
    print(ta)
    print(f"Time range: {ta.time.values[0]} to {ta.time.values[-1]}")
    print(f"Pressure levels: {ta.lev.values/100} hPa")
    print(f"Spatial resolution: {len(ta.lat)} lat x {len(ta.lon)} lon")
    print("="*60 + "\n")

    # calculate the average across all 5400 time steps for each combination of (lev, lat, lon)
    # result: ta_mean has shape (lev, lat, lon) - time dimension is removed
    ta_mean = ta.mean(dim='time')
    # select a single air pressure at each pixel
    ta_at_level = ta_mean.sel(lev=pressureLevel, method='nearest')

    # plot!
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    levels = np.arange(230, 274.5, 0.5)  # 89 levels, very smooth
    contour = ax.contourf(ta_at_level.lon, ta_at_level.lat, ta_at_level, 
                          vmin=230, vmax=274, levels=levels, cmap='RdYlBu_r', 
                          transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                    linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                        pad=0.05, shrink=0.8)
    cbar.set_label('Air Temperature [K]', fontsize=11)
    plt.title(f'{experimentName}: Time-averaged Air Temperature at {pressureLevel/100:.0f} hPa', 
            fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fileName}_ta_contour_{pressureLevel/100:.0f}hPa.png', 
            dpi=300, bbox_inches='tight')
    plt.show()

    return

def create_air_temperature_graph(file1, file2, file3, label1, label2, label3, lev, 
                                     pressureLevel, ymax, totalYears=15):
    plt.figure(figsize=(14, 6))
    
    files = [file1, file2, file3]
    labels = [label1, label2, label3]
    colors = ['blue', 'red', 'green']
    all_data = []
    
    for fname, label, color in zip(files, labels, colors):
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(fname, engine='netcdf4', decode_times=time_coder)
        ta = ds['ta']
        
        ta_at_level = ta.sel(lev=pressureLevel, method='nearest')
        ta_global_mean = ta_at_level.mean(dim=['lat', 'lon'])
        
        # Convert time to years for x-axis
        total_timesteps = len(ta_global_mean)
        time_in_years = np.linspace(0, totalYears, total_timesteps)
        
        plt.plot(time_in_years, ta_global_mean, linewidth=1.5, label=label, color=color, alpha=0.8)
        
        # Store final equilibrated value
        spinup_idx = int(total_timesteps / totalYears)
        equilibrated_mean = ta_global_mean.isel(time=slice(spinup_idx, None)).mean().values
        all_data.append({
            'label': label,
            'equilibrated_mean': equilibrated_mean,
            'final_value': ta_global_mean.isel(time=-1).values
        })

        ds.close()
    
    # Formatting
    plt.xlabel('Time [years]', fontsize=13)
    plt.ylabel('Global Mean Air Temperature [K]', fontsize=13)
    plt.title(f'Air Temperature Time Evolution at {pressureLevel/100:.0f} hPa, LEV = {lev}', 
                fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(top=ymax)
    plt.tight_layout()
    plt.savefig(f'ta_comparison_{pressureLevel/100:.0f}hPa_LEV{lev}.png', dpi=300)
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print(f"Statistics at {pressureLevel/100:.0f} hPa")
    print("="*70)
    print(f"{'Experiment':<25} {'Equilibrated Mean (K)':<25} {'Final Value (K)':<20}")
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

create_air_temperature_map('code_130-lessfric5',  'Less Friction: LEV = 5',   50000)
create_air_temperature_map('code_130-lessfric10', 'Less Friction: LEV = 10',  50000)
create_air_temperature_map('code_130-lessfric15', 'Less Friction: LEV = 15',  50000)
create_air_temperature_map('code_130-morefric5',  'More Friction: LEV = 5',   50000)
create_air_temperature_map('code_130-morefric10', 'More Friction: LEV = 10',  50000)
create_air_temperature_map('code_130-morefric15', 'More Friction: LEV = 15',  50000)
create_air_temperature_map('code_130-ref5',       'Reference: LEV = 5',       50000)
create_air_temperature_map('code_130-ref10',      'Reference: LEV = 10',      50000)
create_air_temperature_map('code_130-ref15',      'Reference: LEV = 15',      50000)


create_air_temperature_graph('code_130-ref15', 'code_130-lessfric15', 'code_130-morefric15',
                             'Reference', 'Less Friction', 'More Friction', 15, 50000, 256)
create_air_temperature_graph('code_130-ref15', 'code_130-lessfric15', 'code_130-morefric15',
                             'Reference', 'Less Friction', 'More Friction', 15, 85000, 273)
create_air_temperature_graph('code_130-ref15', 'code_130-lessfric15', 'code_130-morefric15',
                             'Reference', 'Less Friction', 'More Friction', 15, 100000, 277)
