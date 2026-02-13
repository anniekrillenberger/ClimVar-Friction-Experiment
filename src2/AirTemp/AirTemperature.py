#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def create_air_temperature_map(fileName, dvdiff, pressureLevel):
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

    # Exclude first year (spinup) - assuming 360 time steps per year
    # Adjust this number based on your actual temporal resolution
    steps_per_year = len(ta.time) // 15  # Total steps / 15 years
    spinup_steps = steps_per_year  # First year
    ta_no_spinup = ta.isel(time=slice(spinup_steps, None))

    # calculate the average across all 5400 time steps for each combination of (lev, lat, lon)
    # result: ta_mean has shape (lev, lat, lon) - time dimension is removed
    ta_mean = ta_no_spinup.mean(dim='time')
    # select a single air pressure at each pixel
    ta_at_level = ta_mean.sel(lev=pressureLevel, method='nearest')

    def levels():
        match pressureLevel:
            case 50000:
                return np.arange(233, 269, 0.1)
            case 85000:
                return np.arange(239, 295, 0.1)
            case 100000:
                return np.arange(250, 298, 0.1)
            case _:
                return np.arange(ta_at_level.min().values, ta_at_level.max().values, 0.1)

    # plot!
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    contour = ax.contourf(ta_at_level.lon, ta_at_level.lat, ta_at_level, 
                          levels=levels(), cmap='RdYlBu_r', transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                    linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                        pad=0.05, shrink=0.8)
    cbar.locator = mticker.MaxNLocator(nbins=6, prune=None)
    cbar.update_ticks()
    cbar.set_label('Air Temperature [K]', fontsize=11)
    plt.title(f'Vertical Diffusion Coefficient = {dvdiff} m\u00b2/s: Time-averaged Air Temperature at {pressureLevel/100:.0f} hPa', 
            fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fileName}_ta_contour_{pressureLevel/100:.0f}hPa.png', 
            dpi=300, bbox_inches='tight')
    plt.show()

    return


def create_air_temperature_difference_map(refFile, expFile, pressureLevel, dvdiff):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    dsRef = xr.open_dataset(refFile, engine='netcdf4', decode_times=time_coder)
    dsExp = xr.open_dataset(expFile, engine='netcdf4', decode_times=time_coder)

    taRef = dsRef['ta']
    taExp = dsExp['ta']

    # exclude 1st year (spinup) - assuming 360 time steps per year
    n_years_total = 15  # your experiment length
    steps_per_year_ref = len(taRef.time) // n_years_total
    steps_per_year_exp = len(taExp.time) // n_years_total

    taRef_no_spinup = taRef.isel(time=slice(steps_per_year_ref, None))
    taExp_no_spinup = taExp.isel(time=slice(steps_per_year_exp, None))

    # Time mean
    taRef_mean = taRef_no_spinup.mean(dim='time')
    taExp_mean = taExp_no_spinup.mean(dim='time')

    # Select pressure level
    taRef_at_level = taRef_mean.sel(lev=pressureLevel, method='nearest')
    taExp_at_level = taExp_mean.sel(lev=pressureLevel, method='nearest')

    # Difference: experiment - reference
    ta_diff = taExp_at_level - taRef_at_level

    # Symmetric color scale around 0
    vmax = float(np.nanmax(np.abs(ta_diff.values)))
    vmin = -vmax

    levels = np.linspace(vmin, vmax, 75)
                
    # Plot
    plt.figure(figsize=(8, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    contour = ax.contourf(
        ta_diff.lon, ta_diff.lat, ta_diff,
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
    cbar.locator = mticker.MaxNLocator(nbins=6, prune=None)
    cbar.update_ticks()
    cbar.set_label('Δ Air Temperature [K]', fontsize=11)

    plt.title(f'Difference Between {dvdiff} & 0 m\u00b2/s: Δta at {pressureLevel/100:.0f} hPa',
              fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'diff_{expFile}_{refFile}_{pressureLevel/100:.0f}hPa.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    dsRef.close()
    dsExp.close()


def create_air_temperature_graph(file1, file2, file3, label1, label2, label3, ymax, pressureLevel, totalYears=15):
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
    plt.title(f'Air Temperature Time Evolution at {pressureLevel/100:.0f} hPa', 
                fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', title='Vertical Diffusion Coefficient')
    plt.ylim(top=ymax)
    plt.tight_layout()
    plt.savefig(f'ta_comparison_{pressureLevel/100:.0f}hPa_LEV.png', dpi=300)
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


create_air_temperature_map('code_130-DVDIFF0',  0,  50000)
create_air_temperature_map('code_130-DVDIFF5',  5,  50000)
create_air_temperature_map('code_130-DVDIFF10', 10, 50000)
create_air_temperature_map('code_130-DVDIFF0',  0,  85000)
create_air_temperature_map('code_130-DVDIFF5',  5,  85000)
create_air_temperature_map('code_130-DVDIFF10', 10, 85000)
create_air_temperature_map('code_130-DVDIFF0',  0,  100000)
create_air_temperature_map('code_130-DVDIFF5',  5,  100000)
create_air_temperature_map('code_130-DVDIFF10', 10, 100000)

create_air_temperature_difference_map('code_130-DVDIFF0', 'code_130-DVDIFF10', 50000,   10)
create_air_temperature_difference_map('code_130-DVDIFF0', 'code_130-DVDIFF10', 85000,   10)
create_air_temperature_difference_map('code_130-DVDIFF0', 'code_130-DVDIFF10', 100000,  10)

create_air_temperature_graph('code_130-DVDIFF0', 'code_130-DVDIFF5', 'code_130-DVDIFF10', '0 m\u00b2/s', '5 m\u00b2/s', '10 m\u00b2/s', 255, 50000)
create_air_temperature_graph('code_130-DVDIFF0', 'code_130-DVDIFF5', 'code_130-DVDIFF10', '0 m\u00b2/s', '5 m\u00b2/s', '10 m\u00b2/s', 270.5, 85000)
create_air_temperature_graph('code_130-DVDIFF0', 'code_130-DVDIFF5', 'code_130-DVDIFF10', '0 m\u00b2/s', '5 m\u00b2/s', '10 m\u00b2/s', 277.5, 100000)
