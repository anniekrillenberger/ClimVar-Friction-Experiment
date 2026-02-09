#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_wind_vector_plot(fileName_u, fileName_v, experimentName, pressureLevel, lev, totalYears=15):
    
    # Load data
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    dsu = xr.open_dataset(fileName_u, engine='netcdf4', decode_times=time_coder)
    dsv = xr.open_dataset(fileName_v, engine='netcdf4', decode_times=time_coder)
    
    # Get eastward (u) and northward (v) wind components
    ua = dsu['ua']  # eastward wind
    va = dsv['va']  # northward wind
    
    print(f"\nProcessing: {experimentName}")
    print(f"UA (eastward wind): {ua}")
    print(f"VA (northward wind): {va}")
    
    # Skip spin-up period
    total_timesteps = len(ua.time)
    spinup_timesteps = int(total_timesteps / totalYears)
    
    ua_equilibrated = ua.isel(time=slice(spinup_timesteps, None))
    va_equilibrated = va.isel(time=slice(spinup_timesteps, None))
    
    # Time average
    ua_mean = ua_equilibrated.mean(dim='time')
    va_mean = va_equilibrated.mean(dim='time')
    
    # Select pressure level
    ua_at_level = ua_mean.sel(lev=pressureLevel, method='nearest')
    va_at_level = va_mean.sel(lev=pressureLevel, method='nearest')
    
    # Calculate wind speed (magnitude)
    wind_speed = np.sqrt(ua_at_level**2 + va_at_level**2)
    
    print(f"\nWind speed range: {wind_speed.min().values:.2f} - {wind_speed.max().values:.2f} m/s")
    
    # Create plot
    plt.figure(figsize=(18, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot wind speed as filled contours (background color)
    levels = np.arange(0, 6, 0.5)
    contour = ax.contourf(wind_speed.lon, wind_speed.lat, wind_speed,
                         vmin=0, vmax=6, levels=levels, cmap='YlOrRd', 
                         transform=ccrs.PlateCarree())
    
    # Subsample for quiver plot (every Nth point to avoid clutter)
    skip = 3  # Plot every 2nd point (adjust based on your resolution)
    
    # Create quiver (vector arrows)
    quiver = ax.quiver(ua_at_level.lon[::skip], ua_at_level.lat[::skip],
                      ua_at_level[::skip, ::skip], va_at_level[::skip, ::skip],
                      transform=ccrs.PlateCarree(),
                      scale=100,  # Adjust to change arrow length
                      width=0.003,  # Arrow width
                      headwidth=3,  # Arrow head width
                      headlength=4,  # Arrow head length
                      color='black',
                      alpha=0.7)
    
    # Add quiver key (reference arrow)
    ax.quiverkey(quiver, 0.9, 1.05, 2.5, '2.5 m/s', 
                labelpos='E', coordinates='axes')
    
    # Add geographic features
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Colorbar for wind speed
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('Wind Speed [m/s]', fontsize=11)
    
    plt.title(f'{experimentName}: Mean Wind at {pressureLevel/100:.0f} hPa',
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{fileName_u}_{fileName_v}_wind_vectors_{pressureLevel/100:.0f}hPa_LEV{lev}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    dsu.close()
    dsv.close()

def create_eastward_wind_graph(file1, file2, file3, label1, label2, label3, lev, 
                                     pressureLevel, ymin, totalYears=15):
    plt.figure(figsize=(14, 6))
    
    files = [file1, file2, file3]
    labels = [label1, label2, label3]
    colors = ['blue', 'red', 'green']
    all_data = []
    
    for fname, label, color in zip(files, labels, colors):
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(fname, engine='netcdf4', decode_times=time_coder)
        ua = ds['ua']
        
        ua_at_level = ua.sel(lev=pressureLevel, method='nearest')
        ua_global_mean = ua_at_level.mean(dim=['lat', 'lon'])
        
        # Convert time to years for x-axis
        total_timesteps = len(ua_global_mean)
        time_in_years = np.linspace(0, totalYears, total_timesteps)
        
        plt.plot(time_in_years, ua_global_mean, linewidth=1.5, label=label, color=color, alpha=0.8)
        
        # Store final equilibrated value
        spinup_idx = int(total_timesteps / totalYears)
        equilibrated_mean = ua_global_mean.isel(time=slice(spinup_idx, None)).mean().values
        all_data.append({
            'label': label,
            'equilibrated_mean': equilibrated_mean,
            'final_value': ua_global_mean.isel(time=-1).values
        })

        ds.close()
    
    # Formatting
    plt.xlabel('Time [years]', fontsize=13)
    plt.ylabel('Eastward (u) Wind Velocity [m/s]', fontsize=13)
    plt.title(f'Eastward (u) Wind Velocity Time Evolution at {pressureLevel/100:.0f} hPa, LEV = {lev}', 
                fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(bottom=ymin)
    plt.tight_layout()
    plt.savefig(f'ua_comparison_{pressureLevel/100:.0f}hPa_LEV{lev}.png', dpi=300)
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


def create_northward_wind_graph(file1, file2, file3, label1, label2, label3, lev, 
                                     pressureLevel, ymin, totalYears=15):
    plt.figure(figsize=(14, 6))
    
    files = [file1, file2, file3]
    labels = [label1, label2, label3]
    colors = ['blue', 'red', 'green']
    all_data = []
    
    for fname, label, color in zip(files, labels, colors):
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(fname, engine='netcdf4', decode_times=time_coder)
        va = ds['va']
        
        va_at_level = va.sel(lev=pressureLevel, method='nearest')
        va_global_mean = va_at_level.mean(dim=['lat', 'lon'])
        
        # Convert time to years for x-axis
        total_timesteps = len(va_global_mean)
        time_in_years = np.linspace(0, totalYears, total_timesteps)
        
        plt.plot(time_in_years, va_global_mean, linewidth=1.5, label=label, color=color, alpha=0.8)
        
        # Store final equilibrated value
        spinup_idx = int(total_timesteps / totalYears)
        equilibrated_mean = va_global_mean.isel(time=slice(spinup_idx, None)).mean().values
        all_data.append({
            'label': label,
            'equilibrated_mean': equilibrated_mean,
            'final_value': va_global_mean.isel(time=-1).values
        })

        ds.close()
    
    # Formatting
    plt.xlabel('Time [years]', fontsize=13)
    plt.ylabel('Northward (v) Wind Velocity [m/s]', fontsize=13)
    plt.title(f'Northward (v) Wind Velocity Time Evolution at {pressureLevel/100:.0f} hPa, LEV = {lev}', 
                fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(bottom=ymin)
    plt.tight_layout()
    plt.savefig(f'va_comparison_{pressureLevel/100:.0f}hPa_LEV{lev}.png', dpi=300)
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

def create_wind_difference_plot(fileName_u_ref, fileName_v_ref,
                                fileName_u_exp, fileName_v_exp,
                                title, pressureLevel, lev, totalYears=15):

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    # Load datasets
    dsu_ref = xr.open_dataset(fileName_u_ref, engine='netcdf4', decode_times=time_coder)
    dsv_ref = xr.open_dataset(fileName_v_ref, engine='netcdf4', decode_times=time_coder)
    dsu_exp = xr.open_dataset(fileName_u_exp, engine='netcdf4', decode_times=time_coder)
    dsv_exp = xr.open_dataset(fileName_v_exp, engine='netcdf4', decode_times=time_coder)

    ua_ref = dsu_ref['ua']
    va_ref = dsv_ref['va']
    ua_exp = dsu_exp['ua']
    va_exp = dsv_exp['va']

    total_timesteps = len(ua_ref.time)
    spinup_timesteps = int(total_timesteps / totalYears)

    ua_ref_eq = ua_ref.isel(time=slice(spinup_timesteps, None))
    va_ref_eq = va_ref.isel(time=slice(spinup_timesteps, None))
    ua_exp_eq = ua_exp.isel(time=slice(spinup_timesteps, None))
    va_exp_eq = va_exp.isel(time=slice(spinup_timesteps, None))

    # Time mean
    ua_ref_mean = ua_ref_eq.mean(dim='time')
    va_ref_mean = va_ref_eq.mean(dim='time')
    ua_exp_mean = ua_exp_eq.mean(dim='time')
    va_exp_mean = va_exp_eq.mean(dim='time')

    # Select pressure level
    ua_ref_lev = ua_ref_mean.sel(lev=pressureLevel, method='nearest')
    va_ref_lev = va_ref_mean.sel(lev=pressureLevel, method='nearest')
    ua_exp_lev = ua_exp_mean.sel(lev=pressureLevel, method='nearest')
    va_exp_lev = va_exp_mean.sel(lev=pressureLevel, method='nearest')

    # Differences
    du = ua_exp_lev - ua_ref_lev
    dv = va_exp_lev - va_ref_lev
    diff_speed = np.sqrt(du**2 + dv**2)

    print(f"\nΔWind speed range: {diff_speed.min().values:.3f} – {diff_speed.max().values:.3f} m/s")

    # Plot
    plt.figure(figsize=(18, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Symmetric levels around 0 for signed difference (optional alt below)
    vmax = np.nanmax(np.abs(diff_speed.values))
    levels = np.linspace(0, vmax, 15)

    contour = ax.contourf(
        diff_speed.lon, diff_speed.lat, diff_speed,
        levels=levels, cmap='PuRd',
        transform=ccrs.PlateCarree()
    )

    skip = 3
    quiver = ax.quiver(
        du.lon[::skip], du.lat[::skip],
        du[::skip, ::skip], dv[::skip, ::skip],
        transform=ccrs.PlateCarree(),
        scale=30,
        width=0.003,
        headwidth=3,
        headlength=4,
        color='black',
        alpha=0.8
    )

    ax.quiverkey(quiver, 0.9, 1.05, 0.5, '0.5 m/s change',
                 labelpos='E', coordinates='axes')

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
    cbar.set_label('Δ Wind Speed Magnitude [m/s]', fontsize=11)

    plt.title(f'{title}: Mean Wind Difference at {pressureLevel/100:.0f} hPa (LEV={lev})',
              fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'diff_{fileName_u_exp}_{fileName_v_exp}_{pressureLevel/100:.0f}hPa.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    dsu_ref.close()
    dsv_ref.close()
    dsu_exp.close()
    dsv_exp.close()


create_wind_vector_plot('code_131-ref15',       'code_132-ref15',       'Reference (LEV = 15)',        85000, 15)
create_wind_vector_plot('code_131-lessfric15',  'code_132-lessfric15',  'Less Friction (LEV = 15)',    85000, 15)
create_wind_vector_plot('code_131-morefric15',  'code_132-morefric15',  'More Friction (LEV = 15)',    85000, 15)
create_wind_vector_plot('code_131-lessfric5',   'code_132-lessfric5',   'Less Friction (LEV = 5)',     85000, 5)
create_wind_vector_plot('code_131-lessfric10',  'code_132-lessfric10',  'Less Friction (LEV = 10)',    85000, 10)
create_wind_vector_plot('code_131-morefric5',   'code_132-morefric5',   'More Friction (LEV = 5)',     85000, 5)
create_wind_vector_plot('code_131-morefric10',  'code_132-morefric10',  'More Friction (LEV = 10)',    85000, 10)
create_wind_vector_plot('code_131-ref5',        'code_132-ref5',        'Reference (LEV = 5)',         85000, 5)
create_wind_vector_plot('code_131-ref10',       'code_132-ref10',       'Reference (LEV = 10)',        85000, 10)

create_wind_difference_plot('code_131-ref15', 'code_132-ref15', 'code_131-lessfric15', 'code_132-lessfric15',
                            'Less Friction Difference', 85000, 15)
create_wind_difference_plot('code_131-ref15', 'code_132-ref15', 'code_131-morefric15', 'code_132-morefric15',
                            'More Friction Difference', 85000, 15)

create_eastward_wind_graph('code_131-ref15', 'code_131-lessfric15', 'code_131-morefric15',
                            'Reference', 'Less Friction', 'More Friction', 15, 50000, 6.5)
create_eastward_wind_graph('code_131-ref15', 'code_131-lessfric15', 'code_131-morefric15',
                            'Reference', 'Less Friction', 'More Friction', 15, 85000, 1.5)
create_eastward_wind_graph('code_131-ref15', 'code_131-lessfric15', 'code_131-morefric15',
                            'Reference', 'Less Friction', 'More Friction', 15, 100000, -0.7)

create_northward_wind_graph('code_132-ref15', 'code_132-lessfric15', 'code_132-morefric15',
                            'Reference', 'Less Friction', 'More Friction', 15, 50000, -0.4)
create_northward_wind_graph('code_132-ref15', 'code_132-lessfric15', 'code_132-morefric15',
                            'Reference', 'Less Friction', 'More Friction', 15, 85000, -0.2)
create_northward_wind_graph('code_132-ref15', 'code_132-lessfric15', 'code_132-morefric15',
                            'Reference', 'Less Friction', 'More Friction', 15, 100000, -1)

create_eastward_wind_graph('code_131-ref5', 'code_131-lessfric5', 'code_131-morefric5',
                            'Reference', 'Less Friction', 'More Friction', 5, 50000, 5.5)
create_eastward_wind_graph('code_131-ref5', 'code_131-lessfric5', 'code_131-morefric5',
                            'Reference', 'Less Friction', 'More Friction', 5, 85000, 0.8)
create_eastward_wind_graph('code_131-ref5', 'code_131-lessfric5', 'code_131-morefric5',
                            'Reference', 'Less Friction', 'More Friction', 5, 100000, -0.2)

create_northward_wind_graph('code_132-ref5', 'code_132-lessfric5', 'code_132-morefric5',
                            'Reference', 'Less Friction', 'More Friction', 5, 50000, -0.2)
create_northward_wind_graph('code_132-ref5', 'code_132-lessfric5', 'code_132-morefric5',
                            'Reference', 'Less Friction', 'More Friction', 5, 85000, -0.3)
create_northward_wind_graph('code_132-ref5', 'code_132-lessfric5', 'code_132-morefric5',
                            'Reference', 'Less Friction', 'More Friction', 5, 100000, -0.5)

create_eastward_wind_graph('code_131-ref10', 'code_131-lessfric10', 'code_131-morefric10',
                            'Reference', 'Less Friction', 'More Friction', 10, 50000, 6.5)
create_eastward_wind_graph('code_131-ref10', 'code_131-lessfric10', 'code_131-morefric10',
                            'Reference', 'Less Friction', 'More Friction', 10, 85000, 1.5)
create_eastward_wind_graph('code_131-ref10', 'code_131-lessfric10', 'code_131-morefric10',
                            'Reference', 'Less Friction', 'More Friction', 10, 100000, -0.5)

create_northward_wind_graph('code_132-ref10', 'code_132-lessfric10', 'code_132-morefric10',
                            'Reference', 'Less Friction', 'More Friction', 10, 50000, -0.2)
create_northward_wind_graph('code_132-ref10', 'code_132-lessfric10', 'code_132-morefric10',
                            'Reference', 'Less Friction', 'More Friction', 10, 85000, -0.2)
create_northward_wind_graph('code_132-ref10', 'code_132-lessfric10', 'code_132-morefric10',
                            'Reference', 'Less Friction', 'More Friction', 10, 100000, -1)
