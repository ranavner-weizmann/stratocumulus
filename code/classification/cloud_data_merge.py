import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

'''

This script is used to take the csv created by the FFT.py (which analyses the iamges and saving the parameters in csv_data_fft.csv) and merge the data downloaded
from era5 to the csv.

'''

###
def merge_era5_cloud_data():

    df = pd.read_csv('cloud_data_fft.csv')


    for nc in os.listdir('era5_single_level'):
        nc = os.path.join('era5_single_level', nc)
        ds = xr.open_dataset(nc)

        # ---------- wind ----------
        wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
        # calculating the mean wind speed
        mean_wind_speed = wind_speed.mean()
        print(f'mean wind speed: {mean_wind_speed.values:.2f} m/s')
        # calculating the mean wind direction
        mean_wind_direction = np.arctan2(ds['v10'], ds['u10']).mean()
        print(f'mean wind direction (radian): {mean_wind_direction.values:.2f} rad')
        # converting the wind direction to degrees
        mean_wind_direction = np.rad2deg(mean_wind_direction)
        print(f'mean wind direction (degrees):{mean_wind_direction.values:.2f} degrees')

        # ---------- 2m_temperature ----------
        mean_t2m = ds['t2m'].mean()
        # coveriting to Celsius
        mean_t2m = mean_t2m - 273.15
        print(f't2m: {mean_t2m.values:.2f} Celsius')

        # ---------- sea_surface_temperature ----------
        mean_sst = ds['sst'].mean()
        # coveriting to Celsius
        mean_sst = mean_sst - 273.15
        print(f'sst: {mean_sst.values:.2f} Celsius')

        # ---------- boundary_layer_height -----------
        mean_blh = ds['blh'].mean()
        print(f'blh: {mean_blh.values:.2f} Meters')

        # ---------- total_column_cloud_liquid_water -----------
        mean_clw = ds['tcc'].mean()
        print(f'tcc: {mean_clw.values:.2f} kg/m^2')


        index = nc.split('\\')[1].split('.')[0] # This way we make sure that the index is the same as the name in the 'cloud_data.csv' file

        # Importent to note that this split method is only valid for windows, for linux or mac you should use '/' instead of '\\'

        # adding the different variables to the the 'cloud_data.csv' file based on the index
        
        # The nc file's names are different then those in the csv so we need to fix that
        NAME_PATTERN = re.compile(r"(\d{4})_(\d{1,2})_(\d{1,2})_(\-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)")
        
        fixed_name = NAME_PATTERN.match(index)
        if len(fixed_name.group(2)) < 2:
            corrected_month = f'{int(fixed_name.group(2)):02}'
        else:
            corrected_month = fixed_name.group(2)
        if len(fixed_name.group(3)) < 2:
            corrected_day = f'{int(fixed_name.group(3)):02}'
        else:
            corrected_day = fixed_name.group(3)

        index = f'{(fixed_name.group(1))}{corrected_month}{corrected_day}-{abs(int(fixed_name.group(4)))}-{abs(int(fixed_name.group(5)))}-{abs(int(fixed_name.group(6)))}-{abs(int(fixed_name.group(7)))}'
        print(f'index: {index}')
        if index in df['name'].values:

            df.loc[df['name'] == index, 'mean_2m_temperature'] = mean_t2m.values
            df.loc[df['name'] == index, 'mean_sst'] = mean_sst.values
            df.loc[df['name'] == index, 'mean_wind_speed'] = mean_wind_speed.values
            df.loc[df['name'] == index, 'mean_wind_direction'] = mean_wind_direction.values
            df.loc[df['name'] == index, 'boundry_layer_height'] = mean_blh.values
            df.loc[df['name'] == index, 'total_column_cloud_liquid_water'] = mean_clw.values
        else:
            print(f"Index {index} not found in the 'name' column.")


    # saving the updated 'cloud_data.csv' file
    df.to_csv('cloud_data_fft.csv', index=False)

    print(df)

merge_era5_cloud_data()


