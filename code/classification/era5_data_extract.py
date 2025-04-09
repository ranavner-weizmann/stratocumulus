import cdsapi
import cfgrib
import xarray as xr
import logging
import os
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(15), wait=wait_fixed(20))  # Retry 3 times with a 2-second delay
def extract_data(): # looping through the data and extracting the data for each row

    df = pd.read_csv('cloud_data.csv')

    url = 'https://cds.climate.copernicus.eu/api'
    key = 'a7b596de-4128-4b83-b696-18a5d32b95de'
    client = cdsapi.Client(url=url, key=key)
    dataset = 'reanalysis-era5-single-levels' 

    # creating a directory that will contain the data
    if not os.path.exists('era5_single_level'):
        os.makedirs('era5_single_level')



    for index, row in df.iterrows():

        if f'{row["year"]}_{row['month']}_{row['day']}_{row['minlat']}_{row['maxlat']}_{row['minlon']}_{row['maxlon']}.nc' in os.listdir('era5_single_level'):
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "sea_surface_temperature",
                    "boundary_layer_height",
                    "total_column_cloud_liquid_water"
                ],
            "year": [row['year']],
            "month": [row['month']],
            "day": [row['day']],
            "time": ["13:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [row['maxlat'], row['minlon'], row['minlat'], row['maxlon']]
        }

        target = f'era5_single_level/{row["year"]}_{row['month']}_{row['day']}_{row['minlat']}_{row['maxlat']}_{row['minlon']}_{row['maxlon']}.nc'
        client.retrieve(dataset, request, target)

        total_files = len(df)
        processed_files = index + 1
        progress = (processed_files / total_files) * 100
        print(f"Progress for step 2: {progress:.2f}%")


extract_data()

