import os

# # this code will check the netcdf files for the correct dimensions and variables and will plot it 

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# # Load the netcdf file
ds = xr.open_dataset('era5_single_level/2019_8_1_-10_-9_-93_-92.nc')

# Check the dimensions
print(ds)
print(ds.dims)

# Check the variables
print(ds.data_vars)

# Plot the temperature data
# ds['blh'].plot()

# Show the plot
plt.show()

