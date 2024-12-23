﻿# Grid Rescaling Scripts

Collection of scripts meant to regrid/resample various data across the domain of environmental science.

## Usage

Below is an example of how to run each individual program.

```console
example@example:~$ python <python script> <data path> <dest shape>

example@example:~$ python ba_script.py ./BA 90 144

```

# Breakdown

### UtilityFunc.py - Contains the basic functions needed to run and execute each regridding script
  - [x] _obtain_netcdf_files_ - File Extraction based on a folder path
  - [x] _create_geotiff_file_ - Generates and deletes a Geotiff file that is used for preforming resampling
  - [x] _resample_matrix_ - Resamples the netcdf data using sum calculations allowing for the original data to be roughly maintained
  - [x] _evaluate_upscale_sum_ - Verifies the sum of the original data and the resampled data and returns a boolean based on the difference of the new value (default margin_of_error is 65536)
  - [x] _plot_geodata_ - plots the netcdf data on a visually pleasing map so that users can quickly view the original data and resampled data more swiftly
  - [x] _save_file_ - saves the newly resampled netcdf data to a dynamically created folder known as upscale so that all your resampled data is seperated from the original data



## Contributing

For changes or suggestions feel free to reach out.


