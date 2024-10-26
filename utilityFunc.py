import rasterio
from netCDF4 import Dataset
from rasterio import Affine as A
import numpy as np
from os import listdir, makedirs, remove
from os.path import isfile, join, basename, exists, dirname
import xarray
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import rioxarray as riox
import traceback
import json

# import cartopy.crs as ccrs
from utilityGlobal import (
    EARTH_RADIUS,
)


def handle_user_input(parameters):
    """
    Checks the command line arguments and extracts the directory path and the target shape for the geo data
    Note Valid Command Line Examples:
            - valid command line inputs include
            - python wglc_script.py ./WGLC (90,144)
            - python wglc_script.py ./WGLC 90 144

    :param parameters: a list of all the parameters passed at the command line
    :return: both the directory path and the shape as a tuple
    """
    dir_path = (
        parameters[0]
        if len(parameters) >= 1 and exists(dirname(parameters[0]))
        else input("[*] Please enter the directory path to the file: ")
    )
    new_shape = (
        (int(parameters[1]), int(parameters[2]))
        if len(parameters) >= 3 and parameters[1].isdigit() and parameters[1].isdigit()
        else None
    )
    if new_shape == None:
        scaling_size_height = int(
            input("[*] Please enter the height of the new scale: ")
        )
        scaling_size_width = int(input("[*] Please enter the width of the new scale: "))
        new_shape = (scaling_size_height, scaling_size_width)
    return (dir_path, new_shape)


def plot_geodata(matrix, title, footer) -> None:
    """
    Saves the xarray dataset based on the file inputted to the function

    :param file_path: file path of the current file being upscaled/processed
    :param data_set: data set representing the
    :return: None
    """
    plt.style.use("dark_background")
    latitudes = np.linspace(-90, 90, matrix.shape[0])
    longitudes = np.linspace(-180, 180, matrix.shape[1])
    plt.figure(figsize=(10, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.contourf(longitudes, latitudes, matrix)
    ax.set_title((title + " " + footer), y=-0.01)
    return


def obtain_netcdf_files(dir_path) -> list:
    """
    loops through files in the current director and returns a list of files that are netcdf files

    :param dir_path: the file path
    :return: all files in the "dir_path" that are netcdf files
    """
    return [
        join(dir_path, file)
        for file in listdir(dir_path)
        if isfile(join(dir_path, file))
        and (file.split(".")[-1] == "hdf5" or file.split(".")[-1] == "nc")
    ]


def calculate_grid_area(grid_area_shape):
    # Grid resolution
    nlat = grid_area_shape[0]  # Number of latitude bands
    nlon = grid_area_shape[1]  # Number of longitude bands

    # Latitude and longitude step size (degrees)
    lat_step = 180 / nlat
    lon_step = 360 / nlon

    # Convert step size to radians
    lat_step_rad = np.deg2rad(lat_step)
    lon_step_rad = np.deg2rad(lon_step)

    # Initialize grid cell area matrix
    grid_area = np.zeros((nlat, nlon))

    # Loop over each latitude band
    for i in range(nlat):
        # Latitude at the center of the grid cell
        lat = -90 + (i + 0.25) * lat_step

        # Convert latitude to radians
        lat_rad = np.deg2rad(lat)

        # Calculate the surface area of the grid cell at this latitude
        area = (
            (EARTH_RADIUS**2)
            * lon_step_rad
            * (np.sin(lat_rad + lat_step_rad / 2) - np.sin(lat_rad - lat_step_rad / 2))
        )

        # Assign the area to all longitude cells for this latitude band
        grid_area[i, :] = area

    # Display the grid area matrix
    return grid_area


def create_geotiff_file(
    data_arr, latitude_arr, longitude_arr, save_folder_path, crs="EPSG:3857"
):
    """
    Creates a new geotiff file to be used for resampling or displaying data on the map

    :param data_arr: numpy array containing the geo data
    :param latitude_arr: numpy array representing the latitude
    :param longitude_arr: numpy array representing the longitude
    :return: geotiff file path
    """
    # obtain the data_arr shape
    height, width = data_arr.shape
    # create a transformation of the data to match a global map
    transform = from_origin(
        longitude_arr[0],
        latitude_arr[-1],
        abs(longitude_arr[1] - longitude_arr[0]),
        abs(latitude_arr[-1] - latitude_arr[-2]),
    )

    # outline meta data about the geotiff file
    metadata = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "width": width,
        "height": height,
        "crs": crs,  # optional formats EPSG:3857 (works on panoply) EPSG:4326 (works well on leaflet)
        "transform": transform,
    }

    # obtain the GeoTIFF path
    geotiff_file_path = join(save_folder_path, "output.tif")
    # Create a new GeoTIFF file using the crafted path and add the data to the file
    with rasterio.open(geotiff_file_path, "w", **metadata) as dst:
        # total_data_value = np.flip(data_arr, 0)
        dst.write(data_arr, 1)
    # return the GeoTIFF file path
    return geotiff_file_path


def obtain_variables(netcdf_dataset):
    # obtains the variable names that we care about from the current netcdf dataset
    files_gfed5_variable_names = [
        var_name for var_name in netcdf_dataset.variables.keys()
    ]
    return files_gfed5_variable_names


def evaluate_upscale_sum(origin_matrix, upscaled_matrix, margin_of_error=65536.0):
    """
    Function prints our the original matrix sum and the upscaled matrix sum (post re grid)
    It then determines if the upscaled matrix sum is close enough to the original matrix sum (incorporating a margin of error)

    :param origin_matrix: original matrix before re grid
    :param upscaled_matrix: the original matrix post re grid
    :param margin_of_error: the margin of error the allowed
    :return: boolean
    """
    print()
    print(f"Original Burned Area Total - {origin_matrix.sum()}")
    print(f"\tOriginal Burned Area Dimensions - {origin_matrix.shape}")
    print(f"\tOriginal Burned Area Mean - {origin_matrix.mean()}")
    print(f"Upscale Burned Area Total - {upscaled_matrix.sum()}")
    print(f"\tUpscale Burned Area Dimensions - {upscaled_matrix.shape}")
    print(f"\tUpscale Burned Area Mean - {origin_matrix.mean()}")
    print()

    # returns true if the upscaled matrix sum is within the range of the original matrix sum (margin of error accounts for rounding of values)
    return abs(origin_matrix.sum() - upscaled_matrix.sum()) <= margin_of_error


def obtain_new_filename(file_path, save_folder_path, dest_shape) -> str:
    # creates a file name (adding upscale to the current file name)
    file_name = basename(file_path)
    file_name_list = file_name.split(".")
    if len(file_name_list) > 1:
        file_name_list[-2] = file_name_list[-2] + f"_{dest_shape[0]}{dest_shape[1]}"
        # ensures the file is saved as a netcdf file
        file_name_list[-1] = "nc"
        # return the rejoined list and the added classes save folder path
        return join(save_folder_path, ".".join(file_name_list))
    return join(save_folder_path, file_name)


def save_file(file_path, data_set, save_folder_path, dest_shape) -> None:
    """
    Saves the xarray dataset based on the file inputted to the function

    :param file_path: file path of the current file being upscaled/processed
    :param data_set: data set representing the
    :return: None
    """
    try:
        # create the new file's path & name
        new_file_name = obtain_new_filename(
            file_path=file_path,
            save_folder_path=save_folder_path,
            dest_shape=dest_shape,
        )
        # checks if the save folder path exists (if it does not a folder is created)
        if not exists(save_folder_path):
            makedirs(save_folder_path)
        # saves the file using the created file path and xarray
        data_set.to_netcdf(path=(new_file_name))
        print(f"[+] file {new_file_name} saved")
    except Exception as error:
        print(
            "[-] Failed to save dataset (ensure dataset is from xarray lib): ",
            error,
        )


def obtain_variables_gfed5(netcdf_dataset, gfed5_variable_names):
    # obtains the variable names that we care about from the current netcdf dataset
    files_gfed5_variable_names = [
        var_name
        for var_name in netcdf_dataset.variables.keys()
        if (var_name in gfed5_variable_names)
    ]
    if set(files_gfed5_variable_names) == set(gfed5_variable_names):
        files_gfed5_variable_names.append("Nat")
    return files_gfed5_variable_names


def resample_matrix(source_matrix, dest_shape, geotiff_output_path):
    """
    Function preforms the process of upscaling the passed in matrix using rasterio and geotiff

    :param source_matrix: matrix we wish to compress (upscale)
    :param dest_dimensions: a tuple/list containing the dimensions you want to transform the matrix
    :return: reshaped numpy matrix
    """
    # https://github.com/corteva/rioxarray/discussions/332
    # Obtain the numpy array shape
    height, width = source_matrix.shape
    # create a long and latitude numpy array
    latitude_arr = np.linspace(-90, 90, height)
    longitude_arr = np.linspace(-180, 180, width)

    # create the geotiff file and return the path to that file
    geotiff_file_path = create_geotiff_file(
        source_matrix, latitude_arr, longitude_arr, geotiff_output_path
    )

    # open that newly created geotiff file
    raster = riox.open_rasterio(geotiff_file_path)

    # preform upsampling using rasterio and rioxarray
    up_sampled = raster.rio.reproject(
        raster.rio.crs,
        shape=(int(dest_shape[0]), int(dest_shape[1])),
        resampling=rasterio.warp.Resampling.sum,
    )

    # obtain the data
    data_value = up_sampled.values[0]
    # close the file (script will yell at you if you dont)
    raster.close()
    # remove the geotiff file
    if exists(geotiff_file_path):
        remove(geotiff_file_path)
        print(f"[o] Successfully Removed Geotiff file: {geotiff_file_path}")
    else:
        print(f"[-] Failed to Remove Geotiff file: {geotiff_file_path}")
    # return numpy data array
    return data_value
