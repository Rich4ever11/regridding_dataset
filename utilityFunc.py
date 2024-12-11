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
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# import cartopy.crs as ccrs
from utilityGlobal import EARTH_RADIUS_KM, EARTH_RADIUS_METERS


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


def time_series_plot(
    axis,
    data,
    marker,
    line_style,
    color,
    label,
    axis_title,
    axis_xlabel,
    axis_ylabel,
    grid_visible=True,
):
    """
    Plots the total burned area as a function of year for both GFED and ModelE data.

    Parameters:
    gfed4sba_per_year (np.ndarray): A 2D array with columns (year, totalBA), where totalBA is the sum of burned area for that year.
    modelE_BA_per_year (np.ndarray): A 2D array with columns (year, modelE_BA), where modelE_BA is the sum of burned area for that year.
    """

    # try:
    # Extract years and total burned area for both GFED and ModelE
    years_data = data[:, 0]
    total_data = data[:, 1]

    # Plot the time series of total burned area for both GFED and ModelE
    axis.plot(
        years_data,
        total_data,
        marker=marker,
        linestyle=line_style,
        color=color,
        label=label,
    )
    axis.legend()
    axis.grid(grid_visible)
    axis.set_title(axis_title)
    axis.set_xlabel(axis_xlabel)
    axis.set_ylabel(axis_ylabel)
    # except:
    #     print("title, xlabel...etc already set")


def define_subplot(
    fig,
    ax,
    decade_data,
    lons,
    lats,
    cmap,
    cborientation,
    fraction,
    pad,
    labelpad,
    fontsize,
    title,
    clabel,
    masx=None,
    is_diff=False,
    glob=None,
):
    masx = 0.7 * decade_data.max() if masx == None else masx
    # labelpad sets the distance of the colorbar from the map
    """Define the properties of a subplot with optional difference normalization."""
    ax.coastlines(color="black")
    ax.add_feature(cfeature.LAND, edgecolor="gray")
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=1)

    ax.set_title(title, fontsize=10, pad=1)
    props = dict(boxstyle="round", facecolor="lightgray", alpha=0.5)
    (
        (
            ax.text(
                0.5,
                1.07,
                f"Global Total: {glob}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=props,
                fontsize=10,
            )
        )
        if glob
        else None
    )

    # Handling difference normalization (if is_diff is true)
    if is_diff:
        data_min, data_max = decade_data.min(), decade_data.max()
        print(data_min, data_max)
        if data_min == data_max:
            norm = mcolors.Normalize(vmin=data_min - 1, vmax=data_max + 1)
        else:
            abs_max = max(abs(0.25 * data_min), abs(0.25 * data_max))
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        p = ax.pcolormesh(
            lons,
            lats,
            decade_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            vmin=0 if not is_diff else None,
            vmax=masx if not is_diff else None,
        )
    else:
        norm = None
        # Mask values less than or equal to zero for the custom colormap (set to white)
        # masked_data = np.ma.masked_less_equal(data, 0)  # Mask values <= 0
        # # Create a colormap with white for values <= 0
        # cmap = plt.get_cmap(cmap).copy()
        # cmap.set_bad(color="white")  # Set masked values to white
        print(float(1) if not is_diff else None, float(masx) if not is_diff else None)
        logNorm = mcolors.LogNorm(
            vmin=float(1) if not is_diff else None,
            vmax=float(masx) if not is_diff else None,
        )
        p = ax.pcolormesh(
            lons,
            lats,
            decade_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=logNorm,
            # vmin=0 if not is_diff else None,
            # vmax=masx if not is_diff else None,
        )

    cbar = fig.colorbar(p, ax=ax, orientation=cborientation, fraction=fraction, pad=pad)
    cbar.set_label(f"{clabel}", labelpad=labelpad, fontsize=fontsize)
    return ax


def leap_year_check(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True


def map_plot(
    figure,
    axis,
    axis_length,
    axis_index,
    decade_data,
    longitude,
    latitude,
    subplot_title,
    units,
    cbarmax,
):
    """
    Plots the decadal mean burned area of both GFED and ModelE side by side.

    Parameters:
    decade_mean_gfed4sba (xarray.DataArray): The decadal mean burned area (lat, lon array).
    decade_mean_modelEba (xarray.DataArray): The decadal mean burned area from ModelE(lat, lon array).
    """

    axis_value = axis if axis_length <= 1 else axis[axis_index]
    # GFED4s decadal mean map
    define_subplot(
        figure,
        axis_value,
        decade_data,
        longitude,
        latitude,
        cmap="jet",
        cborientation="horizontal",
        fraction=0.05,
        pad=0.005,
        labelpad=0.5,
        fontsize=10,
        title=subplot_title,
        clabel=units,
        masx=cbarmax,
        is_diff=False,
    )


def draw_map(
    map_figure, map_axis, units, label, latitude, longitude, var_data_xarray, cbarmax
):
    # time_total_data = var_data_xarray.sum(dim=var_data_xarray.dims[0])
    map_plot(
        figure=map_figure,
        axis=map_axis,
        axis_length=0,
        axis_index=0,
        decade_data=var_data_xarray,
        longitude=longitude,
        latitude=latitude,
        subplot_title=label,
        units=units,
        cbarmax=cbarmax,
    )


# pass earth radius into the function params
def calculate_grid_area(grid_area_shape, units="km"):
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

        earth_radius = EARTH_RADIUS_KM if units == "km" else EARTH_RADIUS_METERS

        # Calculate the surface area of the grid cell at this latitude
        area = (
            (earth_radius**2)
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
    print(f"Original Total - {origin_matrix.sum()}")
    print(f"\tOriginal Dimensions - {origin_matrix.shape}")
    print(f"\tOriginal Mean - {origin_matrix.mean()}")
    print(f"Upscale Total - {upscaled_matrix.sum()}")
    print(f"\tUpscale Dimensions - {upscaled_matrix.shape}")
    print(f"\tUpscale Mean - {origin_matrix.mean()}")
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
