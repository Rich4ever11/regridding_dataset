import rasterio
from netCDF4 import Dataset
from rasterio import Affine as A
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, basename, exists
import xarray
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import rioxarray as riox
import traceback


def create_geotiff_file(
    data_arr, latitude_arr, longitude_arr, save_folder_path, crs="EPSG:3857"
):
    """
    Creates a new geotif file to be used for resampling or displaying data on the map

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
    return upscaled_matrix.sum() >= (
        origin_matrix.sum() - margin_of_error
    ) and upscaled_matrix.sum() <= (origin_matrix.sum() + margin_of_error)


def obtain_new_filename(file_path, save_folder_path) -> str:
    # creates a file name (adding upscale to the current file name)
    file_name = basename(file_path)
    file_name_list = file_name.split(".")
    if len(file_name_list) > 1:
        file_name_list[-2] = file_name_list[-2] + "(upscaled)"
        # ensures the file is saved as a netcdf file
        file_name_list[-1] = "nc"
        # return the rejoined list and the added classes save folder path
        return join(save_folder_path, ".".join(file_name_list))
    return join(save_folder_path, file_name)


def save_file(file_path, data_set, save_folder_path) -> None:
    """
    Saves the xarray dataset based on the file inputted to the function

    :param file_path: file path of the current file being upscaled/processed
    :param data_set: data set representing the
    :return: None
    """
    try:
        # create the new file's path & name
        new_file_name = obtain_new_filename(file_path)
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


def obtain_variables(netcdf_dataset, gfed5_variable_names):
    # obtains the variable names that we care about from the current netcdf dataset
    files_gfed5_variable_names = [
        var_name
        for var_name in netcdf_dataset.variables.keys()
        if (var_name in gfed5_variable_names)
    ]
    if set(files_gfed5_variable_names) == set(gfed5_variable_names):
        files_gfed5_variable_names.append("Nat")
    return files_gfed5_variable_names


def upscale_matrix_restario(source_matrix, dest_dimensions):
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
    geotiff_file_path = create_geotiff_file(source_matrix, latitude_arr, longitude_arr)

    # open that newly created geotiff file
    raster = riox.open_rasterio(geotiff_file_path)

    # preform upsampling using rasterio and rioxarray
    up_sampled = raster.rio.reproject(
        raster.rio.crs,
        shape=(int(dest_dimensions[0]), int(dest_dimensions[1])),
        resampling=rasterio.warp.Resampling.sum,
    )

    # obtain the data
    data_value = up_sampled.values[0]
    # close the file (script will yell at you if you dont)
    raster.close()
    # return numpy data array
    return data_value


def upscale_gfed4_data(files, new_shape=(90, 144)):
    """
    loops through each file in the classes files list Regridding (upscaling) datasets from a fine resolution to a coarse (ModelE) resolution
    Note - This is focused on the burned area dataset and uses both netcdf (parsing/reading) and xarray (saving the data)
        Issue (SOLVED) - When saving the dataset the unscaled burned area is classified as a 2D variable instead of a Geo2D variable

    :param: None
    :return: None
    """
    for file in files:
        try:
            with Dataset(file) as netcdf_dataset:
                # dataset containing all xarray data array (used to create the final netcdf file)
                dataset_dict = {}

                # obtain the grid cell area value (allows for the burned area to account for the shape of the earth)
                grid_cell_area_value = netcdf_dataset.groups["ancill"].variables[
                    "grid_cell_area"
                ][:]

                # loop through every burned area month
                for group in netcdf_dataset.groups["burned_area"].groups:
                    # obtain the current burned area group
                    burned_area_group = netcdf_dataset.groups["burned_area"].groups[
                        group
                    ]

                    # obtain the burned_area fraction array for the current month/group we are in
                    burned_area_fraction = burned_area_group.variables[
                        "burned_fraction"
                    ]
                    burned_area_fraction_value = burned_area_fraction[:]

                    # multiplying the grid cell area by the burned fraction value
                    burned_fraction_product = (
                        grid_cell_area_value * burned_area_fraction_value
                    )
                    burned_fraction_product = np.asarray(burned_fraction_product)

                    # upscale the burned fraction
                    burned_fraction_upscaled = upscale_matrix_restario(
                        burned_fraction_product, dest_dimensions=new_shape
                    )

                    # Total of orig resolution after multiplying by gridcell area should be equal to total of final (target) resolution. Both are in m^2.
                    if evaluate_upscale_sum(
                        burned_fraction_product, burned_fraction_upscaled
                    ):
                        burnded_area_attribute_dict = {}

                        # Copy attributes of the burned area fraction
                        for attr_name in burned_area_fraction.ncattrs():
                            burnded_area_attribute_dict[attr_name] = getattr(
                                burned_area_fraction, attr_name
                            )

                        # update the units to match the upscaling process
                        burnded_area_attribute_dict["units"] = "m^2"

                        # obtain the height and width from the upscale shape
                        # create an evenly spaced array representing the longitude and the latitude
                        latitudes = np.linspace(-90, 90, new_shape[0])
                        longitudes = np.linspace(-180, 180, new_shape[1])

                        # plots the burned area before and after the rescale
                        # plot_geodata(
                        #     burned_fraction_product,
                        #     burned_fraction_upscaled,
                        #     longitudes,
                        #     latitudes,
                        # )
                        # flip the data matrix (upside down due to the GFED dataset's orientation)
                        burned_fraction_upscaled = np.flip(burned_fraction_upscaled, 0)

                        # create the xarray data array for the upscaled burned area and add it to the dictionary
                        burned_area_data_array = xarray.DataArray(
                            burned_fraction_upscaled,
                            coords={"latitude": latitudes, "longitude": longitudes},
                            dims=["latitude", "longitude"],
                            attrs=burnded_area_attribute_dict,
                        )
                        dataset_dict[f"burned_areas_{group}"] = burned_area_data_array

                # saves xarray dataset to a file
                save_file(file, xarray.Dataset(dataset_dict))
        except Exception as error:
            print("[-] Failed to parse dataset: ", error)


def upscale_burned_area_data(files, new_shape=(90, 144)) -> None:
    """
    loops through each file in the classes files list Regridding (upscaling) datasets from a fine resolution to a coarse (ModelE) resolution
    Note - This is focused on the burned area dataset and uses both netcdf (parsing/reading) and xarray (saving the data)
        Issue (SOLVED) - When saving the dataset the unscaled burned area is classified as a 2D variable instead of a Geo2D variable

    :param: None
    :return: None
    """
    for file in files:
        with Dataset(file) as netcdf_dataset:
            try:
                # dataset containing all xarray data array (used to create the final netcdf file)
                dataset_dict = {}
                # obtains the variable names that we care about from the current netcdf dataset
                files_gfed5_variable_names = obtain_variables(netcdf_dataset)
                for variable_name in files_gfed5_variable_names:
                    match variable_name:
                        # calculates the Nat array
                        case "Nat":
                            # transform the arrays dimensions to (720, 1440) and convert (km^2 -> m^2)
                            # obtain all needed data array
                            var_total_data_array = netcdf_dataset.variables["Total"][:][
                                0
                            ] * (10**6)
                            var_crop_data_array = netcdf_dataset.variables["Crop"][:][
                                0
                            ] * (10**6)
                            var_defo_data_array = netcdf_dataset.variables["Defo"][:][
                                0
                            ] * (10**6)
                            var_peat_data_array = netcdf_dataset.variables["Peat"][:][
                                0
                            ] * (10**6)
                            # calculate the Nat numpy array
                            # equation: Total - (Crop + Defo + Peat)
                            var_data_array = var_total_data_array - (
                                var_crop_data_array
                                + var_defo_data_array
                                + var_peat_data_array
                            )
                        # base case
                        case _:
                            # obtain the variables in the netcdf_dataset
                            # dimensions (1, 720, 1440)
                            var_data = netcdf_dataset.variables[variable_name]

                            # obtain the numpy array for each netcdf variable
                            # transform the arrays dimensions to (720, 1440) and convert the metric to km^2 -> m^2
                            var_data_array = var_data[:][0] * KM_TO_M

                    # preform resampling/upscaling
                    # Conversion (720, 1440) -> (90, 144)
                    upscaled_var_data_array = upscale_matrix_restario(
                        var_data_array, dest_dimensions=new_shape
                    )

                    attribute_dict = {}

                    # Copy attributes of the burned area fraction
                    for attr_name in var_data.ncattrs():
                        attribute_dict[attr_name] = getattr(var_data, attr_name)

                    # update the units to match the upscaling process
                    attribute_dict["units"] = "m^2"

                    # obtain the height and width from the upscale shape
                    # create an evenly spaced array representing the longitude and the latitude
                    latitudes = np.linspace(-90, 90, new_shape[0])
                    longitudes = np.linspace(-180, 180, new_shape[1])

                    # plots the burned area before and after the rescale
                    # plot_geodata(var_data_array, upscaled_var_data_array, longitudes, latitudes)

                    # create the xarray data array for the upscaled burned area and add it to the dictionary
                    burned_area_data_array = xarray.DataArray(
                        upscaled_var_data_array,
                        coords={"latitude": latitudes, "longitude": longitudes},
                        dims=["latitude", "longitude"],
                        attrs=attribute_dict,
                    )
                    dataset_dict[variable_name] = burned_area_data_array
                # saves xarray dataset to a file
                save_file(file, xarray.Dataset(dataset_dict))
            except Exception as error:
                print("[-] Failed to parse dataset: ", error)

