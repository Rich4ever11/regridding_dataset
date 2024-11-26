from netCDF4 import Dataset
import traceback
import numpy as np
from os import listdir, makedirs, remove
from os.path import isfile, join, basename, exists, dirname
from rasterio.transform import from_origin
import xarray
import rasterio
import rioxarray as riox
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
from utilityGlobal import INIT_YEAR
from utilityFunc import (
    handle_user_input,
    obtain_netcdf_files,
    evaluate_upscale_sum,
    resample_matrix,
    time_series_plot,
    save_file,
    draw_map,
)


class GeoDataResizePopulationDensity:
    """
    GeoDataResize is a class that is meant to manipulate and change the format of geodata, improving the overall analysis of geo data (currently netcdf)

    :Attribute files: the list of files that the user would like to manipulate (currently only netcdf geo data)
    """

    def __init__(self, dir_path, new_shape) -> None:
        """
        initializes the class and sets the files

        :param dir_path: the path of the files we wish to parse
        :return: None
        """
        self.files = obtain_netcdf_files(dir_path)
        self.save_folder_path = join(dir_path, "upscale")
        self.dest_shape = new_shape
        if not exists(self.save_folder_path):
            makedirs(self.save_folder_path)

    def upscale_data(self) -> None:
        """
        loops through each file in the classes files list Regridding (upscaling) datasets from a fine resolution to a coarse (ModelE) resolution
        Note - This is focused on the burned area dataset and uses both netcdf (parsing/reading) and xarray (saving the data)
            Issue (SOLVED) - When saving the dataset the unscaled burned area is classified as a 2D variable instead of a Geo2D variable

        :param: None
        :return: None
        """
        # _, time_analysis_axis = plt.subplots(figsize=(10, 6))
        upscale_sum_values = []
        original_sum_values = []
        for file in self.files:
            try:
                with Dataset(file) as netcdf_dataset:
                    # dataset containing all xarray data array (used to create the final netcdf file)
                    dataset_dict = {}

                    var_data = netcdf_dataset.variables["total-population"]
                    var_data_array = netcdf_dataset.variables["total-population"][:]
                    time_data_array = netcdf_dataset.variables["time"][:]

                    original_var_data = []
                    upscaled_var_data = []
                    for year in range(len(var_data_array)):
                        curr_year_data_array = var_data_array[year]

                        # masked values would disturb the data during resampling
                        # replacing all masked values with 0
                        curr_year_data_array[curr_year_data_array.mask] = 0

                        # preform resampling/upscaling using rasterio
                        # Conversion (720, 1440) -> (90, 144)

                        upscaled_curr_year_data_array = resample_matrix(
                            curr_year_data_array, self.dest_shape, self.save_folder_path
                        )
                        # prints the current year the code is parsing
                        print(f"total-population_year_{INIT_YEAR + year}")

                        evaluate_upscale_sum(
                            curr_year_data_array, upscaled_curr_year_data_array
                        )
                        # flip the data matrix (upside down due to the GFED dataset's orientation)
                        upscaled_curr_year_data_array = np.flip(
                            upscaled_curr_year_data_array, 0
                        )

                        curr_year_data_array = np.flip(curr_year_data_array, 0)

                        # adding the empty values back (this is not required )
                        # upscaled_curr_year_data_array[
                        #     upscaled_curr_year_data_array == 0
                        # ] = np.nan

                        upscale_sum_values.append(upscaled_curr_year_data_array.sum())
                        original_sum_values.append(curr_year_data_array.sum())

                        # create the xarray data array for the upscaled burned area and add it to the dictionary
                        upscaled_var_data.append(
                            upscaled_curr_year_data_array,
                        )
                        original_var_data.append(curr_year_data_array)

                    attribute_dict = {}

                    # Copy attributes of the burned area fraction
                    for attr_name in var_data.ncattrs():
                        attribute_dict[attr_name] = getattr(var_data, attr_name)

                    map_figure_upscale, map_axis_upscale = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(18, 10),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )

                    # obtain the height and width from the upscale shape
                    # create an evenly spaced array representing the longitude and the latitude
                    latitudes = np.linspace(-90, 90, self.dest_shape[0])
                    longitudes = np.linspace(-180, 180, self.dest_shape[1])

                    # creates the data array
                    xarray_data_array = xarray.DataArray(
                        np.asarray(upscaled_var_data),
                        coords={
                            "time": time_data_array,
                            "latitude": latitudes,
                            "longitude": longitudes,
                        },
                        dims=["time", "latitude", "longitude"],
                        attrs=attribute_dict,
                    )

                    draw_map(
                        map_figure=map_figure_upscale,
                        map_axis=map_axis_upscale,
                        units=attribute_dict["units"],
                        label="Upscaled Population Density Data",
                        latitude=latitudes,
                        longitude=longitudes,
                        var_data_xarray=xarray_data_array,
                        cbarmac=None,
                    )

                    # obtain the height and width from the original shape
                    # create an evenly spaced array representing the longitude and the latitude

                    map_figure_original, map_axis_original = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(18, 10),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )

                    longitudes = np.linspace(
                        -180, 180, np.asarray(original_var_data).shape[-1]
                    )
                    latitudes = np.linspace(
                        -90, 90, np.asarray(original_var_data).shape[-2]
                    )

                    # creates the data array
                    original_xarray_data_array = xarray.DataArray(
                        np.asarray(original_var_data),
                        coords={
                            "time": time_data_array,
                            "latitude": latitudes,
                            "longitude": longitudes,
                        },
                        dims=["time", "latitude", "longitude"],
                        attrs=attribute_dict,
                    )

                    draw_map(
                        map_figure=map_figure_original,
                        map_axis=map_axis_original,
                        units=attribute_dict["units"],
                        label="Original Population Density Data",
                        latitude=latitudes,
                        longitude=longitudes,
                        var_data_xarray=original_xarray_data_array,
                        cbarmac=None,
                    )

                    dataset_dict["total-population"] = xarray_data_array
                    # saves xarray dataset to a file
                    save_file(
                        file_path=file,
                        data_set=xarray.Dataset(dataset_dict),
                        save_folder_path=self.save_folder_path,
                        dest_shape=self.dest_shape,
                    )
            except Exception as error:
                print("[-] Failed to parse dataset: ", error)
                print(traceback.format_exc())

        # data_yearly_upscale = np.column_stack(
        #     (
        #         np.arange(
        #             0,
        #             len(upscale_sum_values),
        #         ),
        #         upscale_sum_values,
        #     )
        # )

        # time_series_plot(
        #     axis=time_analysis_axis,
        #     data=(data_yearly_upscale),
        #     marker="o",
        #     line_style="-",
        #     color="b",
        #     label="Population Density Upscaled Data",
        #     axis_title="Population Data",
        #     axis_xlabel="1850 - 2014",
        #     axis_ylabel="Population Density m^2",
        # )

        # data_yearly_origin = np.column_stack(
        #     (
        #         np.arange(
        #             0,
        #             len(original_sum_values),
        #         ),
        #         original_sum_values,
        #     )
        # )

        # time_series_plot(
        #     axis=time_analysis_axis,
        #     data=(data_yearly_origin),
        #     marker="x",
        #     line_style="-",
        #     color="r",
        #     label="Population Density Original Data",
        #     axis_title="Population Original",
        #     axis_xlabel="1850 - 2014",
        #     axis_ylabel=f"Burned Area {attribute_dict['units']}",
        # )

        # diff_value = np.array(original_sum_values) - np.array(upscale_sum_values)
        # data_yearly_diff = np.column_stack(
        #     (
        #         np.arange(
        #             0,
        #             len(diff_value),
        #         ),
        #         diff_value,
        #     )
        # )

        # time_series_plot(
        #     axis=time_analysis_axis,
        #     data=(data_yearly_diff),
        #     marker="x",
        #     line_style="-",
        #     color="r",
        #     label="GFED5 Burned Area Difference",
        #     axis_title="GFED5 Burned Area Total Original Data - GFED5 Burned Area Total Upscaled Data",
        #     axis_xlabel="Monthly 1997 - 2016",
        #     axis_ylabel="Burned Area m^2",
        # )

        plt.show()


def main():
    parameters = list(sys.argv)[1:]
    dir_path, shape = handle_user_input(parameters)
    Analysis = GeoDataResizePopulationDensity(dir_path=dir_path, new_shape=shape)
    Analysis.upscale_data()


if __name__ == "__main__":
    main()
