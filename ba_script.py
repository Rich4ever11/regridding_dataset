from netCDF4 import Dataset
import traceback
import numpy as np
from os import listdir, makedirs, remove
from os.path import isfile, join, basename, exists, dirname
from rasterio.transform import from_origin
import pandas as pd
import xarray
import rasterio
import rioxarray as riox
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from utilityGlobal import KM_TO_M, M2TOMHA
from utilityFunc import (
    handle_user_input,
    obtain_netcdf_files,
    obtain_variables_gfed5,
    evaluate_upscale_sum,
    time_series_plot,
    resample_matrix,
    draw_map,
    save_file,
)


class GeoDataResizeBA:
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
        self.nc_variable_names = ["Crop", "Defo", "Peat", "Total"]

    def upscale_data(self) -> None:
        """
        loops through each file in the classes files list Regridding (upscaling) datasets from a fine resolution to a coarse (ModelE) resolution
        Note - This is focused on the burned area dataset and uses both netcdf (parsing/reading) and xarray (saving the data)
            Issue (SOLVED) - When saving the dataset the unscaled burned area is classified as a 2D variable instead of a Geo2D variable

        :param: None
        :return: None
        """
        _, time_analysis_axis = plt.subplots(figsize=(10, 6))
        upscale_sum_values = []
        original_sum_values = []
        previous_year = 1997
        current_upscale_year_sum_value = 0
        current_origin_year_sum_value = 0
        year_list = []
        for file in self.files:
            try:
                curr_year = int(file.split("BA")[-1].split(".")[0][:4])
                with Dataset(file) as netcdf_dataset:
                    # dataset containing all xarray data array (used to create the final netcdf file)
                    dataset_dict = {}
                    original_dataset_dict = {}
                    files_nc_variable_names = obtain_variables_gfed5(
                        netcdf_dataset, self.nc_variable_names
                    )
                    for variable_name in files_nc_variable_names:
                        match variable_name:
                            # calculates the Nat array
                            case "Nat":
                                # transform the arrays dimensions to (720, 1440) and convert (km^2 -> m^2)
                                # obtain all needed data array
                                var_total_data_array = (
                                    netcdf_dataset.variables["Total"][:][0] * KM_TO_M
                                )
                                var_crop_data_array = (
                                    netcdf_dataset.variables["Crop"][:][0] * KM_TO_M
                                )
                                var_defo_data_array = (
                                    netcdf_dataset.variables["Defo"][:][0] * KM_TO_M
                                )
                                var_peat_data_array = (
                                    netcdf_dataset.variables["Peat"][:][0] * KM_TO_M
                                )
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

                        # preform resampling/upscaling using rasterio
                        # Conversion (720, 1440) -> (90, 144)
                        upscaled_var_data_array = resample_matrix(
                            var_data_array, self.dest_shape, self.save_folder_path
                        )

                        if variable_name == "Total":
                            if curr_year > previous_year:
                                upscale_sum_values.append(
                                    upscaled_var_data_array.sum() * M2TOMHA
                                )
                                original_sum_values.append(
                                    var_data_array.sum() * M2TOMHA
                                )
                                year_list.append(previous_year)
                                current_upscale_year_sum_value = 0
                                current_origin_year_sum_value = 0
                                previous_year = curr_year
                            else:
                                current_upscale_year_sum_value += (
                                    upscaled_var_data_array.sum()
                                )
                                current_origin_year_sum_value += var_data_array.sum()

                        # if evaluate_upscale_sum(
                        #     var_data_array, upscaled_var_data_array
                        # ):
                        attribute_dict = {}

                        # Copy attributes of the burned area fraction
                        for attr_name in var_data.ncattrs():
                            attribute_dict[attr_name] = getattr(var_data, attr_name)

                        # update the units to match the upscaling process
                        attribute_dict["units"] = "m^2"

                        # obtain the height and width from the upscale shape
                        # create an evenly spaced array representing the longitude and the latitude
                        height, width = upscaled_var_data_array.shape
                        latitudes = np.linspace(-90, 90, height)
                        longitudes = np.linspace(-180, 180, width)

                        # flip the data matrix (upside down due to the GFED dataset's orientation)
                        # burned_fraction_upscaled = np.flip(burned_fraction_upscaled, 0)

                        # create the xarray data array for the upscaled burned area and add it to the dictionary
                        burned_area_data_array = xarray.DataArray(
                            upscaled_var_data_array,
                            coords={
                                "latitude": latitudes,
                                "longitude": longitudes,
                            },
                            dims=["latitude", "longitude"],
                            attrs=attribute_dict,
                        )

                        height, width = var_data_array.shape
                        latitudes = np.linspace(-90, 90, height)
                        longitudes = np.linspace(-180, 180, width)
                        time = np.arange(1, 2)

                        original_burned_area_data_array = xarray.DataArray(
                            var_data_array,
                            coords={
                                "latitude": latitudes,
                                "longitude": longitudes,
                            },
                            dims=["latitude", "longitude"],
                            attrs=attribute_dict,
                        )

                        dataset_dict[variable_name] = burned_area_data_array
                        original_dataset_dict[variable_name] = (
                            original_burned_area_data_array
                        )

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

        upscale_sum_values.append(upscaled_var_data_array.sum() * M2TOMHA)
        original_sum_values.append(var_data_array.sum() * M2TOMHA)
        year_list.append(curr_year)

        # data_yearly_upscale = np.rec.fromarrays([date_range, upscale_sum_values])
        data_yearly_upscale = np.column_stack(
            (
                year_list,
                upscale_sum_values,
            )
        )

        time_series_plot(
            axis=time_analysis_axis,
            data=data_yearly_upscale,
            marker="o",
            line_style="-",
            color="b",
            label="GFED5 Burned Area Total Upscaled Data",
            axis_title="GFED5 Burned Area Original",
            axis_xlabel="Monthly 1997 - 2020",
            axis_ylabel="Burned Area [Mha]",
        )

        # data_yearly_origin = np.rec.fromarrays([date_range, upscale_sum_values])
        data_yearly_origin = np.column_stack(
            (
                year_list,
                original_sum_values,
            )
        )

        time_series_plot(
            axis=time_analysis_axis,
            data=(data_yearly_origin),
            marker="x",
            line_style="-",
            color="r",
            label="GFED5 Burned Area Total Original Data",
            axis_title="GFED5 Burned Area Original",
            axis_xlabel="Monthly 1997 - 2020",
            axis_ylabel="Burned Area [Mha]",
        )

        # diff_value = np.array(original_sum_values) - np.array(upscale_sum_values)
        # data_yearly_diff = np.column_stack(
        #     (
        #         year_list,
        #         diff_value * M2TOMHA,
        #     )
        # )

        # time_series_plot(
        #     axis=time_analysis_axis,
        #     data=(data_yearly_diff),
        #     marker="o",
        #     line_style="-",
        #     color="g",
        #     label="GFED5 Burned Area Difference (Original - Upscaled)",
        #     axis_title="GFED5 Burned Area Total Original Data - GFED5 Burned Area Total Upscaled Data",
        #     axis_xlabel="Monthly 1997 - 2020",
        #     axis_ylabel="Burned Area [Mha]",
        # )

        plt.show()


def main():
    parameters = list(sys.argv)[1:]
    dir_path, shape = handle_user_input(parameters)
    Analysis = GeoDataResizeBA(dir_path=dir_path, new_shape=shape)
    Analysis.upscale_data()


if __name__ == "__main__":
    main()
