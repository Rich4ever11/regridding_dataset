from netCDF4 import Dataset
import traceback
import numpy as np
from os import listdir, makedirs, remove
from os.path import isfile, join, basename, exists, dirname
from rasterio.transform import from_origin
import xarray
import rasterio
import rioxarray as riox
import sys
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from utilityGlobal import (
    KM_NEG_2TOM_NEG_2,
    DAYS_TO_SECONDS,
    KM_SQUARE_TO_M_SQUARED,
    DAYS_TO_YEARS,
    DAYS_IN_MONTH,
)
from utilityFunc import (
    handle_user_input,
    obtain_netcdf_files,
    calculate_grid_area,
    evaluate_upscale_sum,
    time_series_plot,
    resample_matrix,
    save_file,
    draw_map,
    leap_year_check,
)


class GeoDataResizeWGLC:
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
        Note - This is focused on the lightning data (wglc) so the program will fail if the data differs

        :param: None
        :return: None
        """
        start_date = "2010-01-01"
        seconds_in_years = 0
        for file in self.files:
            try:
                with Dataset(file) as netcdf_dataset:
                    # dataset containing all xarray data array (used to create the final netcdf file)
                    dataset_dict = {}
                    attribute_dict = {}
                    updated_var_data_array = []
                    origin_var_data_array = []

                    # WGLC density in units of #/km^2/day
                    density_variable = netcdf_dataset.variables["density"]
                    time_data_array = netcdf_dataset.variables["time"][:]

                    # Copy attributes of the burned area fraction
                    for attr_name in density_variable.ncattrs():
                        attribute_dict[attr_name] = getattr(density_variable, attr_name)

                    density_variable_data = np.where(
                        density_variable[:] > 0, density_variable[:], 0
                    )

                    longitudes_y = np.linspace(-180, 180, density_variable[:].shape[-1])
                    latitudes_x = np.linspace(-90, 90, density_variable[:].shape[-2])
                    date_range = pd.date_range(
                        start_date, freq="MS", periods=len(density_variable_data)
                    )
                    variable_data = np.zeros(shape=(density_variable_data[0].shape))
                    year = int(start_date.split("-")[0])
                    for month in range(len(density_variable_data)):
                        # add a flag, if month == 11 (assuming it starts from 0)
                        # you reached a year's worth of data; uppon addition of
                        # monthly data multiply by the number of seconds in that year
                        current_year = str(date_range[month]).split("-")[0]
                        curr_month = str(date_range[month]).split("-")[1]

                        if curr_month != "02":
                            second_in_month = (
                                DAYS_IN_MONTH[curr_month] / DAYS_TO_SECONDS
                            )
                        elif curr_month == "02" and leap_year_check(int(current_year)):
                            second_in_month = 29 / DAYS_TO_SECONDS

                        origin_grid_cell_area = calculate_grid_area(
                            grid_area_shape=(
                                density_variable_data[month].shape[-2],
                                density_variable_data[month].shape[-1],
                            ),
                            units="km",
                        )

                        seconds_in_years += second_in_month
                        # Converting units to #/km^2/s
                        monthly_density_variable = (
                            density_variable_data[month] * second_in_month
                        )
                        units = "strokes km^-2 s^-1"
                        # plot a monthly map; units are strokes km^-2 s^-1
                        # fixed the MM/YYYY below (add a conversion of month to MM/YYYY)
                        # updated the map plot to just draw the figure
                        map_figure_origin, map_axis_origin = plt.subplots(
                            nrows=1,
                            ncols=1,
                            figsize=(18, 10),
                            subplot_kw={"projection": ccrs.PlateCarree()},
                        )
                        draw_map(
                            map_figure=map_figure_origin,
                            map_axis=map_axis_origin,
                            units=units,
                            label=f"Original Resolution - {monthly_density_variable.shape} WGLC Data ({curr_month}/{current_year})",
                            latitude=np.linspace(
                                -90, 90, monthly_density_variable.shape[-2]
                            ),
                            longitude=np.linspace(
                                -180, 180, monthly_density_variable.shape[-1]
                            ),
                            var_data_xarray=monthly_density_variable,
                            cbarmax=None,
                        )

                        # Converting units to #/s
                        monthly_density_variable *= origin_grid_cell_area

                        # Density is now in units of #/s
                        var_data_array = monthly_density_variable

                        # preform resampling/upscaling using rasterio
                        upscaled_var_data_array = resample_matrix(
                            source_matrix=var_data_array,
                            dest_shape=self.dest_shape,
                            geotiff_output_path=self.save_folder_path,
                        )

                        upscale_grid_cell_area = calculate_grid_area(
                            grid_area_shape=self.dest_shape
                        )

                        print(f"density_month_{(month + 1)}")
                        evaluate_upscale_sum(var_data_array, upscaled_var_data_array)
                        # variable is in units of density
                        upscaled_var_data_array = (
                            upscaled_var_data_array / upscale_grid_cell_area
                        )
                        # plot a monthly map; units are strokes km^-2 s^-1
                        # fixed the MM/YYYY below (add a conversion of month to MM/YYYY)
                        # updated the map plot to just draw the figure
                        map_figure_upscale, map_axis_upscale = plt.subplots(
                            nrows=1,
                            ncols=1,
                            figsize=(18, 10),
                            subplot_kw={"projection": ccrs.PlateCarree()},
                        )
                        draw_map(
                            map_figure=map_figure_upscale,
                            map_axis=map_axis_upscale,
                            units=units,
                            label=f"Upscaled (Resolution - {upscaled_var_data_array.shape}) WGLC Data ({curr_month}/{current_year})",
                            latitude=np.linspace(
                                -90, 90, upscaled_var_data_array.shape[-2]
                            ),
                            longitude=np.linspace(
                                -180, 180, upscaled_var_data_array.shape[-1]
                            ),
                            var_data_xarray=upscaled_var_data_array,
                            cbarmax=None,
                        )
                        plt.show()
                        updated_var_data_array.append(upscaled_var_data_array)
                        origin_var_data_array.append(var_data_array)  # strokes/s

                    _, time_analysis_axis = plt.subplots(figsize=(10, 6))
                    data_density_xr = xarray.DataArray(
                        origin_var_data_array,
                        coords={
                            "time": time_data_array,
                            "latitude": latitudes_x,
                            "longitude": longitudes_y,
                        },
                        dims=["time", "latitude", "longitude"],
                    )
                    units = "strokes km^-2 yr^-1"
                    # TO DO: fix second_in_year so it is not hard coded but calculated from the number of days in a year
                    # secondsinyear = number of second in a year, clculate base on year as leap years have different # seconds

                    # 1. read the monthly data and convert to #/km^2/s
                    # 2. calculate annual total: A) add up the monthly data per year and
                    #   B) convert #/km^2/s to #/km^2/yr by multiplying the number of seconds in a year (note leap years)
                    # 3. Calculate the climatological mean
                    # 4. plot map

                    map_figure_origin, map_axis_origin = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(18, 10),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )
                    draw_map(
                        map_figure=map_figure_origin,
                        map_axis=map_axis_origin,
                        units=units,
                        label=f"Original {data_density_xr.shape} WGLC Data mean ({'2010-2021'})",
                        latitude=latitudes_x,
                        longitude=longitudes_y,
                        var_data_xarray=data_density_xr.mean(dim="time"),
                        cbarmax=10,
                    )

                    latitudes = np.linspace(-90, 90, self.dest_shape[0])
                    longitudes = np.linspace(-180, 180, self.dest_shape[1])
                    # !! Once that is done revise the units (attribute_dict) to #/m^2/s
                    attribute_dict["units"] = "strokes km-2 d-1"
                    # creates the data array and saves it to a file
                    var_data_array_xarray = xarray.DataArray(
                        (updated_var_data_array),
                        coords={
                            "time": time_data_array,
                            "latitude": latitudes,
                            "longitude": longitudes,
                        },
                        dims=["time", "latitude", "longitude"],
                        attrs=attribute_dict,
                    )

                    map_figure_upscale, map_axis_upscale = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(18, 10),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )
                    draw_map(
                        map_figure=map_figure_upscale,
                        map_axis=map_axis_upscale,
                        units=units,
                        label=f"Upscaled {var_data_array_xarray.shape} WGLC Data mean ({'2010-2021'})",
                        latitude=latitudes,
                        longitude=longitudes,
                        var_data_xarray=var_data_array_xarray.mean(dim="time"),
                        cbarmax=10,
                    )
                    # check the draw_map mean calculation

                    dataset_dict["density"] = var_data_array_xarray
                    # saves xarray dataset to a file

                    years = np.arange(1, 144 + 1)
                    data_per_year_stack_upscale = np.column_stack(
                        (
                            years,
                            var_data_array_xarray.sum(
                                dim=(
                                    var_data_array_xarray.dims[-2],
                                    var_data_array_xarray.dims[-1],
                                )
                            ).values,
                        )
                    )

                    data_per_year_stack_origin = np.column_stack(
                        (
                            years,
                            data_density_xr.sum(
                                dim=(
                                    data_density_xr.dims[-2],
                                    data_density_xr.dims[-1],
                                )
                            ).values,
                        )
                    )

                    data_per_year_stack_diff = np.column_stack(
                        (
                            years,
                            data_per_year_stack_upscale - data_per_year_stack_origin,
                        )
                    )

                    time_series_plot(
                        axis=time_analysis_axis,
                        data=(data_per_year_stack_upscale),
                        marker="o",
                        line_style="-",
                        color="b",
                        label="Upscaled WGLC Data",
                        axis_title="",
                        axis_xlabel="",
                        axis_ylabel="",
                    )

                    time_series_plot(
                        axis=time_analysis_axis,
                        data=(data_per_year_stack_origin),
                        marker="x",
                        line_style="-",
                        color="r",
                        label="Original WGLC Data",
                        axis_title="",
                        axis_xlabel="",
                        axis_ylabel="",
                    )

                    # fix the script so you can run it once and that units and titles are assigned to the appropriate figures
                    time_series_plot(
                        axis=time_analysis_axis,
                        data=(data_per_year_stack_diff),
                        marker="x",
                        line_style="-",
                        color="g",
                        label="Upscaled WGLC Data - Original WGLC Data (WGLC Difference)",
                        axis_title="WGL Resampling Results",
                        axis_xlabel="Daily Lightning Strikes (1 - 144)",
                        axis_ylabel="Lightning Strokes km^2 y-1",
                    )

                    save_file(
                        file_path=file,
                        data_set=xarray.Dataset(dataset_dict),
                        save_folder_path=self.save_folder_path,
                        dest_shape=self.dest_shape,
                    )
                    plt.show()
            except Exception as error:
                print("[-] Failed to parse dataset: ", error)
                print(traceback.format_exc())


def main():
    parameters = list(sys.argv)[1:]
    dir_path, shape = handle_user_input(parameters)
    Analysis = GeoDataResizeWGLC(dir_path=dir_path, new_shape=shape)
    Analysis.upscale_data()


if __name__ == "__main__":
    main()
