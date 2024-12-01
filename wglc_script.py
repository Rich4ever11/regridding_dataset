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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from utilityGlobal import (
    KM_NEG_2TOM_NEG_2,
    DAYS_TO_SECONDS,
    KM_SQUARE_TO_M_SQUARED,
    DAYS_TO_YEARS,
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
        _, time_analysis_axis = plt.subplots(figsize=(10, 6))
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

                    map_figure, map_axis = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(18, 10),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )
                    print(density_variable[:].shape)
                    longitudes_y = np.linspace(-180, 180, density_variable[:].shape[-1])
                    latitudes_x = np.linspace(-90, 90, density_variable[:].shape[-2])

                    for month in range(len(density_variable_data)):
                        origin_grid_cell_area = calculate_grid_area(
                            grid_area_shape=(
                                density_variable_data[month].shape[-2],
                                density_variable_data[month].shape[-1],
                            ),
                            units="km",
                        )
                        monthly_density_variable = density_variable_data[month]

                        # Density is now in units of #/s
                        var_data_array = monthly_density_variable

                        # preform resampling/upscaling using rasterio
                        # Conversion (720, 1440) -> (90, 144)
                        upscaled_var_data_array = resample_matrix(
                            source_matrix=var_data_array,
                            dest_shape=self.dest_shape,
                            geotiff_output_path=self.save_folder_path,
                        )

                        # !!Need to divide var_data_array_xarray by the upscaled area matrix or axyp (should be the same) (doing this causes an error and makes the values come out uneven)
                        upscale_grid_cell_area = calculate_grid_area(
                            grid_area_shape=self.dest_shape
                        )

                        print(f"density_month_{(month + 1)}")
                        evaluate_upscale_sum(var_data_array, upscaled_var_data_array)
                        # variable is in units of density
                        # upscaled_var_data_array = (
                        #     upscaled_var_data_array / upscale_grid_cell_area
                        # )
                        updated_var_data_array.append(upscaled_var_data_array)
                        origin_var_data_array.append(var_data_array)

                    data_density_xr = xarray.DataArray(
                        origin_var_data_array,
                        coords={
                            "time": time_data_array,
                            "latitude": latitudes_x,
                            "longitude": longitudes_y,
                        },
                        dims=["time", "latitude", "longitude"],
                    )
                    draw_map(
                        map_figure=map_figure,
                        map_axis=map_axis,
                        units=attribute_dict["units"],
                        label="Original WGLC Data",
                        latitude=latitudes_x,
                        longitude=longitudes_y,
                        var_data_xarray=data_density_xr,
                        cbarmac=8,
                    )

                    latitudes = np.linspace(-90, 90, self.dest_shape[0])
                    longitudes = np.linspace(-180, 180, self.dest_shape[1])
                    print(len(latitudes), len(longitudes))
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

                    dataset_dict["density"] = var_data_array_xarray
                    # saves xarray dataset to a file

                    map_figure, map_axis = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(18, 10),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )

                    draw_map(
                        map_figure=map_figure,
                        map_axis=map_axis,
                        units=attribute_dict["units"],
                        label="Upscaled WGLC Data",
                        latitude=latitudes,
                        longitude=longitudes,
                        var_data_xarray=var_data_array_xarray,
                        cbarmac=8,
                    )

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
