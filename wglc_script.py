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
from utilityGlobal import KM_NEG_2TOM_NEG_2, DAYS_TO_SECONDS
from utilityFunc import (
    handle_user_input,
    obtain_netcdf_files,
    calculate_grid_area,
    evaluate_upscale_sum,
    resample_matrix,
    save_file,
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
        for file in self.files:
            try:
                with Dataset(file) as netcdf_dataset:
                    # dataset containing all xarray data array (used to create the final netcdf file)
                    dataset_dict = {}
                    attribute_dict = {}
                    updated_var_data_array = []

                    # update the units to match the upscaling process
                    attribute_dict["units"] = "m^2"

                    density_variable = netcdf_dataset.variables["density"]
                    time_data_array = netcdf_dataset.variables["time"][:]
                    grid_cell_area = calculate_grid_area(grid_area_shape=(360, 720))

                    # Copy attributes of the burned area fraction
                    for attr_name in density_variable.ncattrs():
                        attribute_dict[attr_name] = getattr(density_variable, attr_name)

                    for month in range(len(density_variable[:])):
                        var_data_array = (
                            (density_variable[:][month] * grid_cell_area)
                            * KM_NEG_2TOM_NEG_2
                            / DAYS_TO_SECONDS
                        )

                        # preform resampling/upscaling using rasterio
                        # Conversion (720, 1440) -> (90, 144)
                        upscaled_var_data_array = resample_matrix(
                            source_matrix=var_data_array,
                            dest_shape=self.dest_shape,
                            geotiff_output_path=self.save_folder_path,
                        )

                        print(f"density_month_{(month + 1)}")
                        evaluate_upscale_sum(var_data_array, upscaled_var_data_array)
                        updated_var_data_array.append(upscaled_var_data_array)

                    latitudes = np.linspace(-90, 90, self.dest_shape[0])
                    longitudes = np.linspace(-180, 180, self.dest_shape[1])

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
                    # uncommenting this creates issues on panoply
                    # var_grid_cell_area = xarray.DataArray(
                    #     np.asarray(calculate_grid_area(grid_area_shape=(720, 1440))),
                    #     coords={
                    #         "latitude": np.linspace(-90, 90, 720),
                    #         "longitude": np.linspace(-180, 180, 1440),
                    #     },
                    #     dims=["latitude", "longitude"],
                    #     attrs=attribute_dict,
                    # )
                    dataset_dict["density"] = var_data_array_xarray
                    # dataset_dict["grid_cell_area"] = var_grid_cell_area
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


def main():
    parameters = list(sys.argv)[1:]
    dir_path, shape = handle_user_input(parameters)
    Analysis = GeoDataResizeWGLC(dir_path=dir_path, new_shape=shape)
    Analysis.upscale_data()


if __name__ == "__main__":
    main()
