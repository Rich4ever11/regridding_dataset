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
from utilityGlobal import KM_NEG_2TOM_NEG_2, DAYS_TO_SECONDS, KM_SQUARE_TO_M_SQUARED
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

                    # WGLC density in units of #/km^2/day
                    density_variable = netcdf_dataset.variables["density"]
                    time_data_array = netcdf_dataset.variables["time"][:]

                    # Copy attributes of the burned area fraction
                    for attr_name in density_variable.ncattrs():
                        attribute_dict[attr_name] = getattr(density_variable, attr_name)

                    density_variable_data = np.where(
                        density_variable[:] > 0, density_variable[:], 0
                    )
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
                        var_data_array = (
                            (monthly_density_variable * origin_grid_cell_area)
                        ) / (DAYS_TO_SECONDS)

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
                        upscaled_var_data_array = (
                            upscaled_var_data_array / upscale_grid_cell_area
                        )
                        print(attribute_dict["units"])
                        updated_var_data_array.append(upscaled_var_data_array)

                    latitudes = np.linspace(-90, 90, self.dest_shape[0])
                    longitudes = np.linspace(-180, 180, self.dest_shape[1])
                    # !! Once that is done revise the units (attribute_dict) to #/m^2/s
                    attribute_dict["units"] = "strokes m-2 s-1"
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
