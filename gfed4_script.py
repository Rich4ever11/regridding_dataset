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
from utilityFunc import (
    handle_user_input,
    obtain_netcdf_files,
    evaluate_upscale_sum,
    resample_matrix,
    save_file,
)


class GeoDataResizeGFED4:
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
        for file in self.files:
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
                        burned_fraction_upscaled = resample_matrix(
                            burned_fraction_product,
                            self.dest_shape,
                            self.save_folder_path,
                        )

                        # Total of orig resolution after multiplying by gridcell area should be close to equal to total of final (target) resolution.
                        # Both are in m^2.
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
                            height, width = burned_fraction_upscaled.shape
                            latitudes = np.linspace(-90, 90, height)
                            longitudes = np.linspace(-180, 180, width)

                            # flip the data matrix (upside down due to the GFED dataset's orientation)
                            burned_fraction_upscaled = np.flip(
                                burned_fraction_upscaled, 0
                            )

                            # create the xarray data array for the upscaled burned area and add it to the dictionary
                            burned_area_data_array = xarray.DataArray(
                                burned_fraction_upscaled,
                                coords={"latitude": latitudes, "longitude": longitudes},
                                dims=["latitude", "longitude"],
                                attrs=burnded_area_attribute_dict,
                            )
                            dataset_dict[f"burned_areas_{group}"] = (
                                burned_area_data_array
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


def main():
    parameters = list(sys.argv)[1:]
    dir_path, shape = handle_user_input(parameters)
    Analysis = GeoDataResizeGFED4(dir_path=dir_path, new_shape=shape)
    Analysis.upscale_data()


if __name__ == "__main__":
    main()
