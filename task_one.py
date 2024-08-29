from netCDF4 import Dataset
import rioxarray
import rasterio
from rasterio.enums import Resampling
import numpy as np
from os import listdir
from os.path import isfile, join
import xarray


# https://www.youtube.com/watch?v=79o6DXr_3zM
# https://spatial-dev.guru/2022/09/24/upsample-and-downsample-raster-in-python-using-rioxarray/
# https://stackoverflow.com/questions/69873101/problems-with-netcdf-regridding-python
# https://stackoverflow.com/questions/73455966/regrid-xarray-dataset-to-a-grid-that-is-2-5-times-coarser


class EnvironmentalTranslation:
    def __init__(self, file_path) -> None:
        self.data_set_file_path = self.obtain_hdf5_files(file_path)
        self.upscale_factor = 2

    def obtain_hdf5_files(self, file_path):
        return [
            file
            for file in listdir(file_path)
            if isfile(join(file_path, file))
            and (file.split(".")[-1] == "hdf5" or file.split(".")[-1] == "nc")
        ]

    def upscale_data(self) -> None:
        try:
            for file_path in self.data_set_file_path:
                with xarray.open_dataset(file_path) as data_set:
                    netcdf_dataset = Dataset(file_path)
                    # obtain the grid cell area value (allows for the burned area to account for the shape of the earth)
                    grid_cell_area_value = netcdf_dataset.groups["ancill"].variables["grid_cell_area"][:]
                    # loop through every burned area month
                    for group in netcdf_dataset.groups["burned_area"].groups:
                        burned_area_group = netcdf_dataset.groups["burned_area"].groups[group]
                        # obtain the burned_area percentage/fraction array for the current month we are in
                        burned_area_fraction_value = burned_area_group.variables["burned_fraction"][:]
                        burned_fraction_product = grid_cell_area_value * burned_area_fraction_value
                        print(burned_fraction_product)
                        
                        return
                        # obtain the grid_area_array percentage/fraction array
                        
                        pass
                    
                    

                    # self.save_netcdf_file_xarray(file_path, data_set)
        except Exception as error:
            print("[-] Failed to parse dataset: ", error)

    def save_netcdf_file_xarray(self, file_path, data_set) -> None:
        try:
            file_path_list = file_path.split(".")
            file_path_list[0] = file_path_list[0] + "(upscaled)"
            new_file_name = ".".join(file_path_list)
            data_set.to_netcdf(path=new_file_name)
            print(f"[+] file {new_file_name} saved")
        except Exception as error:
            print(
                "[-] Failed to save dataset (ensure dataset is from xarray lib): ",
                error,
            )

def main():
    Analysis = EnvironmentalTranslation(".")
    Analysis.upscale_data()


if __name__ == "__main__":
    main()
