from netCDF4 import Dataset
import numpy as np
from os import listdir
from os.path import isfile, join
import nctoolkit as nc
import xarray


# https://www.youtube.com/watch?v=79o6DXr_3zM
# https://stackoverflow.com/questions/69873101/problems-with-netcdf-regridding-python
# https://stackoverflow.com/questions/73455966/regrid-xarray-dataset-to-a-grid-that-is-2-5-times-coarser


class EnvironmentalTranslation:
    def __init__(self, file_path) -> None:
        self.data_set_file_path = self.obtain_hdf5_files(file_path)

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
                    # obtain the min and max long values
                    min_lon = data_set.lon.min().item()
                    max_lon = data_set.lon.max().item()
                    # obtain the min and max latitude values
                    min_lat = data_set.lat.min().item()
                    max_lat = data_set.lat.max().item()

                    # obtain the new longitude and latitude values (using np)
                    # evenly space out the values within a given interval
                    # new_grid_lon = np.arange(
                    #     np.ceil(min_lon / 2) * 2, (np.floor(max_lon / 2) + 0.5) * 2, 2
                    # )
                    # new_grid_lat = np.arange(
                    #     np.ceil(min_lat / 2) * 2, (np.floor(max_lat / 2) + 0.5) * 2, 2
                    # )

                    # interpolate using nearest neighbor (can use linear, etc. if desired) (xarray) (shrink the grid)
                    # data_set = data_set.interp(
                    #     phony_dim_146=new_grid_lat,
                    #     phony_dim_147=new_grid_lon,
                    #     method="nearest",
                    # )

                    # method for rescaling using nctoolkit
                    data_set_nc = nc.open_data(file_path)
                    # changes the resolution for r2x2.5 using the current nc files longitude and latitude
                    data_set_nc.to_latlon(
                        lon=[min_lon, max_lon], lat=[min_lat, max_lat], res=[2, 2.5]
                    )

                    self.save_netcdf_file_nc(file_path, data_set)
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

    def save_netcdf_file_nc(self, file_path, data_set) -> None:
        try:
            file_path_list = file_path.split(".")
            file_path_list[0] = file_path_list[0] + "(upscaled)"
            new_file_name = ".".join(file_path_list)
            data_set.to_nc(path=new_file_name)
            print(f"[+] file {new_file_name} saved")
        except Exception as error:
            print(
                "[-] Failed to save dataset (ensure dataset is from xarray lib): ",
                error,
            )


def main():
    Analysis = EnvironmentalTranslation(".")
    Analysis.upscale_data()


main()
