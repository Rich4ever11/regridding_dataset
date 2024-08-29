from netCDF4 import Dataset
import rioxarray as riox
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import rasterio
from rasterio.enums import Resampling
import numpy as np
from os import listdir
from os.path import isfile, join
import xarray

# https://rasterio.readthedocs.io/en/stable/topics/reproject.html
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
                    ncfile = Dataset('./new.hdf5',mode='w',format='NETCDF4_CLASSIC') 
                    print(ncfile)
                    netcdf_dataset = Dataset(file_path)
                    # obtain the grid cell area value (allows for the burned area to account for the shape of the earth)
                    grid_cell_area_value = netcdf_dataset.groups["ancill"].variables["grid_cell_area"][:]
                    # loop through every burned area month
                    for group in netcdf_dataset.groups["burned_area"].groups:
                        burned_area_group = netcdf_dataset.groups["burned_area"].groups[group]
                        # obtain the burned_area percentage/fraction array for the current month we are in
                        burned_area_fraction_value = burned_area_group.variables["burned_fraction"][:]
                        
                        # multiplying the grid cell area by the burned fraction value
                        burned_fraction_product = grid_cell_area_value * burned_area_fraction_value
                        
                        # allocate the resulting array
                        upscaling_shape = (90, 144)
                        destination = np.zeros(upscaling_shape)
                        burned_fraction_upscaled = self.upscale_matrix(burned_fraction_product)
                        
                        origin_resolution_sum = np.sum(burned_fraction_product)
                        rescaled_resolution_sum = np.sum(burned_fraction_upscaled)
                        
                        print(origin_resolution_sum, rescaled_resolution_sum)
                        # Total of orig resolution after multiplying by gridcell area should be equal to total of final (target) resolution. Both are in m^2.
                        if rescaled_resolution_sum == origin_resolution_sum:
                            print("TRUE")
                        


                        return
                        # obtain the grid_area_array percentage/fraction array
                        
                        pass
                    
                    

                    # self.save_netcdf_file_xarray(file_path, data_set)
        except Exception as error:
            print("[-] Failed to parse dataset: ", error)

    def upscale_matrix(self, source_matrix, window_height = 8, window_width = 10):
        source_shape = source_matrix.shape   
        if (source_shape[0] % window_height == 0) and (source_shape[1] % window_width == 0):
            result = source_matrix.reshape(source_shape[0] // window_height, window_height, source_shape[1] // window_width, window_width).sum()
            print("[+] Successfully unscaled matrix, current updated matrix shape: ", result.shape)
            return (result)
        print("[-] Failed to upscale matrix (window size does not match grid)")
        return source_matrix
        pass
    
    def upscale_matrix_restario(self, source_matrix, destination_matrix, window_height = 8, window_width = 10):
        # source = np.asarray(burned_fraction_product)
        
        # # Preforms no Affline Transformation
        # src_transform = A.identity() 
        # dst_transform = A.identity()
        
        # src_crs = {'init': 'EPSG:3857'}
        
        # result = (reproject(
        #     source=source,
        #     src_transform=src_transform,
        #     dst_transform=dst_transform,
        #     destination=destination,
        #     resampling=Resampling.max))
        
        # # riox.open_rasterio.reproject()

        # print(len(result[0]), len(result[0][0]))
        # print(np.sum(result[:][0]))
        # if (np.sum(burned_fraction_product) == np.sum(result[:][0])):
        #     "Upscaling Complete"
        pass
    
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
