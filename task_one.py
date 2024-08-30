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
                with Dataset(file_path) as netcdf_dataset:
                    new_netcdf_file = Dataset('./new.hdf5', mode='w', format="NETCDF4")
                    
                    latitude_name = "lat"
                    longitude_name = "lon"
                    
                    lat = (netcdf_dataset.variables["lat"])
                    lon = (netcdf_dataset.variables["lon"])
                    
                    for dim_name, dim in netcdf_dataset.dimensions.items():
                        new_netcdf_file.createDimension(dim_name, len(netcdf_dataset.dimensions[dim_name]))
                            
                    # Create the latitude variable in the new netcdf4 dataset
                    new_lat_data = new_netcdf_file.createVariable(latitude_name, lat.dtype, dimensions=netcdf_dataset.dimensions)
                    # Copy attributes
                    for attr_name in lat.ncattrs():
                        setattr(new_lat_data, attr_name, getattr(lat, attr_name))
                        
                        
                    # Create the longitude variable in the new netcdf4 dataset
                    new_lon_data = new_netcdf_file.createVariable(longitude_name, lon.dtype, dimensions=netcdf_dataset.dimensions)
                    # copy attributes
                    for attr_name in lon.ncattrs():
                        setattr(new_lon_data, attr_name, getattr(lon, attr_name))


                    # Write the data to the destination variable
                    new_lat_data[:] = lat[:]
                    new_lat_data[:] = lon[:]
                    
                    # create burned area group
                    new_burned_area_group = new_netcdf_file.createGroup("burned_area")
                    
                    # obtain the grid cell area value (allows for the burned area to account for the shape of the earth)
                    grid_cell_area_value = netcdf_dataset.groups["ancill"].variables["grid_cell_area"][:]
                    
                    # loop through every burned area month
                    for group in netcdf_dataset.groups["burned_area"].groups:
                        # create new group in upscale hdf5
                        curr_group = new_netcdf_file.createGroup(f"burned_area/{group}")
                        
                        #obtain the current burned area group
                        burned_area_group = netcdf_dataset.groups["burned_area"].groups[group]
                        
                        # obtain the burned_area percentage/fraction array for the current month we are in
                        burned_area_fraction = burned_area_group.variables["burned_fraction"]
                        burned_area_source = burned_area_group.variables["burned_fraction"]
                        
                        burned_area_fraction_value = burned_area_group.variables["burned_fraction"][:]
                        
                        # multiplying the grid cell area by the burned fraction value
                        burned_fraction_product = grid_cell_area_value * burned_area_fraction_value
                        burned_fraction_product = np.asarray(burned_fraction_product)
                        
                        # upscale the burned fraction array
                        (burned_fraction_upscaled, burned_fraction_upscaled_sum) = self.upscale_matrix(burned_fraction_product)
                        
                        # calculate the sum for the pre upscale burned fraction and post upscaling burned fraction
                        origin_resolution_sum = (burned_fraction_product).sum()
                        
                        
                        # Total of orig resolution after multiplying by gridcell area should be equal to total of final (target) resolution. Both are in m^2.
                        if burned_fraction_upscaled_sum == origin_resolution_sum:
                            curr_group.createDimension("phony_dim_26", burned_fraction_upscaled.shape[0])
                            curr_group.createDimension("phony_dim_27", burned_fraction_upscaled.shape[1])
                            
                            # Create the burned fraction variable in the burned_area group dataset
                            new_burned_faction_var = curr_group.createVariable("burned_fraction", burned_area_fraction.dtype, dimensions=curr_group.dimensions)
                            # Copy attributes
                            for attr_name in burned_area_fraction.ncattrs():
                                setattr(new_burned_faction_var, attr_name, getattr(burned_area_fraction, attr_name))
                                
                            # Create the sources variable in the burned_area group dataset
                            new_burned_area_source_var = curr_group.createVariable("source", burned_area_source.dtype, dimensions=curr_group.dimensions)
                            # Copy attributes
                            for attr_name in burned_area_source.ncattrs():
                                setattr(new_burned_area_source_var, attr_name, getattr(burned_area_source, attr_name))
                                
                            new_burned_faction_var[:] = np.asarray(burned_fraction_upscaled)
                            
                            print("[+] Total's Match")
                        
                        


                        return

                    # self.save_netcdf_file_xarray(file_path, data_set)
        except Exception as error:
            print("[-] Failed to parse dataset: ", error)

    def upscale_matrix(self, source_matrix, window_height = 8, window_width = 10):
        # upscaling_shape = (90, 144)
        # destination = np.zeros(upscaling_shape)
        source_shape = source_matrix.shape   
        if (source_shape[0] % window_height == 0) and (source_shape[1] % window_width == 0):
            result = source_matrix.reshape(source_shape[0] // window_height, window_height, source_shape[1] // window_width, window_width).sum(axis=(1,3))
            result_sum = source_matrix.reshape(source_shape[0] // window_height, window_height, source_shape[1] // window_width, window_width).sum()
            print("[+] Successfully unscaled matrix, current updated matrix shape: ", np.asarray(result).shape)
            return (result, result_sum)
        print("[-] Failed to upscale matrix (window size does not match grid)")
        return source_matrix
        pass
    
    def upscale_matrix_restario(self, source_matrix, destination_matrix, window_height = 8, window_width = 10):
        source = np.asarray(source_matrix)
        
        # Preforms no Affline Transformation
        src_transform = A.identity() 
        dst_transform = A.identity()
        
        src_crs = {'init': 'EPSG:3857'}
        
        result = (reproject(
            source=source,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            destination=destination_matrix,
            dest_crs=src_crs,
            resampling=Resampling.max))
        
        # riox.open_rasterio.reproject()
        print(result)
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
