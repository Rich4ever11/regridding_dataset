from netCDF4 import Dataset
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import rasterio
import numpy as np
from os import listdir
from os.path import isfile, join
import xarray
from skimage.measure import block_reduce


class GeoDataResize:
    """
    GeoDataResize is a class that is meant to manipulate and change the format of geodata, improving the overall analysis of geo data (currently netcdf)

    :Attribute files: the list of files that the user would like to manipulate (currently only netcdf geo data)
    """
    
    def __init__(self, dir_path) -> None:
        """
        initializes the class and sets the files

        :param dir_path: the path of the files we wish to parse
        :return: None
        """
        self.files = self.obtain_hdf5_files(dir_path)

    def obtain_hdf5_files(self, dir_path) -> list:
        """
        loops through files in the current director and returns a list of files that are netcdf files

        :param dir_path: the file path
        :return: all files in the "dir_path" that are netcdf files
        """
        return [
            file
            for file in listdir(dir_path)
            if isfile(join(dir_path, file))
            and (file.split(".")[-1] == "hdf5" or file.split(".")[-1] == "nc")
        ]

    def upscale_data_n(self) -> None:
        """
        loops through each file in the classes files list Regridding (upscaling) datasets from a fine resolution to a coarse (ModelE) resolution
        Note - This is focused on the burned area dataset and uses only netcdf
            Issue - When saving the dataset the unscaled burned area is classified as a 2D variable instead of a Geo2D variable

        :param: None
        :return: None
        """
        for file in self.files:
            try:
                with Dataset(file) as netcdf_dataset:
                    new_netcdf_file = Dataset(self.obtain_new_filename(file), mode='w')
                    dataset_dict = {}
                    
                    latitude_name = "lat"
                    longitude_name = "lon"
                    lat = (netcdf_dataset.variables["lat"])
                    lon = (netcdf_dataset.variables["lon"])
                    
                    # maintain the same dimensions
                    for dim_name, _ in netcdf_dataset.dimensions.items():
                        new_netcdf_file.createDimension(dim_name, None)
                        
                    # Create the latitude variable in the new netcdf4 dataset
                    new_lat_data = new_netcdf_file.createVariable(latitude_name, lat.dtype, dimensions=new_netcdf_file.dimensions)
                    # Copy attributes
                    for attr_name in lat.ncattrs():
                        setattr(new_lat_data, attr_name, getattr(lat, attr_name))
                        
                    # Create the longitude variable in the new netcdf4 dataset
                    new_lon_data = new_netcdf_file.createVariable(longitude_name, lon.dtype, dimensions=netcdf_dataset.dimensions)
                    # copy attributes
                    for attr_name in lon.ncattrs():
                        setattr(new_lon_data, attr_name, getattr(lon, attr_name))
                        
                    # Write the data to the destination variable
                    new_lon_data[:] = lon[:]
                    new_lat_data[:] = lat[:]
                    
                    # obtain the grid cell area value (allows for the burned area to account for the shape of the earth)
                    grid_cell_area_value = netcdf_dataset.groups["ancill"].variables["grid_cell_area"][:]
                    
                    # loop through every burned area month
                    for group in netcdf_dataset.groups["burned_area"].groups:
                        # create new group in upscale hdf5
                        curr_group = new_netcdf_file.createGroup(f"burned_areas/{group}")
                        
                        #obtain the current burned area group
                        burned_area_group = netcdf_dataset.groups["burned_area"].groups[group]
                        
                        # obtain the burned_area percentage/fraction array for the current month we are in
                        burned_area_fraction = burned_area_group.variables["burned_fraction"]
                        burned_area_source = burned_area_group.variables["source"]
                        
                        burned_area_fraction_value = burned_area_fraction[:]
                        
                        # multiplying the grid cell area by the burned fraction value
                        burned_fraction_product = grid_cell_area_value * burned_area_fraction_value
                        burned_fraction_product = np.asarray(burned_fraction_product)
                                                
                        # upscale the burned fraction
                        burned_fraction_upscaled = self.upscale_matrix_numpy(burned_fraction_product)
                        
                        # Total of orig resolution after multiplying by gridcell area should be equal to total of final (target) resolution. Both are in m^2.
                        if self.evaluate_upscale_sum(burned_fraction_product, burned_fraction_upscaled):
                            attribute_dict = {}
                            
                            # create the dimensions for the burned area
                            curr_group.createDimension("lat", None)
                            curr_group.createDimension("lon", None)
                            
                            # Create the burned fraction variable in the burned_area group dataset
                            new_burned_faction_var = curr_group.createVariable("burned_area", burned_area_fraction.dtype, dimensions=curr_group.dimensions)
                            # Copy attributes
                            for attr_name in burned_area_fraction.ncattrs():
                                attribute_dict[attr_name] = getattr(burned_area_fraction, attr_name)
                                setattr(new_burned_faction_var, attr_name, getattr(burned_area_fraction, attr_name))
                            attribute_dict["units"] = "m^2"
                                
                            # Create the sources variable in the burned_area group dataset
                            new_burned_area_source_var = curr_group.createVariable("source", burned_area_source.dtype, dimensions=curr_group.dimensions)
                            # Copy attributes
                            for attr_name in burned_area_source.ncattrs():
                                setattr(new_burned_area_source_var, attr_name, getattr(burned_area_source, attr_name))
                                
                            new_burned_faction_var[:] = burned_fraction_upscaled

            except Exception as error:
                print("[-] Failed to parse dataset: ", error)
        
    def upscale_data_x(self) -> None:
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
                    grid_cell_area_value = netcdf_dataset.groups["ancill"].variables["grid_cell_area"][:]
                    
                    # loop through every burned area month
                    for group in netcdf_dataset.groups["burned_area"].groups:
                        #obtain the current burned area group
                        burned_area_group = netcdf_dataset.groups["burned_area"].groups[group]
                        
                        # obtain the burned_area fraction array for the current month/group we are in
                        burned_area_fraction = burned_area_group.variables["burned_fraction"]
                        burned_area_fraction_value = burned_area_fraction[:]
                        
                        # multiplying the grid cell area by the burned fraction value
                        burned_fraction_product = grid_cell_area_value * burned_area_fraction_value
                        burned_fraction_product = np.asarray(burned_fraction_product)
                                                
                        # upscale the burned fraction
                        burned_fraction_upscaled = self.upscale_matrix_numpy(burned_fraction_product)
                        
                        # Total of orig resolution after multiplying by gridcell area should be equal to total of final (target) resolution. Both are in m^2.
                        if self.evaluate_upscale_sum(burned_fraction_product, burned_fraction_upscaled):
                            burnded_area_attribute_dict = {}
                            
                            # Copy attributes of the burned area fraction
                            for attr_name in burned_area_fraction.ncattrs():
                                burnded_area_attribute_dict[attr_name] = getattr(burned_area_fraction, attr_name)
                                
                            # update the units to match the upscaling process
                            burnded_area_attribute_dict["units"] = "m^2"
                            
                            # obtain the height and width from the upscale shape 
                            # create an evenly spaced array representing the longitude and the latitude
                            height, width = burned_fraction_upscaled.shape
                            latitudes = np.linspace(-90, 90, height)  
                            longitudes = np.linspace(-180, 180, width) 
                            
                            # flip the data matrix (upside down due to the GFED dataset's orientation)
                            burned_fraction_upscaled = np.flip(burned_fraction_upscaled, 0)
                            
                            # create the xarray data array for the upscaled burned area and add it to the dictionary
                            burned_area_data_array = xarray.DataArray(burned_fraction_upscaled, coords={'latitude': latitudes, 'longitude': longitudes}, dims=['latitude', 'longitude'], attrs=burnded_area_attribute_dict)
                            dataset_dict[f"burned_areas_{group}"] = burned_area_data_array

                    # saves xarray dataset to a file
                    self.save_file(file, xarray.Dataset(dataset_dict))
            except Exception as error:
                print("[-] Failed to parse dataset: ", error)
            
    def evaluate_upscale_sum(self, origin_matrix, upscaled_matrix, margin_of_error = 65536.0):
        """
        Function prints our the original matrix sum and the upscaled matrix sum (post re grid)
        It then determines if the upscaled matrix sum is close enough to the original matrix sum (incorporating a margin of error)
        
        :param origin_matrix: original matrix before re grid
        :param upscaled_matrix: the original matrix post re grid
        :param margin_of_error: the margin of error the allowed 
        :return: boolean
        """
        print()
        print(f"Original Burned Area Total - {origin_matrix.sum()}")
        print(f"\tOriginal Burned Area Dimensions - {origin_matrix.shape}")
        print(f"Upscale Burned Area Total - {upscaled_matrix.sum()}")
        print(f"\tUpscale Burned Area Dimensions - {upscaled_matrix.shape}")
        print()
        
        # returns true if the upscaled matrix sum is within the range of the original matrix sum (margin of error accounts for rounding of values)
        return upscaled_matrix.sum() >= (origin_matrix.sum() - margin_of_error) and origin_matrix.sum() <= (origin_matrix.sum() + margin_of_error)
        

    def upscale_matrix_numpy(self, source_matrix, window_height = 8, window_width = 10):
        """
        Function preforms the process of upscaling the passed in matrix using numpy or skimage
        (first determining if this is possible and then preforming the operation)
        
        :param source_matrix: matrix we wish to compress (upscale)
        :param window_height: height of the window we wish to iterate over with
        :param window_width: width of the window we wish to iterate over with
        :return: upscaled matrix
        """
        
        try:
            source_shape = source_matrix.shape   
            # check if the window size lines up evenly with the passed in matrix
            if (source_shape[0] % window_height == 0) and (source_shape[1] % window_width == 0):
                # This is another method to reduce a matrix with a window using a sum calculation (both work the same)
                # downscaled_data = block_reduce(source_matrix, block_size=(window_height, window_width), func=np.sum)
                # reshape the matrix into a 4D matrix (shows each window of the matrix)
                reshape_result = source_matrix.reshape(source_shape[0] // window_height, window_height, source_shape[1] // window_width, window_width)
                # sum the windows and creates the 2D matrix
                result = reshape_result.sum(axis=(1, 3))
                print("[+] Successfully upscaled matrix, current updated matrix shape: ", np.asarray(result).shape)
                return result
        except Exception as error:
            print("[-] Failed to upscale matrix", error)
            return source_matrix
    
    def upscale_matrix_restario(self, source_matrix, destination_matrix):
        """
        Function preforms the process of upscaling the passed in matrix using rasterio
        Issues - There is no errors however the result produces an array matching the desired dimensions but missing any values
        
        :param source_matrix: matrix we wish to compress (upscale)
        :param destination_matrix: a matrix with the desired dimensions and filled with empty
        :return: upscaled matrix
        """
        source = np.asarray(source_matrix)
        
        src_crs = 'EPSG:4326'
        dst_crs = 'EPSG:4326'
        
        src_transform = rasterio.transform.from_origin(0, len(source_matrix), 1, 1)
        dst_transform = rasterio.transform.from_origin(0, len(destination_matrix), 1, 1)
        
        # src_transform = A.identity()
        # dst_transform = A.identity()
        
        result = reproject(
            source=source,
            destination=destination_matrix,
            src_transform=src_transform,
            dst_transform=dst_transform,
            src_crs=src_crs,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.max)
        
        print(np.max(result[0][:]))
        print((result[0][:]).sum())
        print(result[0][:].shape)
        return (result)
    
    def obtain_new_filename(self, file_path) -> str:
        # creates a file name
        file_path_list = file_path.split(".")
        file_path_list[-2] = file_path_list[-2] + "(upscaled)"
        # ensures the file is saved as a netcdf file
        file_path_list[-1] = "nc"
        # rejoin the list
        return ".".join(file_path_list)
    
    def save_file(self, file_path, data_set) -> None:
        """
        Saves the xarray dataset based on the file inputted to the function
        
        :param file_path: file path of the current file being upscaled/processed 
        :param data_set: data set representing the 
        :return: None
        """
        try:
            # create the new file's path & name
            new_file_name = self.obtain_new_filename(file_path)
            # saves the file using the created file path and xarray
            data_set.to_netcdf(path=(new_file_name))
            print(f"[+] file {new_file_name} saved")
        except Exception as error:
            print(
                "[-] Failed to save dataset (ensure dataset is from xarray lib): ",
                error,
            )

def main():
    Analysis = GeoDataResize(".")
    Analysis.upscale_data_x()


if __name__ == "__main__":
    main()
