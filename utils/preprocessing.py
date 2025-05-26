"""
python gdal has difficulty controlling memory. 
Methods that cannot free memory are
    1. ``concurrent.futures.ProcessPoolExecutor``
    2. ``concurrent.futures.ThreadPoolExecutor``
This module was created because it is necessary to use one of the following.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import gc
import os
import tempfile
from typing import Union

import numpy as np
from osgeo import gdal
gdal.UseExceptions()

from gdal_drawer.utils.config import (
    CRS,
    GEOMETRY,
    crs_checker,
    geometry_checker,
    Bounds
)
from gdal_drawer.utils.raster_info import (
    crs_unit_name,
    get_bounds,
    resolution_from_dataset
)


@dataclass
class TempFileDataset:
    temp_file: tempfile.NamedTemporaryFile
    dataset: gdal.Dataset
    
    def close(self):
        """
        ## Summary:
            Close and delete the temporary file.
        ## Examples:
            >>> temp_dataset = TempDataset(temp_file, dataset)
            >>> temp_dataset.close()
        """
        self.dataset = None
        self.temp_file.close()
        os.unlink(self.temp_file.name)
        gc.collect()


def copy_temppfile(dst: gdal.Dataset | str, prefix: str = 'GDAL_CUSTOM_', **kwargs) -> TempFileDataset:
    """
    ## Summary:
        Create a temporary file and copy the gdal.Dataset to it.
    Args:
        dst (gdal.Dataset): gdal.Dataset
        prefix (str): Prefix for the temporary file name.
        kwargs(dict):
            - xsize (int): Width of the dataset. Default is dst.RasterXSize.
            - ysize (int): Height of the dataset. Default is dst.RasterYSize.
            - count (int): Number of bands in the dataset. Default is dst.RasterCount.
            - eType (int): Data type of the dataset. Default is dst.GetRasterBand(1).DataType.
            - projection (str): Projection of the dataset. Default is dst.GetProjection().
            - transform (list): GeoTransform of the dataset. Default is dst.GetGeoTransform().
    Returns:
        TempDataset(dataclass): Temporary file and gdal.Dataset
            - temp_file(tempfile.NamedTemporaryFile)
            - dataset(gdal.Dataset)
    Examples:
        >>> temp_dataset = copy_temppfile(dst)
        >>> temp_dataset.close()
    """
    driver = gdal.GetDriverByName("GTiff")
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=".tif",
        delete=False
    )
    options = {
        'xsize': kwargs.get('xsize', dst.RasterXSize),
        'ysize': kwargs.get('ysize', dst.RasterYSize),
        'bands': kwargs.get('bands', dst.RasterCount),
        'eType': kwargs.get('eType', dst.GetRasterBand(1).DataType),
    }
    temp_dst = driver.Create(temp_file.name, **options)
    temp_dst.SetProjection(kwargs.get('projection', dst.GetProjection()))
    temp_dst.SetGeoTransform(kwargs.get('transform', dst.GetGeoTransform()))
    for i in range(dst.RasterCount):
        band = dst.GetRasterBand(i + 1)
        temp_band = temp_dst.GetRasterBand(i + 1)
        temp_band.SetNoDataValue(band.GetNoDataValue())
        data = band.ReadAsArray()
        temp_band.WriteArray(data)
    temp_dst.FlushCache()
    return TempFileDataset(temp_file, temp_dst)


def copy_memory(dst: gdal.Dataset, **kwargs) -> gdal.Dataset:
    """
    ## Summary:
        Create a copy of the gdal.Dataset in memory.
    Args:
        dst (gdal.Dataset): gdal.Dataset
    Returns:
        gdal.Dataset: gdal.Dataset in memory
    """
    driver = gdal.GetDriverByName("MEM")
    options = {
        'xsize': kwargs.get('xsize', dst.RasterXSize),
        'ysize': kwargs.get('ysize', dst.RasterYSize),
        'bands': kwargs.get('bands', dst.RasterCount),
        'eType': kwargs.get('eType', dst.GetRasterBand(1).DataType),
    }
    new_dst = driver.Create('', **options)
    new_dst.SetProjection(kwargs.get('projection', dst.GetProjection()))
    new_dst.SetGeoTransform(kwargs.get('transform', dst.GetGeoTransform()))
    for i in range(dst.RasterCount):
        band = dst.GetRasterBand(i + 1)
        new_band = new_dst.GetRasterBand(i + 1)
        new_band.SetNoDataValue(band.GetNoDataValue())
        data = band.ReadAsArray()
        new_band.WriteArray(data)
    new_dst.FlushCache()
    return new_dst
    

def nodata_to_nan(dst: gdal.Dataset) -> gdal.Dataset:
    """
    ## Summary:
        Convert NoData values to NaN in a gdal.Dataset. Value type must be float.
    Args:
        dst (gdal.Dataset): gdal.Dataset
    Returns:
        gdal.Dataset: gdal.Dataset with NoData values converted to NaN
    Examples:
        >>> dst = gdal.Open(file_path)
        >>> dst = nodata_to_nan(dst)
        >>> band = dst.GetRasterBand(1)
        >>> print(band.GetNoDataValue())
        np.nan
    """
    for i in range(dst.RasterCount):
        band = dst.GetRasterBand(i + 1)
        nodata_value = band.GetNoDataValue()
        data_type = gdal.GetDataTypeName(band.DataType)        
        if data_type not in ["Float32", "Float64"]:
            # Skip if the data type is not float
            continue
        
        ary = band.ReadAsArray()
        ary = np.where(ary == nodata_value, np.nan, ary)
        ary = np.where(np.isinf(ary), np.nan, ary)
        band.WriteArray(ary)
        band.SetNoDataValue(np.nan)
    return dst
    

def fill_nodata(
    dst: gdal.Dataset, 
    nodata_nan: bool = True,
    max_search: int = 10, 
    smoothing: int = 1,
    return_array: bool = True,
    release_memory: bool = True
) -> Union[gdal.Dataset, np.array]:
    """
    ## Summary:
        Fill in the value of NoData using the surrounding values.
        Use ``ThreadPoolExecutor`` to prevent memory leaks.
        Note that if return_array is False, memory cannot be freed.
    Args:
        dst (gdal.Dataset): 
            gdal.Dataset
        nodata_nan (bool): 
            Convert NoData values to NaN. Default is True.
        max_search (int): 
            Maximum search distance for filling NoData values. Default is 10.
        smoothing (int): 
            Number of smoothing iterations. Default is 1.
        return_array (bool): 
            Return a numpy array instead of a gdal.Dataset. Default is True.
            Note that it is harder to control memory usage in the case of gdal.Dataset.
        release_memory (bool):
            Whether to free Dataset memory before processing. Default is True.
    Returns:
        gdal.Dataset: 
            gdal.Dataset with NoData values filled in.
        np.array:
            Numpy array with NoData values filled in if return_array is True.
    Examples:
        >>> dst = gdal.Open(file_path)
        >>> filled_dst = fill_nodata(dst, nodata_nan=True, max_search=10, smoothing=1, return_array=False)
        >>> print(type(filled_dst))
        <class 'osgeo.gdal.Dataset'>
        >>> filled_array = fill_nodata(dst, nodata_nan=True, max_search=10, smoothing=1, return_array=True)
        >>> print(type(filled_array))
        <class 'numpy.ndarray'>
    """
    file_path = dst.GetDescription()
    
    if release_memory:
        # Release memory before processing
        dst = None
        
    if return_array:
        with ProcessPoolExecutor() as executor:
            future = executor.submit(
                _fill_nodata,
                file_path,
                nodata_nan,
                max_search,
                smoothing,
                return_array
            )
        result = future.result()
    else:
        result = _fill_nodata(
            file_path,
            nodata_nan,
            max_search,
            smoothing,
            return_array
        )
    return result


def _fill_nodata(
    file_path: str,
    nodata_nan: bool = True,
    max_search: int = 10,
    smoothing: int = 1,
    return_array: bool = True
) -> Union[gdal.Dataset, np.array]:
    """
    ## Summary:
        Fill in the value of NoData using the surrounding values.
    Args:
        file_path (str):
            Path to the raster file.
        nodata_nan (bool): 
            Convert NoData values to NaN. Default is True.
        max_search (int): 
            Maximum search distance for filling NoData values. Default is 10.
        smoothing (int): 
            Number of smoothing iterations. Default is 1.
        return_array (bool): 
            Return a numpy array instead of a gdal.Dataset. Default is True.
            Note that it is harder to control memory usage in the case of gdal.Dataset.
    Returns:
        gdal.Dataset: 
            gdal.Dataset with NoData values filled in.
        np.array: 
            Numpy array with NoData values filled in if return_array is True.
    """
    # Create a temporary file to write the filled data.
    dst = gdal.Open(file_path)
    write_temp_dataset = copy_temppfile(dst)
    if nodata_nan:
        # Convert NoData values to NaN in the dataset.
        write_temp_dataset.dataset = nodata_to_nan(write_temp_dataset.dataset)
    # Create a temporary file to use as a mask.
    mask_temp_dataset = copy_temppfile(write_temp_dataset.dataset)
    for i in range(dst.RasterCount):
        # Fill in the NoData values for each band.
        write_band = write_temp_dataset.dataset.GetRasterBand(i + 1)
        mask_band = mask_temp_dataset.dataset.GetRasterBand(i + 1)
        gdal.FillNodata(
            write_band,
            mask_band,
            maxSearchDist=max_search,
            smoothingIterations=smoothing,
        )
    write_temp_dataset.dataset.FlushCache()
    mask_temp_dataset.close()
    if return_array:
        # Read the filled data into a numpy array.
        result_array = write_temp_dataset.dataset.ReadAsArray()
        write_temp_dataset.close()
        return result_array
    return write_temp_dataset.dataset


def expansion_dst(
    dst: gdal.Dataset,
    vertical_cells: int,
    horizontal_cells: int,
    datum_name: str = 'JGD2011',
    memory: bool = False,
    return_array: bool = True
) -> Union[gdal.Dataset, np.array]:
    """
    ## Summary:
        Expand the size of a gdal.Dataset by adding cells to the top, bottom, left, and right.
    Args:
        dst (gdal.Dataset): 
            gdal.Dataset to be expanded.
        vertical_cells (int): 
            Number of cells to add vertically.
        horizontal_cells (int): 
            Number of cells to add horizontally.
        datum_name (str): 
            Name of the datum. Default is 'JGD2011'.
        memory (bool): 
            Whether to use memory for the new dataset. Default is False.
        return_array (bool): 
            Return a numpy array instead of a gdal.Dataset. Default is True.
    Returns:
        gdal.Dataset or np.array:
            Expanded data of gdal.Dataset or numpy array.
    """
    # Calculate the resolution of the dataset
    new_transform = _calc_expanded_transform(
        dst, 
        vertical_cells, 
        horizontal_cells,
        datum_name=datum_name
    )
    # Create a new array with the expanded size
    expanded_ary = _get_expanded_array(dst, vertical_cells, horizontal_cells)
    # Set the NoData value to NaN
    if return_array:
        return expanded_ary
    # Create a new dataset with the expanded size
    options = {
        'xsize': expanded_ary.shape[1],
        'ysize': expanded_ary.shape[0],
        'bands': dst.RasterCount,
        'eType': dst.GetRasterBand(1).DataType,
    }
    if memory:
        new_dst = copy_memory(dst, **options)
    else:
        temp_file = copy_temppfile(dst, **options)
        new_dst = temp_file.dataset
    # Set the new transform and write the array to the new dataset
    new_dst.SetGeoTransform(new_transform)
    new_dst.GetRasterBand(1).WriteArray(expanded_ary)
    new_dst.FlushCache()
    return new_dst
    

def _calc_expanded_transform(
    dst: gdal.Dataset, 
    vertical_cells: int, 
    horizontal_cells: int,
    datum_name: str = 'JGD2011'
) -> list:
    unit_name = crs_unit_name(dst.GetProjection())
    digit = 3 if unit_name == 'metre' else 9
    resolution = resolution_from_dataset(
        dst, 
        unit=unit_name, 
        digit=digit, 
        datum_name=datum_name
    )
    # Calculate the expanded size of the dataset
    x_direction_expantion = resolution.x * horizontal_cells
    y_direction_expantion = resolution.y * vertical_cells
    dst_transform = list(dst.GetGeoTransform())
    dst_transform[0] -= x_direction_expantion
    dst_transform[3] += y_direction_expantion
    return dst_transform


def _get_expanded_array(
    dst: gdal.Dataset, 
    vertical_cells: int, 
    horizontal_cells: int
) -> np.array:
    # 配列の次元数分繰り返す処理に変更したい
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    org_ary = dst.ReadAsArray()
    upper_ary = org_ary[0, :]
    upper_ary = np.hstack([
        np.zeros(horizontal_cells) + upper_ary[0],
        upper_ary,
        np.zeros(horizontal_cells) + upper_ary[-1]
    ])
    upper_ary = np.vstack([upper_ary for _ in range(vertical_cells)])
    lower_ary = org_ary[-1, :]
    lower_ary = np.hstack([
        np.zeros(horizontal_cells) + lower_ary[0],
        lower_ary,
        np.zeros(horizontal_cells) + lower_ary[-1]
    ])
    lower_ary = np.vstack([lower_ary for _ in range(vertical_cells)])
    left_ary = np.hstack([org_ary[:, : 1] for _ in range(horizontal_cells)])
    right_ary = np.hstack([org_ary[:, -1:] for _ in range(horizontal_cells)])
    result_ary = np.hstack([left_ary, org_ary, right_ary])
    result_ary = np.vstack([upper_ary, result_ary, lower_ary])
    return result_ary
    


def resample_with_specified_resolution():
    pass


def _resample_option_template_with_resolution(
    bounds: Bounds,
    x_resolution: float,
    y_resolution: float,
    resample_algorithm: int = gdal.GRA_CubicSpline
) -> gdal.WarpOptions:
    return gdal.WarpOptions(
            format="MEM",
            xRes=x_resolution,
            yRes=y_resolution,
            resampleAlg=resample_algorithm,
            outputBounds=bounds
        )


def resample_with_specified_cells():
    pass

