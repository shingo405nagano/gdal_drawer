import os
import tempfile

import numpy as np
from osgeo import gdal
import pytest
import pyproj
import shapely

from gdal_drawer.custom import CustomGdalDataset
from gdal_drawer.tests.data import (
    TEMP_DST_1D_MERCATOR,
    TEMP_DST_1D_DEGREE,
    TEMP_DST_3D_MERCATOR,
)
from gdal_drawer.utils.colors import dimensional_count
from gdal_drawer.utils.config import Bounds, CellSize



def test_check_crs_from_custom_gdal_dataset_cls():
    """Test the check_crs_from_custom_gdal_dataset_cls function."""
    class Test(object):
        @CustomGdalDataset._check_crs(0, 'crs')
        def dummy_function(self, crs):
            return crs

    test = Test()
    crs = pyproj.CRS.from_epsg(4326)
    assert isinstance(test.dummy_function(4326), str)
    assert isinstance(test.dummy_function(crs=4326), str)
    assert test.dummy_function(4326) == crs.to_wkt()
    assert test.dummy_function(crs=crs) == crs.to_wkt()
    assert test.dummy_function(crs=crs.to_wkt()) == crs.to_wkt()
    assert test.dummy_function(crs=None) is None
    with pytest.raises(Exception):
        test.dummy_function(crs=100)
    with pytest.raises(ValueError):
        test.dummy_function(crs=['100'])
        

@pytest.mark.parametrize(
    "datum_name",
    ["WGS84", "JGD2000", "JGD2011"]
)
def test_check_datum_from_custom_gdal_dataset_cls(datum_name):
    """Test the check_datum_from_custom_gdal_dataset_cls function."""
    class Test(object):
        @CustomGdalDataset._check_datum(0, 'datum_name')
        def dummy_function(self, datum_name):
            return datum_name
        
    test = Test()
    test.dummy_function(datum_name)
    with pytest.raises(ValueError):
        test.dummy_function(datum_name='dummy')
        
        

@pytest.mark.parametrize(
    "iterable, expected",
    [
        (1, True),
        ([1, 2, 3], True),
        ([[1, 2], [3, 4]], True),
        ([1.0, 2.0, 3.0], False),
        (None, True),
        ('1', False),
    ]
)
def test_is_iterable_of_ints_from_custom_gdal_datasets_cls(iterable, expected):
    """Test the is_iterable_of_ints_from_custom_gdal_datasets_cls function."""
    class Test(object):
        @CustomGdalDataset._is_iterable_of_ints(0, 'arg')
        def dummy_function(self, arg=None):
            return arg

    test = Test()
    if expected:
        test.dummy_function(iterable)
        test.dummy_function()
    else:
        with pytest.raises(Exception):
            test.dummy_function(arg=iterable)
            
            
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (shapely.Point(0, 0).wkt, True),
        ('invalid geometry', False),
        (shapely.Point(0, 0), True),
        (shapely.GeometryCollection(shapely.Point(0, 0)), False),
    ]
)
def test_wkt_geometry_check_from_custom_gdal_datasets_cls(geometry, expected):
    """Test the wkt_geometry_check_from_custom_gdal_datasets_cls function."""
    class Test(object):
        @CustomGdalDataset._wkt_geometry_check(0, 'arg')
        def dummy_function(self, arg):
            return arg
    
    test = Test()
    
    if expected:
        res = test.dummy_function(geometry)
        res = test.dummy_function(arg=geometry)
        assert isinstance(res, str)
    else:
        with pytest.raises(Exception):
            test.dummy_function(arg=geometry)
            

def test_band_check_from_custom_gdal_datasets_cls():
    """Test the band_check_from_custom_gdal_datasets_cls function."""
    class Test(object):
        def __init__(self):
            self.RasterCount = 1
            
        @CustomGdalDataset._band_check(count=1)
        def dummy_function(self, band):
            return band
        
    test = Test()
    test.dummy_function('')
    with pytest.raises(Exception):
        test.RasterCount = 2
        test.dummy_function('')
        
        
def test_x_resolution_check_from_custom_gdal_datasets_cls():
    """Test the x_resolution_check_from_custom_gdal_datasets_cls function."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    res = dst.x_resolution
    assert isinstance(res, int | float)


def test_y_resolution_check_from_custom_gdal_datasets_cls():
    """Test the y_resolution_check_from_custom_gdal_datasets_cls function."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    res = dst.y_resolution
    assert isinstance(res, int | float)
    

def test_array_from_custom_gdal_datasets_cls():
    """Test the array_from_custom_gdal_datasets_cls function."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    res = dst.array()
    assert isinstance(res, np.ndarray)
    assert dimensional_count(res) == 2
    
    res = dst.array(1)
    assert isinstance(res, np.ndarray)
    assert dimensional_count(res) == 2
    
    res = dst.array([1])
    assert isinstance(res, np.ndarray)
    assert dimensional_count(res) == 2
    
    with pytest.raises(Exception):
        dst.array(0)
        

def test_array_of_image_from_custom_gdal_datasets_cls():
    """Test the array_of_image_from_custom_gdal_datasets_cls function."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    with pytest.raises(Exception):
        dst.array_of_image()
    
    dst = CustomGdalDataset(TEMP_DST_3D_MERCATOR)
    res = dst.array_of_image()
    assert isinstance(res, np.ndarray)
    assert dimensional_count(res) == 3
    

def test_copy_dataset_from_custom_gdal_datasets_cls():
    """Test the copy_dataset_from_custom_gdal_datasets_cls function."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    array = dst.array()
    new_dst = dst.copy_dataset()
    new_array = new_dst.array()
    assert isinstance(new_dst, CustomGdalDataset)
    total = np.nansum(array[array != new_array])
    assert total == 0
    assert id(dst) != id(new_dst)
    
    
def test_save_dst_from_custom_gdal_datasets_cls():
    """Test the save_dst_from_custom_gdal_datasets_cls function."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
        temp_file_path = temp_file.name
        dst.save_dst(temp_file_path)
        assert os.path.exists(temp_file_path)
    

def test_write_ary_to_mem_from_custom_gdal_datasets_cls():
    """Test the write_ary_to_mem_from_custom_gdal_datasets_cls function."""
    # Single band dataset
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    org_array = dst.array()
    org_array_total = np.nansum(org_array)
    org_array -= 100
    new_dst = dst.write_ary_to_mem(org_array)
    new_array = new_dst.array()
    new_array_total = np.nansum(new_array)
    assert isinstance(new_dst, CustomGdalDataset)
    assert org_array_total != new_array_total
    assert id(dst) != id(new_dst)
    # Multi band dataset
    dst = CustomGdalDataset(TEMP_DST_3D_MERCATOR)
    org_array = dst.array()
    org_array_total = np.nansum(org_array)
    org_array *= 0
    new_dst = dst.write_ary_to_mem(
        org_array, 
        data_type=gdal.GDT_Byte,
        out_nodata=0,
        raster_count=3
    )
    new_array = new_dst.array()
    new_array_total = np.nansum(new_array)
    assert isinstance(new_dst, CustomGdalDataset)
    assert org_array_total != new_array_total
    assert id(dst) != id(new_dst)
    

def test_fill_nodata_from_custom_gdal_datasets_cls():
    """Test the fill_nodata_from_custom_gdal_datasets_cls function."""
    # Single band dataset
    single_dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    single_ary = single_dst.array()
    single_nodata = single_dst.GetRasterBand(1).GetNoDataValue()
    single_nodata_count = \
        single_ary[np.isnan(single_ary)].size if np.isnan(single_nodata) else \
        single_ary[single_ary == single_nodata].size
    filled_single_dst = single_dst.fill_nodata()
    filled_single_ary = filled_single_dst.array()
    filled_single_nodata = filled_single_dst.GetRasterBand(1).GetNoDataValue()
    filled_single_nodata_count = \
        filled_single_ary[np.isnan(filled_single_ary)].size if np.isnan(filled_single_nodata) else \
        filled_single_ary[filled_single_ary == filled_single_nodata].size
    assert filled_single_nodata_count < single_nodata_count
    # Multi band dataset
    multi_dst = CustomGdalDataset(TEMP_DST_3D_MERCATOR)
    multi_ary = multi_dst.array()
    multi_nodata = multi_dst.GetRasterBand(1).GetNoDataValue()
    multi_nodata_count = \
        multi_ary[np.isnan(multi_ary)].size if np.isnan(multi_nodata) else \
        multi_ary[multi_ary == multi_nodata].size
    filled_multi_ary = multi_dst.fill_nodata(return_array=True)
    filled_multi_nodata = 0
    filled_multi_nodata_count = filled_multi_ary[filled_multi_ary == filled_multi_nodata].size
    assert filled_multi_nodata_count < multi_nodata_count