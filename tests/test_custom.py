import os
import tempfile

import geopandas as gpd
import numpy as np
from osgeo import gdal
import pandas as pd
import pytest
import pyproj
import shapely

from gdal_drawer.custom import CustomGdalDataset
from gdal_drawer.tests.data import (
    TEMP_DST_1D_MERCATOR,
    TEMP_DST_1D_DEGREE,
    TEMP_DST_3D_MERCATOR,
)
from gdal_drawer.utils.geometry import reprojection_geometry
from gdal_drawer.utils.colors import dimensional_count
from gdal_drawer.utils.config import Bounds, CellSize
from gdal_drawer.utils.preprocessing import copy_memory


def test_instantiate_custom_gdal_dataset():
    """Test the instantiation of CustomGdalDataset."""
    dst = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
    with pytest.raises(Exception):
        CustomGdalDataset('invalid_path')
        non_crs_dst = copy_memory(TEMP_DST_1D_MERCATOR)
        non_crs_dst.SetProjection('')
        CustomGdalDataset(non_crs_dst)


global CUSTOM_TEMP_DST_1D_MERCATOR
CUSTOM_TEMP_DST_1D_MERCATOR = CustomGdalDataset(TEMP_DST_1D_MERCATOR)
global CUSTOM_TEMP_DST_1D_DEGREE
CUSTOM_TEMP_DST_1D_DEGREE = CustomGdalDataset(TEMP_DST_1D_DEGREE)
global CUSTOM_TEMP_DST_3D_MERCATOR
CUSTOM_TEMP_DST_3D_MERCATOR = CustomGdalDataset(TEMP_DST_3D_MERCATOR)
        


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
    res = CUSTOM_TEMP_DST_1D_MERCATOR.x_resolution
    assert isinstance(res, int | float)


def test_y_resolution_check_from_custom_gdal_datasets_cls():
    """Test the y_resolution_check_from_custom_gdal_datasets_cls function."""
    res = CUSTOM_TEMP_DST_1D_MERCATOR.y_resolution
    assert isinstance(res, int | float)
    

# def test_array_from_custom_gdal_datasets_cls():
#     """Test the array_from_custom_gdal_datasets_cls function."""
#     res = CUSTOM_TEMP_DST_1D_MERCATOR.array()
#     assert isinstance(res, np.ndarray)
#     assert dimensional_count(res) == 2
    
#     res = CUSTOM_TEMP_DST_1D_MERCATOR.array(1)
#     assert isinstance(res, np.ndarray)
#     assert dimensional_count(res) == 2
    
#     res = CUSTOM_TEMP_DST_1D_MERCATOR.array([1])
#     assert isinstance(res, np.ndarray)
#     assert dimensional_count(res) == 2
    
#     res = CUSTOM_TEMP_DST_3D_MERCATOR.array()
#     assert isinstance(res, np.ndarray)
    
#     with pytest.raises(Exception):
#         CUSTOM_TEMP_DST_3D_MERCATOR.array(0)
        

# def test_array_of_image_from_custom_gdal_datasets_cls():
#     """Test the array_of_image_from_custom_gdal_datasets_cls function."""
#     with pytest.raises(Exception):
#         CUSTOM_TEMP_DST_1D_MERCATOR.array_of_image()
    
#     res = CUSTOM_TEMP_DST_3D_MERCATOR.array_of_image()
#     assert isinstance(res, np.ndarray)
#     assert dimensional_count(res) == 3
    

@pytest.mark.parametrize("memory", [True, False])
def test_copy_dataset_from_custom_gdal_datasets_cls(memory):
    """Test the copy_dataset_from_custom_gdal_datasets_cls function."""
    array = CUSTOM_TEMP_DST_1D_MERCATOR.array()
    new_dst = CUSTOM_TEMP_DST_1D_MERCATOR.copy_dataset(memory=memory)
    new_array = new_dst.array()
    assert isinstance(new_dst, CustomGdalDataset)
    total = np.nansum(array[array != new_array])
    assert total == 0
    assert id(CUSTOM_TEMP_DST_1D_MERCATOR) != id(new_dst)
    
    
def test_save_dst_from_custom_gdal_datasets_cls():
    """Test the save_dst_from_custom_gdal_datasets_cls function."""
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
        temp_file_path = temp_file.name
        CUSTOM_TEMP_DST_1D_DEGREE.save_dst(temp_file_path)
        assert os.path.exists(temp_file_path)
    

def test_write_ary_to_mem_from_custom_gdal_datasets_cls():
    """Test the write_ary_to_mem_from_custom_gdal_datasets_cls function."""
    # Single band dataset
    org_array = CUSTOM_TEMP_DST_1D_MERCATOR.array()
    org_array_total = np.nansum(org_array)
    org_array -= 100
    new_dst = CUSTOM_TEMP_DST_1D_MERCATOR.write_ary_to_mem(org_array)
    new_array = new_dst.array()
    new_array_total = np.nansum(new_array)
    assert isinstance(new_dst, CustomGdalDataset)
    assert org_array_total != new_array_total
    assert id(CUSTOM_TEMP_DST_1D_MERCATOR) != id(new_dst)
    # Multi band dataset
    org_array = CUSTOM_TEMP_DST_3D_MERCATOR.array()
    org_array_total = np.nansum(org_array)
    org_array *= 0
    new_dst = CUSTOM_TEMP_DST_3D_MERCATOR.write_ary_to_mem(
        org_array, 
        data_type=gdal.GDT_Byte,
        out_nodata=0,
        raster_count=3
    )
    new_array = new_dst.array()
    new_array_total = np.nansum(new_array)
    assert isinstance(new_dst, CustomGdalDataset)
    assert org_array_total != new_array_total
    assert id(CUSTOM_TEMP_DST_3D_MERCATOR) != id(new_dst)
    

def test_fill_nodata_from_custom_gdal_datasets_cls():
    """Test the fill_nodata_from_custom_gdal_datasets_cls function."""
    # Single band dataset
    single_ary = CUSTOM_TEMP_DST_1D_MERCATOR.array()
    single_nodata = CUSTOM_TEMP_DST_1D_MERCATOR.GetRasterBand(1).GetNoDataValue()
    single_nodata_count = \
        single_ary[np.isnan(single_ary)].size if np.isnan(single_nodata) else \
        single_ary[single_ary == single_nodata].size
    filled_single_dst = CUSTOM_TEMP_DST_1D_MERCATOR.fill_nodata()
    filled_single_ary = filled_single_dst.array()
    filled_single_nodata = filled_single_dst.GetRasterBand(1).GetNoDataValue()
    filled_single_nodata_count = \
        filled_single_ary[np.isnan(filled_single_ary)].size if np.isnan(filled_single_nodata) else \
        filled_single_ary[filled_single_ary == filled_single_nodata].size
    assert filled_single_nodata_count < single_nodata_count
    # Multi band dataset
    multi_ary = CUSTOM_TEMP_DST_3D_MERCATOR.array()
    multi_nodata = CUSTOM_TEMP_DST_3D_MERCATOR.GetRasterBand(1).GetNoDataValue()
    multi_nodata_count = \
        multi_ary[np.isnan(multi_ary)].size if np.isnan(multi_nodata) else \
        multi_ary[multi_ary == multi_nodata].size
    filled_multi_ary = CUSTOM_TEMP_DST_3D_MERCATOR.fill_nodata(return_array=True)
    filled_multi_nodata = 0
    filled_multi_nodata_count = filled_multi_ary[filled_multi_ary == filled_multi_nodata].size
    assert filled_multi_nodata_count < multi_nodata_count
    

@pytest.mark.parametrize(
    "vertical_cells, horizontal_cells, datum_name, return_array",
    [
        (10, 20, 'JGD2011', True),
        (20, 10, 'WGS84', False)
    ]
)
def test_expantion_dst_from_custom_gdal_datasets_cls(vertical_cells, horizontal_cells, datum_name, return_array):
    """Test the expantion_dst_from_custom_gdal_datasets_cls function."""
    expd_dst = CUSTOM_TEMP_DST_1D_MERCATOR.expansion_dst(
        vertical=vertical_cells,
        horizontal=horizontal_cells,
        datum_name=datum_name,
        return_array=return_array
    )
    if return_array:
        rows, cols = expd_dst.shape
    else:
        rows, cols = expd_dst.RasterYSize, expd_dst.RasterXSize
    assert CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize + horizontal_cells * 2 == cols
    assert CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + vertical_cells * 2 == rows


def test_bounds_from_custom_gdal_datasets_cls():
    """Test the bounds_from_custom_gdal_datasets_cls function."""
    bounds = CUSTOM_TEMP_DST_1D_MERCATOR.bounds()
    assert isinstance(bounds, Bounds)
    transform = CUSTOM_TEMP_DST_1D_MERCATOR.GetGeoTransform()
    x_min = transform[0]
    y_max = transform[3]
    rows = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize
    cols = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    x_max = x_min + cols * transform[1]
    y_min = y_max + rows * transform[-1]
    assert bounds.x_min == x_min
    assert bounds.y_min == y_min
    assert bounds.x_max == x_max
    assert bounds.y_max == y_max


def test_reprojected_bounds_from_custom_gdal_datasets_cls():
    """Test the reprojected_bounds_from_custom_gdal_datasets_cls function."""
    out_crs = pyproj.CRS.from_epsg(4326)
    bounds = CUSTOM_TEMP_DST_1D_MERCATOR.reprojected_bounds(out_crs=out_crs)
    assert isinstance(bounds, Bounds)
    geometry = shapely.box(*CUSTOM_TEMP_DST_1D_MERCATOR.bounds())
    reprojected_geom = reprojection_geometry(
        geometry=geometry,
        in_crs=CUSTOM_TEMP_DST_1D_MERCATOR.GetProjection(),
        out_crs= out_crs
    )
    reprojected_bounds = Bounds(*reprojected_geom.bounds)
    assert bounds.x_min == reprojected_bounds.x_min
    assert bounds.y_min == reprojected_bounds.y_min
    assert bounds.x_max == reprojected_bounds.x_max
    assert bounds.y_max == reprojected_bounds.y_max


def test_center_from_custom_gdal_datasets_cls():
    """Test the center_from_custom_gdal_datasets_cls function."""
    center = CUSTOM_TEMP_DST_1D_MERCATOR.center()
    assert isinstance(center, tuple)
    bounds = CUSTOM_TEMP_DST_1D_MERCATOR.bounds()
    x_center = (bounds.x_min + bounds.x_max) / 2
    y_center = (bounds.y_min + bounds.y_max) / 2
    assert center.x == pytest.approx(x_center, rel=1e-5)
    assert center.y == pytest.approx(y_center, rel=1e-5)
    reprojected_center = CUSTOM_TEMP_DST_1D_MERCATOR.center(out_crs=pyproj.CRS.from_epsg(4326))
    assert isinstance(reprojected_center, tuple)
    assert reprojected_center.x != center.x
    assert reprojected_center.y != center.y
    

def tet_cells_center_coordinates_from_custom_gdal_datasets_cls():
    """Test the cells_center_coordinates_from_custom_gdal_datasets_cls function."""
    coordinates = CUSTOM_TEMP_DST_1D_MERCATOR.cells_center_coordinates()
    assert isinstance(coordinates.X, np.ndarray)
    assert isinstance(coordinates.Y, np.ndarray)
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    assert coordinates.X.size == pytest.approx(raster_size, abs=allowable)
    assert coordinates.Y.size == pytest.approx(raster_size, abs=allowable)
    

def test_cells_upper_left_corner_coordinates_from_custom_gdal_datasets_cls():
    """Test the cells_upper_left_corner_coordinates_from_custom_gdal_datasets_cls function."""
    coordinates = CUSTOM_TEMP_DST_1D_MERCATOR.cells_upper_left_corner_coordinates()
    assert isinstance(coordinates.X, np.ndarray)
    assert isinstance(coordinates.Y, np.ndarray)
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize 
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    assert coordinates.X.size == pytest.approx(raster_size, abs=allowable)
    assert coordinates.Y.size == pytest.approx(raster_size, abs=allowable)
    

def test_cells_upper_right_corner_coordinates_from_custom_gdal_datasets_cls():
    """Test the cells_upper_right_corner_coordinates_from_custom_gdal_datasets_cls function."""
    coordinates = CUSTOM_TEMP_DST_1D_MERCATOR.cells_upper_right_corner_coordinates()
    assert isinstance(coordinates.X, np.ndarray)
    assert isinstance(coordinates.Y, np.ndarray)
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize 
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    assert coordinates.X.size == pytest.approx(raster_size, abs=allowable)
    assert coordinates.Y.size == pytest.approx(raster_size, abs=allowable)


def test_cells_lower_left_corner_coordinates_from_custom_gdal_datasets_cls():
    """Test the cells_lower_left_corner_coordinates_from_custom_gdal_datasets_cls function."""
    coordinates = CUSTOM_TEMP_DST_1D_MERCATOR.cells_lower_left_corner_coordinates()
    assert isinstance(coordinates.X, np.ndarray)
    assert isinstance(coordinates.Y, np.ndarray)
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize 
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    assert coordinates.X.size == pytest.approx(raster_size, abs=allowable)
    assert coordinates.Y.size == pytest.approx(raster_size, abs=allowable)
    
    
def test_cells_lower_right_corner_coordinates_from_custom_gdal_datasets_cls():
    """Test the cells_lower_right_corner_coordinates_from_custom_gdal_datasets_cls function."""
    coordinates = CUSTOM_TEMP_DST_1D_MERCATOR.cells_lower_right_corner_coordinates()
    assert isinstance(coordinates.X, np.ndarray)
    assert isinstance(coordinates.Y, np.ndarray)
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize 
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    assert coordinates.X.size == pytest.approx(raster_size, abs=allowable)
    assert coordinates.Y.size == pytest.approx(raster_size, abs=allowable)
 

def test_to_geodataframe_xy():
    """Test the to_geodataframe_xy function."""
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize 
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    
    center_gdf = CUSTOM_TEMP_DST_1D_MERCATOR.to_geodataframe_xy(position='center')
    assert isinstance(center_gdf, gpd.GeoDataFrame)
    assert len(center_gdf) == pytest.approx(raster_size, abs=allowable)
    del center_gdf
    
    ulc_gdf = CUSTOM_TEMP_DST_1D_MERCATOR.to_geodataframe_xy(position='upper_left')
    assert isinstance(ulc_gdf, gpd.GeoDataFrame)
    assert len(ulc_gdf) == pytest.approx(raster_size, abs=allowable)
    del ulc_gdf
    
    urc_gdf = CUSTOM_TEMP_DST_1D_MERCATOR.to_geodataframe_xy(position='upper_right')
    assert isinstance(urc_gdf, gpd.GeoDataFrame)
    assert len(urc_gdf) == pytest.approx(raster_size, abs=allowable)
    del urc_gdf
    
    llc_gdf = CUSTOM_TEMP_DST_1D_MERCATOR.to_geodataframe_xy(position='lower_left')
    assert isinstance(llc_gdf, gpd.GeoDataFrame)
    assert len(llc_gdf) == pytest.approx(raster_size, abs=allowable)
    del llc_gdf
    
    lrc_gdf = CUSTOM_TEMP_DST_1D_MERCATOR.to_geodataframe_xy(position='lower_right')
    assert isinstance(lrc_gdf, gpd.GeoDataFrame)
    assert len(lrc_gdf) == pytest.approx(raster_size, abs=allowable)
    del lrc_gdf
    
    
def test_to_pandas_xy():
    """Test the to_pandas_xy function."""
    raster_size = CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize * CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize 
    allowable = CUSTOM_TEMP_DST_1D_MERCATOR.RasterYSize + CUSTOM_TEMP_DST_1D_MERCATOR.RasterXSize
    
    center_df = CUSTOM_TEMP_DST_1D_MERCATOR.to_pandas_xy(position='center')
    assert isinstance(center_df, pd.DataFrame)
    assert len(center_df) == pytest.approx(raster_size, abs=allowable)
    del center_df
    

@pytest.mark.parametrize(
    "dst, excpected",
    [
        (CUSTOM_TEMP_DST_1D_MERCATOR, True),
        (CUSTOM_TEMP_DST_1D_DEGREE, False),
        (CUSTOM_TEMP_DST_3D_MERCATOR, True),
    ]
)
def test_check_crs_is_metre(dst, excpected):
    """Test the check_crs_is_metre function."""
    assert dst.check_crs_is_metre() == excpected
    
    
def test_estimate_utm_crs_from_custom_gdal_datasets_cls():
    """Test the estimate_utm_crs_from_custom_gdal_datasets_cls function."""
    crs = CUSTOM_TEMP_DST_1D_MERCATOR.estimate_utm_crs(datum_name='JGD2011')
    assert isinstance(crs, str)
    assert pyproj.CRS(crs)
    
    crs = CUSTOM_TEMP_DST_1D_DEGREE.estimate_utm_crs(datum_name='WGS84')
    assert isinstance(crs, str)
    assert pyproj.CRS(crs)
    with pytest.raises(Exception):
        CUSTOM_TEMP_DST_1D_DEGREE.estimate_utm_crs(datum_name='dummy')
        
        
def test_cell_size_in_metre_from_custom_gdal_datasets_cls():
    """Test the cells_size_in_metre_from_custom_gdal_datasets_cls function."""
    cell_size = CUSTOM_TEMP_DST_1D_MERCATOR.cell_size_in_metre()
    assert isinstance(cell_size, CellSize)
    assert 0 < cell_size.x
    assert 0 < cell_size.y
    
    cell_size = CUSTOM_TEMP_DST_1D_DEGREE.cell_size_in_metre()
    assert isinstance(cell_size, CellSize)
    assert 0 < cell_size.x
    assert 0 < cell_size.y
    
    
def test_cell_size_in_degree_from_custom_gdal_datasets_cls():
    """Test the cells_size_in_degree_from_custom_gdal_datasets_cls function."""
    cell_size = CUSTOM_TEMP_DST_1D_DEGREE.cell_size_in_degree()
    assert isinstance(cell_size, CellSize)
    assert 0 < cell_size.x
    assert 0 < cell_size.y
    
    cell_size = CUSTOM_TEMP_DST_1D_MERCATOR.cell_size_in_degree()
    assert isinstance(cell_size, CellSize)
    assert 0 < cell_size.x
    assert 0 < cell_size.y
    
    
def test_reprojected_dataset_from_custom_gdal_datasets_cls():
    """Test the reprojected_dataset_from_custom_gdal_datasets_cls function."""
    out_crs = pyproj.CRS.from_epsg(4326)
    reprojected_dst = CUSTOM_TEMP_DST_1D_MERCATOR.reprojected_dataset(out_crs=out_crs)
    reprojected_crs = pyproj.CRS(reprojected_dst.GetProjection())
    assert isinstance(reprojected_dst, CustomGdalDataset)
    assert reprojected_crs.to_epsg() == out_crs.to_epsg()
    for org, repro in zip(CUSTOM_TEMP_DST_1D_MERCATOR.bounds(), reprojected_dst.bounds()):
        assert org != repro, "Bounds should be different after reprojection"
    
    with pytest.raises(Exception):
        CUSTOM_TEMP_DST_1D_DEGREE.reprojected_dataset(out_crs='invalid_crs')
        

def test_estimate_utm_and_reprojected_dataset_from_custom_gdal_datasets_cls():
    """Test the estimate_utm_and_reprojected_dataset_from_custom_gdal_datasets_cls function."""
    reprojected_dst = CUSTOM_TEMP_DST_1D_DEGREE.estimate_utm_and_reprojected_dataset(datum_name='WGS84')
    assert isinstance(reprojected_dst, CustomGdalDataset)
    assert reprojected_dst.GetProjection()
    for org, repro in zip(CUSTOM_TEMP_DST_1D_DEGREE.bounds(), reprojected_dst.bounds()):
        assert org != repro, "Bounds should be different after reprojection"
    assert reprojected_dst.check_crs_is_metre()
    
    with pytest.raises(Exception):
        CUSTOM_TEMP_DST_1D_MERCATOR.estimate_utm_and_reprojected_dataset(datum_name='dummy')

