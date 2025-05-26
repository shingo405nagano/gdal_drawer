import os
import tempfile

import numpy as np
from osgeo import gdal
import pytest

from gdal_drawer.tests.data import TEMP_DST_1D_MERCATOR, TEMP_FILE_1D_MERCATOR
from gdal_drawer.utils.preprocessing import (
    expansion_dst,
    fill_nodata,
    _fill_nodata,
    nodata_to_nan,
    copy_memory,
    copy_temppfile,    
)



def test_copy_temppfile():
    """Test the copy_temppfile function."""
    temp_file_dataset = copy_temppfile(TEMP_DST_1D_MERCATOR)
    assert temp_file_dataset.dataset is not None
    assert os.path.exists(temp_file_dataset.temp_file.name)
    assert temp_file_dataset.temp_file.name.endswith('.tif')
    temp_file_dataset.close()
    assert not os.path.exists(temp_file_dataset.temp_file.name)
    assert temp_file_dataset.dataset is None
    

def test_copy_memory():
    """Test the copy_memory function."""
    memory_dataset = copy_memory(TEMP_DST_1D_MERCATOR)
    assert memory_dataset is not None
    assert memory_dataset.GetDriver().ShortName == 'MEM'
    memory_dataset = None  # Close the dataset
    assert memory_dataset is None
    

def test_nodata_to_nan():
    """Test the nodata_to_nan function."""
    org_ary = TEMP_DST_1D_MERCATOR.GetRasterBand(1).ReadAsArray()
    org_nodata = TEMP_DST_1D_MERCATOR.GetRasterBand(1).GetNoDataValue()
    org_nodata_count = org_ary[org_ary == org_nodata].size
    
    nodata_dst = nodata_to_nan(TEMP_DST_1D_MERCATOR)
    result_ary = nodata_dst.GetRasterBand(1).ReadAsArray()
    result_nodata = nodata_dst.GetRasterBand(1).GetNoDataValue()
    result_nodata_count = result_ary[np.isnan(result_ary)].size
    
    assert org_nodata != result_nodata
    assert org_nodata_count == result_nodata_count
    assert result_ary[result_ary == org_nodata].size == 0
    

@pytest.mark.parametrize(
    "nodata_nan, return_array, result_type",
    [
        (False, True, np.ndarray),
        (True, False, gdal.Dataset)
    ],   
)
def test_fill_nodata(nodata_nan, return_array, result_type):
    """Test the fill_nodata function."""
    org_ary = TEMP_DST_1D_MERCATOR.GetRasterBand(1).ReadAsArray()
    org_nodata_val = TEMP_DST_1D_MERCATOR.GetRasterBand(1).GetNoDataValue()
    if np.isnan(org_nodata_val):
        org_nodata_count = org_ary[np.isnan(org_ary)].size
    else:
        org_nodata_count = org_ary[org_ary == org_nodata_val].size
    result = fill_nodata(
        TEMP_DST_1D_MERCATOR,
        nodata_nan=nodata_nan,
        return_array=return_array
    )
    assert isinstance(result, result_type)
    if isinstance(result, gdal.Dataset):
        result_ary = result.GetRasterBand(1).ReadAsArray()
    else:
        result_ary = result
    if nodata_nan:
        result_nodata_count = result_ary[np.isnan(result_ary)].size
    else:
        result_nodata_count = result_ary[result_ary == org_nodata_val].size
    assert result_nodata_count < org_nodata_count
    

@pytest.mark.parametrize(
    "file_path, return_array, result_type",
    [
        (TEMP_FILE_1D_MERCATOR.name, True, np.ndarray),
    ]
)
def _fill_nodata(file_path, return_array, result_type):
    """Test the fill_nodata function with a file path."""
    res = _fill_nodata(file_path, return_array=return_array)
    assert isinstance(res, result_type)
    
    
@pytest.mark.parametrize(
    "dst, memory, return_array, result_type",
    [
        (TEMP_DST_1D_MERCATOR, False, False, gdal.Dataset),
        (TEMP_DST_1D_MERCATOR, True, False, gdal.Dataset),
        (TEMP_DST_1D_MERCATOR, False, True, np.ndarray),
    ],   
)
def test_expansion_dst(dst, memory, return_array, result_type):
    """Test the expansion_dst function."""
    vertical = 5
    horizontal = 10
    org_ary = dst.GetRasterBand(1).ReadAsArray()
    org_shape = org_ary.shape
    result = expansion_dst(
        dst=dst,
        vertical_cells=vertical,
        horizontal_cells=horizontal,
        memory=memory,
        return_array=return_array,
    )
    if isinstance(result, gdal.Dataset):
        result_ary = result.GetRasterBand(1).ReadAsArray()
    else:
        result_ary = result
    result_shape = result_ary.shape
    assert isinstance(result, result_type)
    assert org_shape[0] + vertical * 2 == result_shape[0]
    assert org_shape[1] + horizontal * 2 == result_shape[1]
    