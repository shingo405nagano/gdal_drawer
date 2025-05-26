import os

from osgeo import gdal
import pyproj
import pytest

from gdal_drawer.tests.data import (
    TEMP_DST_1D_MERCATOR, 
    TEMP_DST_1D_DEGREE
)
from gdal_drawer.utils.config import Bounds, CellSize
from gdal_drawer.utils.raster_info import (
    get_bounds,
    estimate_utm_crs_from_datasets,
    resolution_from_dataset,
)


def test_get_bounds():
    """Test the get_bounds function."""
    bounds = get_bounds(TEMP_DST_1D_MERCATOR)
    assert isinstance(bounds, Bounds)


@pytest.mark.parametrize(
    "dst, datum_name, expected_epsg",
    [
        (TEMP_DST_1D_MERCATOR, 'JGD2011', 6691),
        (TEMP_DST_1D_MERCATOR, 'WGS 84', 32654),
    ]
)
def test_estimate_utm_crs_from_datasets(dst, datum_name, expected_epsg):
    """Test the estimate_utm_crs_from_datasets function."""
    crs = estimate_utm_crs_from_datasets(dst, datum_name)
    assert crs.to_epsg() == expected_epsg
    assert isinstance(crs, pyproj.CRS)


@pytest.mark.parametrize(
    "dst, unit",
    [
        (TEMP_DST_1D_MERCATOR, 'metre'),
        (TEMP_DST_1D_MERCATOR, 'metre'),
        (TEMP_DST_1D_MERCATOR, 'degree'),
        (TEMP_DST_1D_DEGREE, 'metre'),
    ]
)
def test_resolution_from_dataset(dst, unit):
    """Test the resolution_from_dataset function."""
    result = resolution_from_dataset(dst, unit)
    assert isinstance(result, CellSize)
    with pytest.raises(ValueError):
        resolution_from_dataset(dst, 'invalid_unit')
