import pyproj.exceptions
import pytest
import pyproj
import shapely

from gdal_drawer.utils.config import (
    crs_checker,
    geometry_checker,
)



def test_crs_checker():
    """Test the crs_checker decorator."""
    @crs_checker(0, 'crs')
    def test(crs):
        return crs
    
    expected_crs = pyproj.CRS.from_epsg(4326)
    assert test(4326) == expected_crs
    assert test('EPSG:4326') == expected_crs
    wkt_crs = pyproj.CRS.from_epsg(4326).to_wkt()
    assert test(wkt_crs) == expected_crs
    assert test(crs=4326) == expected_crs
    with pytest.raises(pyproj.exceptions.CRSError):
        test(123)
    


def test_geometry_checker():
    """Test the geometry_checker decorator."""
    @geometry_checker(0, 'geom')
    def test(geom):
        return geom
    
    expected_geom = shapely.Point(0, 1)
    assert test(expected_geom) == expected_geom
    assert test('POINT (0 1)') == expected_geom
    with pytest.raises(Exception):
        test('INVALID WKT')
        test(123)
