import pytest
import shapely

from gdal_drawer.utils.geometry import (
    estimate_utm_crs,
    estimate_utm_crs_from_geometry,
    reprojection_geometry,
)


@pytest.mark.parametrize(
    "index, lon, lat, datum_name, epsg",
    [
        (0, 133, 35, "JGD2011", 6690),
        (1, 139, 35, "JGD2011", 6691),
        (2, 145, 35, "JGD2011", 6692),
        (3, 140, 35, "WGS84", 32654),
    ],   
)
def test_estimate_utm_crs(index, lon, lat, datum_name, epsg):
    """Test the estimate_utm_crs function."""
    result = estimate_utm_crs(lon, lat, datum_name)
    assert result.to_epsg() == epsg
    if index == 0:
        with pytest.raises(IndexError):
            estimate_utm_crs(120, 35, datum_name)
            estimate_utm_crs(180, 35, datum_name)
                

@pytest.mark.parametrize(
    "geometry, in_crs, out_crs, expected_bounds",
    [
        (shapely.Point(133, 35), 4326, 6691, (-230827.85931852192, 3902420.595922888)),
        (shapely.Point(133, 35), 4326, 6678, (-715784.2967959298, -526711.7336266733)),
        ('POINT (133 35)', 4326, 6691, (-230827.85931852192, 3902420.595922888)),
        ('POINT (133 35)', 4326, 4326, (133, 35)),
    ],
)
def test_reprojection_geometry(geometry, in_crs, out_crs, expected_bounds):
    """Test the reprojection_geometry function."""
    result = reprojection_geometry(geometry, in_crs, out_crs)
    assert result.x == expected_bounds[0]
    assert result.y == expected_bounds[1]
    

@pytest.mark.parametrize(
    "geometry, in_crs, datum_name, expected_epsg",
    [
        (shapely.Point(133, 35), 4326, 'JGD2011', 6690),
        (shapely.Point(0, 0), 6678, 'JGD2011', 6691),
    ]
)
def test_estimate_utm_crs_from_geometry(geometry, in_crs, datum_name, expected_epsg):
    """Test the estimate_utm_crs_from_geometry function."""
    crs = estimate_utm_crs_from_geometry(geometry, in_crs, datum_name)
    assert crs.to_epsg() == expected_epsg

