from osgeo import gdal
import pyproj
import shapely

from gdal_drawer.utils.geometry import (
    crs_unit_name,
    estimate_utm_crs_from_geometry,
    reprojection_geometry
)
from gdal_drawer.utils.config import Bounds, CellSize


def get_bounds(dst: gdal.Dataset) -> Bounds:
    """
    ## Summary:
        Get the bounds of the gdal.Dataset.
    Args:
        dst (gdal.Dataset): gdal.Dataset
    Returns:
        Bounds(NamedTuple): (x_min, y_min, x_max, y_max)
    Examples:
        >>> dst = gdal.Open('path/to/raster.tif')
        >>> bounds: Bounds = bounds(dst)
        >>> print(bounds)
        Bounds(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
    """
    transform = dst.GetGeoTransform()
    x_min = transform[0]
    y_max = transform[3]
    rows = dst.RasterYSize
    cols = dst.RasterXSize
    x_max = x_min + cols * transform[1]
    y_min = y_max + rows * transform[-1]
    return Bounds(x_min, y_min, x_max, y_max)


def estimate_utm_crs_from_datasets(
    dst: gdal.Dataset,
    datum_name: str = 'JGD2011'
) -> pyproj.CRS:
    """
    ## Summary:
        Estimate the UTM CRS from the gdal.Dataset.
    Args:
        dst (gdal.Dataset): 
            gdal.Dataset
        datum_name (str): 
            'WGS 84', 'JGD2011' ...  default='JGD2011'\n
            https://en.wikipedia.org/wiki/Geodetic_datum
    Returns:
        (pyproj.CRS): UTM CRS
    Examples:
        >>> dst: gdal.Dataset = gdal.Open('path/to/raster.tif')
        >>> crs: str = estimate_utm_crs_from_datasets(dst)
        >>> print(crs)
        EPSG:6691
        >>> print(type(crs))
        <class 'pyproj.crs.crs.CRS'>
    """
    bounds = get_bounds(dst)
    return estimate_utm_crs_from_geometry(
        geometry=shapely.box(*bounds),
        in_crs=dst.GetProjection(),
        datum_name=datum_name
    )


def resolution_from_dataset(
    dst: gdal.Dataset, 
    unit: str = 'metre', 
    digit: int = 3, 
    datum_name: str = 'JGD2011'
) -> CellSize:
    """
    gdal.Datasetの解像度を取得する。
    Args:
        dst (gdal.Dataset): 
            gdal.Dataset
        unit (str):
            Unit of the resolution for the output.
            - 'metre'
            - 'degree'
        digit (int): 
            Number of digits to round the resolution.
            Default is 3.
        datum_name (str):
            'WGS 84', 'JGD2011' ...  default='JGD2011'\n
            https://en.wikipedia.org/wiki/Geodetic_datum
    Returns:
        CellSize(NamedTuple): (x, y)
            - x: resolution in x direction
            - y: resolution in y direction
    Examples:
        >>> dst: gdal.Dataset = gdal.Open('path/to/raster.tif')
        >>> resol: CellSize = resolution(dst, unit='metre', digit=3, datum_name='JGD2011')
        >>> print(resol)
        CellSize(x=1.048, y=1.048)
    """
    # Get the original resolution
    org_crs = pyproj.CRS.from_wkt(dst.GetProjection())
    org_unit_name = crs_unit_name(org_crs)
    bounds = get_bounds(dst)
    if org_unit_name != unit:
        # Convert the resolution to the desired unit
        if unit == 'metre':
            # If the CRS is in degrees, convert to metres.
            utm_crs = estimate_utm_crs_from_geometry(
                geometry=shapely.box(*bounds),
                in_crs=org_crs,
                datum_name=datum_name
            )
            shape_bounds = reprojection_geometry(
                geometry=shapely.box(*bounds),
                in_crs=org_crs,
                out_crs=utm_crs
            )
            bounds = Bounds(*shape_bounds.bounds)
        elif unit == 'degree':
            # If the CRS is in metres, convert to degrees.
            wgs_84_crs = pyproj.CRS.from_epsg(4326)
            geom = reprojection_geometry(
                geometry=shapely.box(*bounds),
                in_crs=org_crs,
                out_crs=wgs_84_crs
            )
            bounds = Bounds(*geom.bounds)
        else:
            raise ValueError(f"Invalid unit from raster dataset: {unit}"
                             "must be 'metre' or 'degree'")
    # Get the resolution
    x_len = abs(bounds.x_max - bounds.x_min)
    y_len = abs(bounds.y_max - bounds.y_min)
    x_resol = round(x_len / dst.RasterXSize, digit)
    y_resol = round(y_len / dst.RasterYSize, digit)
    return CellSize(x=x_resol, y=y_resol)

