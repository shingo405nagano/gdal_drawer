from typing import Optional

import numpy as np
import pyproj
import shapely
from osgeo import gdal

from gdal_drawer.utils.config import CRS, XY, Bounds, CellSize, Coordinates
from gdal_drawer.utils.geometry import (
    crs_unit_name,
    estimate_utm_crs_from_geometry,
    reprojection_geometry,
)


def get_bounds(dst: gdal.Dataset) -> Bounds:
    """
    ## Summary:
        `gdal.Dataset`のXYの最小値と最大値を取得します。
    Args:
        dst (gdal.Dataset):
    Returns:
        Bounds(NamedTuple):
            (x_min, y_min, x_max, y_max)
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


def get_reprojected_bounds(dst: gdal.Dataset, out_crs: CRS) -> Bounds:
    """
    ## Summary:
        `gdal.Dataset`のXYの最小値と最大値を指定されたCRSに再投影して取得します。
    Args:
        dst (gdal.Dataset):
        out_crs (pyproj.CRS):
    Returns:
        Bounds(NamedTuple): (x_min, y_min, x_max, y_max)
    """
    bounds = get_bounds(dst)
    geometry = shapely.box(*bounds)
    reprojected_geometry = reprojection_geometry(
        geometry=geometry, in_crs=dst.GetProjection(), out_crs=out_crs
    )
    return Bounds(*reprojected_geometry.bounds)


def get_center_from_dataset(dst: gdal.Dataset, out_crs: Optional[CRS] = None) -> XY:
    """
    ## Summary:
        `gdal.Dataset`の中心座標を取得します。
    Args:
        dst (gdal.Dataset):
        out_crs (pyproj.CRS, optional):
            出力CRS。Noneの場合は元のCRSで取得します。
    Returns:
        XY(NamedTuple):
            (x, y)
    """
    if out_crs is None:
        bounds = get_bounds(dst)
    else:
        bounds = get_reprojected_bounds(dst, out_crs)
    x_center = (bounds.x_min + bounds.x_max) / 2
    y_center = (bounds.y_min + bounds.y_max) / 2
    return XY(x=x_center, y=y_center)


def estimate_utm_crs_from_datasets(
    dst: gdal.Dataset,  #
    datum_name: str = "JGD2011",
) -> pyproj.CRS:
    """
    ## Summary:
        `gdal.Dataset`からUTM座標系を推定します。
    Args:
        dst (gdal.Dataset):
        datum_name (str):
            JGD2011, WGS84, NAD84, ETRS89, GDA94, KGD2002, TWD67, SAD69
            ...  default='JGD2011'\n
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
        geometry=shapely.box(*bounds), in_crs=dst.GetProjection(), datum_name=datum_name
    )


def resolution_from_dataset(
    dst: gdal.Dataset,  #
    unit: str = "metre",
    digit: int = 3,
    datum_name: str = "JGD2011",
) -> CellSize:
    """
    gdal.Datasetの解像度を取得する。
    Args:
        dst (gdal.Dataset):
        unit (str): 出力したい解像度の単位
            - 'metre'
            - 'degree'
        digit (int): 浮動小数点以下の桁数
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
        if unit == "metre":
            # If the CRS is in degrees, convert to metres.
            utm_crs = estimate_utm_crs_from_geometry(
                geometry=shapely.box(*bounds), in_crs=org_crs, datum_name=datum_name
            )
            shape_bounds = reprojection_geometry(
                geometry=shapely.box(*bounds), in_crs=org_crs, out_crs=utm_crs
            )
            bounds = Bounds(*shape_bounds.bounds)
        elif unit == "degree":
            # If the CRS is in metres, convert to degrees.
            wgs_84_crs = pyproj.CRS.from_epsg(4326)
            geom = reprojection_geometry(
                geometry=shapely.box(*bounds), in_crs=org_crs, out_crs=wgs_84_crs
            )
            bounds = Bounds(*geom.bounds)
        else:
            raise ValueError(
                f"Invalid unit from raster dataset: {unit}must be 'metre' or 'degree'"
            )
    # Get the resolution
    x_len = abs(bounds.x_max - bounds.x_min)
    y_len = abs(bounds.y_max - bounds.y_min)
    x_resol = round(x_len / dst.RasterXSize, digit)
    y_resol = round(y_len / dst.RasterYSize, digit)
    return CellSize(x=x_resol, y=y_resol)


def cells_center_coordinates(dst: gdal.Dataset) -> Coordinates:
    """
    ## Summary:
        `gdal.Dataset`の各セルの中心座標を取得します。
    Args:
        dst (gdal.Dataset):
    Returns:
        Coordinates(dataclass):
            1. X (np.ndarray): shape is (RasterYSize, RasterXSize)
            2. Y (np.ndarray): shape is (RasterYSize, RasterXSize)
    """
    bounds = get_bounds(dst)
    transform = dst.GetGeoTransform()
    x_resol = transform[1]
    y_resol = abs(transform[5])
    # Half is used to get the center of the cell.
    half = 0.5
    # Create a meshgrid of coordinates
    X = np.arange(bounds.x_min, bounds.x_max, x_resol) + x_resol * half
    Y = np.arange(bounds.y_min, bounds.y_max, y_resol) + y_resol * half
    return Coordinates(*np.meshgrid(X, Y))


def cells_upper_left_corner_coordinates(dst: gdal.Dataset) -> Coordinates:
    """
    ## Summary:
        `gdal.Dataset`の各セルの上端左隅の座標を取得します。
    Args:
        dst (gdal.Dataset):
    Returns:
        Coordinates(dataclass):
            1. X (np.ndarray): shape is (RasterYSize, RasterXSize)
            2. Y (np.ndarray): shape is (RasterYSize, RasterXSize)
    """
    bounds = get_bounds(dst)
    transform = dst.GetGeoTransform()
    x_resol = transform[1]
    y_resol = abs(transform[5])
    # Create a meshgrid of coordinates
    X = np.arange(bounds.x_min, bounds.x_max, x_resol)
    Y = np.arange(bounds.y_min, bounds.y_max, y_resol)
    return Coordinates(*np.meshgrid(X, Y))


def cells_upper_right_corner_coordinates(dst: gdal.Dataset) -> Coordinates:
    """
    ## Summary:
        `gdal.Dataset`の各セルの上端右隅の座標を取得します。
    Args:
        dst (gdal.Dataset):
            gdal.Dataset
    Returns:
        Coordinates(dataclass):
            1. X (np.ndarray): shape is (RasterYSize, RasterXSize)
            2. Y (np.ndarray): shape is (RasterYSize, RasterXSize)
    """
    bounds = get_bounds(dst)
    transform = dst.GetGeoTransform()
    x_resol = transform[1]
    y_resol = abs(transform[5])
    # Create a meshgrid of coordinates
    X = np.arange(bounds.x_min + x_resol, bounds.x_max + x_resol, x_resol)
    Y = np.arange(bounds.y_min, bounds.y_max, y_resol)
    return Coordinates(*np.meshgrid(X, Y))


def cells_lower_left_corner_coordinates(dst: gdal.Dataset) -> Coordinates:
    """
    ## Summary:
        `gdal.Dataset`の各セルの下端左隅の座標を取得します。
    Args:
        dst (gdal.Dataset):
    Returns:
        Coordinates(dataclass):
            1. X (np.ndarray): shape is (RasterYSize, RasterXSize)
            2. Y (np.ndarray): shape is (RasterYSize, RasterXSize)
    """
    bounds = get_bounds(dst)
    transform = dst.GetGeoTransform()
    x_resol = transform[1]
    y_resol = abs(transform[5])
    # Create a meshgrid of coordinates
    X = np.arange(bounds.x_min, bounds.x_max, x_resol)
    Y = np.arange(bounds.y_min + y_resol, bounds.y_max + y_resol, y_resol)
    return Coordinates(*np.meshgrid(X, Y))


def cells_lower_right_corner_coordinates(dst: gdal.Dataset) -> Coordinates:
    """
    ## Summary:
        `gdal.Dataset`の各セルの下端右隅の座標を取得します。
    Args:
        dst (gdal.Dataset):
    Returns:
        Coordinates(dataclass):
            1. X (np.ndarray): shape is (RasterYSize, RasterXSize)
            2. Y (np.ndarray): shape is (RasterYSize, RasterXSize)
    """
    bounds = get_bounds(dst)
    transform = dst.GetGeoTransform()
    x_resol = transform[1]
    y_resol = abs(transform[5])
    # Create a meshgrid of coordinates
    X = np.arange(bounds.x_min + x_resol, bounds.x_max + x_resol, x_resol)
    Y = np.arange(bounds.y_min + y_resol, bounds.y_max + y_resol, y_resol)
    return Coordinates(*np.meshgrid(X, Y))
