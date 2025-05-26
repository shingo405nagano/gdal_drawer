import pyproj
import shapely

from gdal_drawer.utils.config import (
    CRS, 
    crs_checker, 
    GEOMETRY, 
    geometry_checker
)


def estimate_utm_crs(lon: float, lat: float, datum_name: str = 'JGD2011') -> str:
    """
    ## Summary:
        Estimate the UTM CRS. In Japan, specify "datum_name='JGD2011'".
    Args:
        lon (float): Longitude
        lat (float): Latitude
        datum_name(str): 'WGS 84', 'JGD2011' ...  default='JGD2011'
    Returns:
        (str): WKT-CRS
    """
    # Estimate the UTM CRS
    aoi = pyproj.aoi.AreaOfInterest(
        west_lon_degree=lon,
        south_lat_degree=lat,
        east_lon_degree=lon,
        north_lat_degree=lat,
    )
    utm_crs_lst = pyproj.database.query_utm_crs_info(
        datum_name=datum_name, area_of_interest=aoi
    )
    return pyproj.CRS.from_epsg(utm_crs_lst[0].code)


@geometry_checker(0, 'geometry')
@crs_checker(1, 'in_crs')
@crs_checker(2, 'out_crs')
def reprojection_geometry(
    geometry: GEOMETRY, 
    in_crs: CRS, 
    out_crs: CRS
) -> shapely.geometry.base.BaseGeometry:
    """
    ## Summary:
        Reproject the geometry to the specified CRS.
    Args:
        geometry (GEOMETRY): Geometry
        in_crs (CRS): Input CRS
        out_crs (CRS): Output CRS
    Returns:
        GEOMETRY: Reprojected geometry
    """
    if in_crs == out_crs:
        # No reprojection needed
        return geometry
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    try:
        geom = shapely.transform(geometry, transformer.transform, interleaved=False)
    except:
        geom = shapely.ops.transform(
            transformer.transform, geometry
        )
    return geom


@geometry_checker(0, 'geometry')
@crs_checker(1, 'in_crs')
def estimate_utm_crs_from_geometry(
    geometry: GEOMETRY,
    in_crs: CRS,
    datum_name: str = 'JGD2011',
) -> pyproj.CRS:
    """
    ## Summary:
        Estimate the UTM CRS from the geometry.
    Args:
        geometry (GEOMETRY): 
            Geometry is a shapely geometry object or WKT string object.
        in_crs (CRS): 
            Input CRS. CRS is a pyproj.CRS object or EPSG code.
        datum_name (str): 
            'WGS 84', 'JGD2011' ...  default='JGD2011'
    Returns:
        pyproj.CRS: Estimated UTM CRS
    """
    projected_geometry = reprojection_geometry(
        geometry,
        in_crs,
        out_crs=4326
    )
    bounds = projected_geometry.bounds
    lon = (bounds[0] + bounds[2]) / 2
    lat = (bounds[1] + bounds[3]) / 2
    return estimate_utm_crs(lon, lat, datum_name=datum_name)


@crs_checker(index=0, kward='crs')
def crs_unit_name(crs: CRS) -> str:
    """
    ## Summary:
        Get the unit name of the CRS.
    Args:
        crs (CRS): CRS
    Returns:
        str: Unit name
    Examples:
        >>> crs = pyproj.CRS.from_epsg(4326)
        >>> unit_name = crs_unit_name(crs)
        >>> print(unit_name)
        degree
        >>> crs = pyproj.CRS.from_epsg(6691)
        >>> unit_name = crs_unit_name(crs)
        >>> print(unit_name)
        metre
    """
    return crs.axis_info[0].unit_name