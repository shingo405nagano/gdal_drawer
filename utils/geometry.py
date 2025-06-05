import pyproj
import shapely

from gdal_drawer.utils.config import CRS, GEOMETRY, crs_checker, geometry_checker


def estimate_utm_crs(lon: float, lat: float, datum_name: str = "JGD2011") -> str:
    """
    ## Summary:
        経緯度（度単位）からUTM座標系を推定する。
    Args:
        lon (float): 経度
        lat (float): 緯度
        datum_name(str): 'WGS 84', 'JGD2011' ...  default='JGD2011'
    Returns:
        (str): WKT-CRS
    """
    try:
        # Check if the datum_name is valid
        pyproj.CRS.from_user_input(datum_name)
    except Exception as e:
        raise ValueError("Invalid datum_name. Use 'WGS 84', 'JGD2011', etc.") from e
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


@geometry_checker(0, "geometry")
@crs_checker(1, "in_crs")
@crs_checker(2, "out_crs")
def reprojection_geometry(
    geometry: GEOMETRY,  #
    in_crs: CRS,
    out_crs: CRS,
) -> shapely.geometry.base.BaseGeometry:
    """
    ## Summary:
        Geometryを指定されたCRSから別のCRSに再投影します。
    Args:
        geometry (GEOMETRY): GeometryオブジェクトあるいはWKT文字列のGeometry
        in_crs (CRS): 入力GeometryのCRS
        out_crs (CRS): 出力GeometryのCRS
    Returns:
        GEOMETRY: 投影変換されたGeometryオブジェクト
    """
    if in_crs == out_crs:
        # No reprojection needed
        return geometry
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    try:
        geom = shapely.transform(geometry, transformer.transform, interleaved=False)
    except Exception:  # noqa: E722
        geom = shapely.ops.transform(transformer.transform, geometry)
    return geom


@geometry_checker(0, "geometry")
@crs_checker(1, "in_crs")
def estimate_utm_crs_from_geometry(
    geometry: GEOMETRY,
    in_crs: CRS,
    datum_name: str = "JGD2011",
) -> pyproj.CRS:
    """
    ## Summary:
        GeometryからUTM座標系を推定します。
    Args:
        geometry (GEOMETRY):
            GeometryオブジェクトあるいはWKT文字列のGeometry
        in_crs (CRS):
            入力GeometryのCRS
        datum_name (str):
            'WGS 84', 'JGD2011' ...  default='JGD2011'
    Returns:
        pyproj.CRS: Estimated UTM CRS
    """
    projected_geometry = reprojection_geometry(geometry, in_crs, out_crs=4326)
    bounds = projected_geometry.bounds
    lon = (bounds[0] + bounds[2]) / 2
    lat = (bounds[1] + bounds[3]) / 2
    return estimate_utm_crs(lon, lat, datum_name=datum_name)


@crs_checker(index=0, kward="crs")
def crs_unit_name(crs: CRS) -> str:
    """
    ## Summary:
        "metre"や"degree"などの単位名を返します。
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
