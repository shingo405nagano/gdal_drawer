"""
# Summary
ラスターデータの投影変換を行うモジュール

-------------------------------------------------------------------------------
## 経緯度からUTMのCRSを推定する場合
>>> wkt_crs = estimate_utm_crs(lon, lat, datum_name='WGS 84')

-------------------------------------------------------------------------------
## ラスターデータの範囲を取得する場合
>>> projection = RasterProjection()
>>> bounds = projection.dataset_bounds(dst)

-------------------------------------------------------------------------------
## 範囲の投影変換を行う場合
>>> projection = RasterProjection()
>>> reprojected_bounds = projection.reprojected_bounds(bounds, in_crs, out_crs)

-------------------------------------------------------------------------------
## 投影法がメートル法かどうかを判定する場合
>>> projection = RasterProjection()
>>> is_metre = projection.check_crs_is_metre(wkt_crs)

-------------------------------------------------------------------------------
## ラスターデータの投影変換を行う場合
>>> dst = gdal.Open('./raster.tif')
>>> out_crs = pyproj.CRS.from_epsg(6678).to_wkt()
>>> new_dst = reprojection_raster(dst, out_crs)

-------------------------------------------------------------------------------
## UTMを推定してラスターデータの投影変換を行う場合
>>> dst = gdal.Open('./raster.tif') # EPSG:4326
>>> new_dst = estimating_utm_crs_to_projection_raster(dst, 'JGD2011')

-------------------------------------------------------------------------------
"""
from typing import List
from typing import NamedTuple

from osgeo import gdal
import pyproj


def estimate_utm_crs(lon: float, lat: float, datum_name: str='WGS 84') -> str:
    """
    経緯度からUTMのCRSを推定する。
    Args:
        lon(float): Longitude
        lat(float): Latitude
        datum_name: 'WGS 84', 'JGD2011' ...
    Returns:
        (str): WKT-CRS
    Examples:
        >>> lon = 141.00
        >>> lat = 40.00
        >>> wkt_crs = estimate_utm_crs(lon, lat)
        -----------------------------------------
        >>> wkt_crs = estimate_utm_crs(lon, lat, 'JGD2011')
    """
    aoi = pyproj.aoi.AreaOfInterest(
        west_lon_degree=lon,
        south_lat_degree=lat,
        east_lon_degree=lon,
        north_lat_degree=lat,
    )
    utm_crs_lst = pyproj.database.query_utm_crs_info(
        datum_name=datum_name,
        area_of_interest=aoi
    )
    return pyproj.CRS.from_epsg(utm_crs_lst[0].code).to_wkt()


class Bounds(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class RasterProjection(object):
    def dataset_bounds(self, dst: gdal.Dataset) -> Bounds:
        """
        ラスターデータの範囲を取得する
        Args:
            dst(gdal.Dataset): ラスターデータ
        Returns:
            Bounds(NamedTuple): (x_min, y_min, x_max, y_max)
        Examples:
            >>> dst = gdal.Open('./raster.tif')
            >>> projection = RasterProjection()
            >>> bounds = projection.dataset_bounds(dst)
            -----------------------------------------
            >>> bounds_poly = shapely.box(*bounds)
        """
        transform = dst.GetGeoTransform()
        x_min = transform[0]
        y_max = transform[3]
        rows = dst.RasterYSize
        cols = dst.RasterXSize
        x_resol = transform[1]
        y_resol = transform[-1]
        x_max = x_min + cols * x_resol
        y_min = y_max + rows * y_resol
        return Bounds(x_min, y_min, x_max, y_max)
    
    def reprojected_bounds(self, bounds: Bounds, in_crs: str, out_crs: str) -> Bounds:
        """
        範囲の投影変換
        Args:
            bounds(Bounds): 範囲
            in_crs(str): 入力のWKT-CRS
            out_crs(str): 出力WKT-CRS
        Returns:
            Bounds(NamedTuple): 投影変換後の範囲
        Examples:
            >>> bounds = Bounds(139.00, 35.00, 140.00, 36.00)
            >>> in_crs = pyproj.CRS.from_epsg(4326).to_wkt()
            >>> out_crs = pyproj.CRS.from_epsg(6678).to_wkt()
            >>> projection = RasterProjection()
            >>> reprojected_bounds = projection.reprojected_bounds(bounds, in_crs, out_crs)
        """
        xs, ys = self._reproject_xy(
            [bounds.x_min, bounds.x_max],
            [bounds.y_min, bounds.y_max],
            in_crs, out_crs
        )
        return Bounds(xs[0], ys[0], xs[1], ys[1])
    
    def check_crs_is_metre(self, wkt_crs: str) -> bool:
        """
        投影法がメートル法かどうかを判定する。
        Args:
            wkt_crs(str): WKT-CRS
        Returns:
            (bool): メートル法かどうか
        Examples:
            >>> wkt_crs = pyproj.CRS.from_epsg(4326).to_wkt()
            >>> projection = RasterProjection()
            >>> is_metre = projection.check_projection_is_metre(wkt_crs)
            False
            >>> wkt_crs = pyproj.CRS.from_epsg(6678).to_wkt()
            >>> is_metre = projection.check_projection_is_metre(wkt_crs)
            True
        """
        crs = pyproj.CRS(wkt_crs)
        return crs.axis_info[0].unit_name == 'metre'
    
    def _reproject_xy(self,
        xs: float | List[float],
        ys: float | List[float],
        in_crs: str,
        out_crs: str
    ) -> List[List[float]]:
        """
        XYの投影変換
        Args:
            xs(float | List[float]): x座標
            ys(float | List[float]): y座標
            in_crs(str): 入力のWKT-CRS
            out_crs(str): 出力のWKT-CRS
        Returns:
            List[List[float], List[float]]: (x座標, y座標)
        Examples:
            >>> xs = [139.00, 140.00]
            >>> ys = [35.00, 36.00]
            >>> in_crs = pyproj.CRS.from_epsg(4326).to_wkt()
            >>> out_crs = pyproj.CRS.from_epsg(6678).to_wkt()
            >>> projection = RasterProjection()
            >>> x, y = projection._reproject_xy(xs, ys, in_crs, out_crs)
            >>> x
            [-167354.7591972514, -75129.71899538414]
            >>> y
            [-553344.5836010452, -443620.80651071604]
        """
        tf = pyproj.Transformer.from_crs(
            in_crs, out_crs, always_xy=True
        )
        return tf.transform(xs, ys)

    def reprojection_geometry(self, wkt_geometry: str, in_wkt_crs: str, out_wkt_crs: str) -> str:
        geom = shapely.from_wkt(wkt_geometry)
        project = (
            pyproj
            .Transformer
            .from_crs(in_wkt_crs, out_wkt_crs, always_xy=True)
            .transform
        )
        transformed_geom = shapely.ops.transform(project, geom )
        return transformed_geom.wkt
    
    def metre_to_degree(metre: float, in_crs: str, x: float, y: float) -> float:
        line = shapely.LineString([[x, y], [x + metre, y]]).wkt
        out_crs = pyproj.CRS(4326).to_wkt()
        new_line = self.transform_geometry(line, wkt_crs, out_crs)
        return shapely.from_wkt(new_line).length
    
    def reprojection_raster(self, dst: gdal.Dataset, out_crs: str) -> gdal.Dataset:
        """
        ラスターデータの投影変換
        Args:
            dst(gdal.Dataset): ラスターデータ
            out_crs(str): 出力のWKT-CRS
        Returns:
            (gdal.Dataset): 投影変換後のラスターデータ
        """
        # 投影変換の為のオプションを設定
        bounds = self.dataset_bounds(dst)
        ops = gdal.WarpOptions(
            format='MEM',
            srcSRS=dst.GetProjection(),
            dstSRS=out_crs,
            outputBounds=self.reprojected_bounds(bounds, dst.GetProjection(), out_crs),
            width=dst.RasterXSize,
            height=dst.RasterYSize,
            resampleAlg=gdal.GRA_CubicSpline
        )
        return gdal.Warp('', dst, options=ops)
    


def reprojection_raster(dst: gdal.Dataset, out_crs: str) -> gdal.Dataset:
    """
    ラスターデータの投影変換。
    Args:
        dst(gdal.Dataset): ラスターデータ
        out_crs(str): 出力のWKT-CRS
    Returns:
        (gdal.Dataset): 投影変換後のラスターデータ
    Examples:
        >>> dst = gdal.Open('./raster.tif') # EPSG:4326
        >>> epsg = 6678
        >>> out_crs = pyproj.CRS.from_epsg(epsg).to_wkt()
        >>> projected_dst = reprojection_raster(dst, out_crs)
    """
    projection = RasterProjection()
    return projection.reprojection_raster(dst, out_crs)


def estimating_utm_crs_to_projection_raster(
    dst: gdal.Dataset, 
    datum_name: str='WGS 84'
) -> gdal.Dataset:
    """
    ラスターデータの投影変換。この関数は前提条件として、座標値が経緯度であることを想定している。
    Args:
        dst(gdal.Dataset): ラスターデータ
        datum_name: 'WGS 84', 'JGD2011' など
    Returns:
        (gdal.Dataset): 投影変換後のラスターデータ
    Examples:
        >>> dst = gdal.Open('./raster.tif') # EPSG:4326
        >>> projected_dst = estimating_utm_crs_to_projection_raster(dst, 'JGD2011')
    """
    projection = RasterProjection()
    bounds = projection.dataset_bounds(dst)
    x_mean = (bounds.x_min + bounds.x_max) / 2
    y_mean = (bounds.y_min + bounds.y_max) / 2
    out_crs = estimate_utm_crs(x_mean, y_mean, datum_name)
    return reprojection_raster(dst, out_crs)
