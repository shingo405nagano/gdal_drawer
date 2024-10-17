"""
ラスターデータの投影変換を行う
"""
from typing import Any
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
        """
        xs, ys = self._reproject_xy(
            [bounds.x_min, bounds.x_max],
            [bounds.y_min, bounds.y_max],
            in_crs, out_crs
        )
        return Bounds(xs[0], ys[0], xs[1], ys[1])

    def lyr_crs(self, lyr: Any) -> pyproj.CRS:
        """
        layerのCRSを取得する
        Args:
            lyr(Any): QgsVectorLayer or QgsRasterLayer
        Returns:
            (pyproj.CRS): CRS
        """
        return pyproj.CRS(lyr.crs().authid())
    
    def check_projection_is_metre(self, lyr: Any) -> bool:
        """
        RasterLayerの投影法がメートル法かどうかを判定する。
        Args:
            lyr(Any): QgsRasterLayer
        Returns:
            (bool): メートル法かどうか
        """
        crs = self.lyr_crs(lyr)
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
            xs(List[float]): x座標
            ys(List[float]): y座標
            in_crs(str): 入力のWKT-CRS
            out_crs(str): 出力のWKT-CRS
        Returns:
            List[List[float], List[float]]: (x座標, y座標)
        """
        tf = pyproj.Transformer.from_crs(
            in_crs, out_crs, always_xy=True
        )
        return tf.transform(xs, ys)
    
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