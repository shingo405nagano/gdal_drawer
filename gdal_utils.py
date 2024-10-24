from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import numpy as np
from osgeo import gdal
import pyproj
import shapely
import shapely.ops


class Bounds(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class XY(NamedTuple):
    x: float | List[float]
    y: float | List[float]


class GdalUtils(object):
    def bounds(self, dst: gdal.Dataset) -> Bounds:
        """
        `gdal.Dataset`の範囲を取得する。
        Returns:
            Bounds(NamedTuple): (x_min, y_min, x_max, y_max)
        Examples:
            >>> bounds: Bounds = bounds(dst)
        """
        transform = dst.GetGeoTransform()
        x_min = transform[0]
        y_max = transform[3]
        rows = dst.RasterYSize
        cols = dst.RasterXSize
        x_max = x_min + cols * transform[1]
        y_min = y_max + rows * transform[-1]
        return Bounds(x_min, y_min, x_max, y_max)
    
    def estimate_utm_crs(self, lon: float, lat: float, **kwargs) -> str:
        """
        UTMのCRSを推定する。日本の場合は "datum_name='JGD2011'" を指定する。
        Args:
            lon(float): Longitude
            lat(float): Latitude
            kwargs:
                datum_name: 'WGS 84', 'JGD2011' ...  default='JGD2011'
        Returns:
            (str): WKT-CRS
        """
        datum_name = kwargs.get('datum_name', 'JGD2011')
        # UTMのCRSを推定する
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

    def reproject_xy(self,
        xs: float | List[float],
        ys: float | List[float],
        in_crs: str,
        out_crs: str
    ) -> XY:
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
            >>> gprojection = GdalProjection()
            >>> x, y = gprojection.reproject_xy(xs, ys, in_crs, out_crs)
            >>> x
            [-167354.7591972514, -75129.71899538414]
            >>> y
            [-553344.5836010452, -443620.80651071604]
        """
        tf = pyproj.Transformer.from_crs(
            in_crs, out_crs, always_xy=True
        )
        x, y = tf.transform(xs, ys)
        return XY(x, y)
    
    def reprojection_geometry(self, 
        wkt_geometry: str, 
        in_wkt_crs: str, 
        out_wkt_crs: str
    ) -> str:
        """
        ジオメトリの投影変換
        Args:
            wkt_geometry(str): ジオメトリ
            in_wkt_crs(str): 入力のWKT-CRS
            out_wkt_crs(str): 出力のWKT-CRS
        Returns:
            (str): 投影変換後のジオメトリ
        Examples:
            >>> wkt_geometry = 'POINT(139.00 35.00)'
            >>> in_wkt_crs = pyproj.CRS.from_epsg(4326).to_wkt()
            >>> out_wkt_crs = pyproj.CRS.from_epsg(6678).to_wkt()
            >>> gprojection = GdalProjection()
            >>> new_wkt_geometry = gprojection.reprojection_geometry(wkt_geometry, in_wkt_crs, out_wkt_crs)
        """
        geom = shapely.from_wkt(wkt_geometry)
        project = (
            pyproj
            .Transformer
            .from_crs(in_wkt_crs, out_wkt_crs, always_xy=True)
            .transform
        )
        transformed_geom = shapely.ops.transform(project, geom)
        return transformed_geom.wkt

    def degree_from_metre(self, 
        metre: float, 
        x: float, 
        y: float, 
        in_wkt_crs: str,
        digit: int=10,
        **kwargs
    ) -> float:
        """
        MetreからDegreeに変換する。
        Args:
            metre(float): メートル
            x(float): x座標
            y(float): y座標
            in_wkt_crs(str): 入力のWKT-CRS
            digit(int): 小数点以下の桁数
            kwargs:
                - x_direction(bool): x方向のベクトルを指定する. default=True
        Returns:
            (float): 度
        Examples:
            >>> gutils = GdalUtils()
            >>> metre = 1000
            >>> x = -167354.7591972514 
            >>> y = -553344.5836010452
            >>> in_wkt_crs = pyproj.CRS.from_epsg(6678).to_wkt()
            >>> degree = gutils.degree_from_metre(metre, x, y, in_wkt_crs)
            0.0109510791
        """
        x_direction = kwargs.get('x_direction', True)
        if x_direction:
            # x方向のベクトルを指定する
            line = shapely.LineString([[x, y], [x + metre, y]]).wkt
        else:
            # y方向のベクトルを指定する
            line = shapely.LineString([[x, y], [x, y + metre]]).wkt
        # 出力のWKT-CRSを指定して投影変換を行う
        out_wkt_crs = pyproj.CRS(4326).to_wkt()
        new_line = self.reprojection_geometry(line, in_wkt_crs, out_wkt_crs)
        # 線分の長さを取得する
        degree = shapely.from_wkt(new_line).length
        return round(degree, digit)
    
    def metre_from_degree(self,
        degree: float,
        lon: float,
        lat: float,
        digit: int=4,
        **kwargs
    ) -> float:
        """
        DegreeからMetreに変換する
        Args:
            degree(float): 度
            lon(float): Longitude
            lat(float): Latitude
            digit(int): 小数点以下の桁数
            kwargs:
                - x_direction(bool): x方向のベクトルを指定する. default=True
        Returns:
            (float): メートル
        Examples:
            >>> gutils = GdalUtils()
            >>> degree = 0.0001
            >>> lon = 139.00
            >>> lat = 35.00
            >>> metre = gutils.metre_from_degree(degree, lon, lat)
            9.1289
        """
        x_direction = kwargs.get('x_direction', True)
        if x_direction:
            # x方向のベクトルを指定する
            line = shapely.LineString([[lon, lat], [lon + degree, lat]]).wkt
        else:
            # y方向のベクトルを指定する
            line = shapely.LineString([[lon, lat], [lon, lat + degree]]).wkt
        # UTMのCRSを推定し投影変換を行う
        in_wkt_crs = pyproj.CRS.from_epsg(4326).to_wkt()
        out_wkt_crs = self.estimate_utm_crs(lon, lat)
        new_line = self.reprojection_geometry(line, in_wkt_crs, out_wkt_crs)
        metre = shapely.from_wkt(new_line).length
        return round(metre, digit)
"""
band_numbers = 1. None, 2. int, 3. List[int]
1. None: すべてのバンドを読み込む

2. int: 指定したバンドを読み込む
3. List[int]: 指定したバンドを読み込む
"""