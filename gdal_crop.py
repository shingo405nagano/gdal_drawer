"""
# Summary
ラスターデータの切り抜きを行うモジュール

-------------------------------------------------------------------------------
## PolygonのGeometryを使ってRasterを切り抜く
>>> dst = gdal.Open('./raster.tif')
>>> wkt_poly = 'POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'
>>> croped_dst = gcrop.with_wkt_poly(dst, wkt_poly)

-------------------------------------------------------------------------------
## PolygonのBoundsを使ってRasterを切り抜く
>>> dst = gdal.Open('./raster.tif')
>>> wkt_poly = 'POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'
>>> croped_dst = gcrop.with_wkt_poly_bounds(dst, wkt_poly)

-------------------------------------------------------------------------------
## Polygonを使用した最小の四角形でRasterを切り抜く
>>> dst = gdal.Open('./raster.tif')
>>> wkt_poly = 'POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'
>>> croped_dst = gcrop.with_wkt_poly_fit_bounds(dst, wkt_poly)

-------------------------------------------------------------------------------
"""

from typing import Any
from typing import Callable

import numpy as np
from osgeo import gdal
import shapely
import shapely.geometry.base


def geometry_check(func: Callable) -> str:
    # ジオメトリがWKT形式であるかチェックする。shapely.geometryだった場合はWKT形式に変換する
    def wrapper(*args, **kwargs):
        args_ = False
        if 'wkt_poly' not in kwargs.keys():
            poly = args[2]
            args_ = True
        else:
            poly = kwargs.get('wkt_poly')
        if isinstance(poly, str):
            try:
                shapely.from_wkt(poly)
            except shapely.errors.WKTReadingError:
                raise ValueError('The geometry must be in WKT format')
            else:
                return func(*args, **kwargs)
        elif isinstance(poly, shapely.geometry.base.BaseGeometry):
            if args_:
                lst = list(args)
                lst[2] = poly.wkt
                args = tuple(lst)
            else:
                kwargs['wkt_poly'] = poly.wkt
            return func(*args, **kwargs)
        else:
            raise ValueError('The geometry must be in WKT format')
    return wrapper



class GdalCrop(object):    
    def option_template_with_wkt_poly_spec(self,
        dst: gdal.Dataset,
        wkt_poly: str,
        fmt: str='MEM',
        nodata: Any=np.nan,
    ) -> gdal.Dataset:
        """
        gdal.WarpでRasterを切り抜く為のオプションテンプレートを作成する
        Args:
            dst (gdal.Dataset): 切り抜くRaster
            wkt_poly (str): 切り抜く範囲のWKT形式のポリゴン
            fmt (str, optional): 出力形式. Defaults to 'MEM'.
            nodata (Any, optional): 出力RasterのNoData値. Defaults to np.nan.
        Returns:
            gdal.WarpOptions: gdal.WarpでRasterを切り抜く為のオプション
        """
        return gdal.WarpOptions(
            format=fmt,
            cutlineWKT=wkt_poly,
            cropToCutline=True,
            dstNodata=nodata,
            srcNodata=dst.GetRasterBand(1).GetNoDataValue()
        )

    @geometry_check
    def with_wkt_poly(self,
        dst: gdal.Dataset, 
        wkt_poly: str, 
        nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        Wkt-PolygonでRasterを切り抜く
        Args:
            dst (gdal.Dataset): 切り抜くRasterData
            wkt_poly (str): 切り抜く範囲のWKT形式のポリゴン
            nodata (Any, optional): 出力RasterのNoData値. Defaults to np.nan.
        Returns:
            gdal.Dataset: 切り抜かれたRasterData
        Examples:
            >>> dst = gdal.Open('./raster.tif')
            >>> wkt_poly = 'POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'
            >>> gcrop = GdalCrop()
            >>> croped_dst = gcrop.with_wkt_poly(dst, wkt_poly)
        """
        options = self.option_template_with_wkt_poly_spec(
            dst, wkt_poly, nodata=nodata
        )
        return gdal.Warp('', dst, options=options)


    @geometry_check
    def with_wkt_poly_bounds(self,
        dst: gdal.Dataset, 
        wkt_poly: str, 
        nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        PolygonのBoundsでRasterを切り抜く
        Args:
            dst (gdal.Dataset): 切り抜くRasterData
            wkt_poly (str): 切り抜く範囲のWKT形式のポリゴン
            nodata (Any, optional): 出力RasterのNoData値. Defaults to np.nan.
        Returns:
            gdal.Dataset: 切り抜かれたRasterData
        Examples:
            >>> dst = gdal.Open('./raster.tif')
            >>> wkt_poly = 'POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'
            >>> gcrop = GdalCrop()
            >>> croped_dst = gcrop.with_wkt_poly_bounds(dst, wkt_poly)
        """
        wkt_poly = shapely.from_wkt(wkt_poly).envelope.wkt
        return self.with_wkt_poly(dst, wkt_poly, nodata)

    @geometry_check
    def with_wkt_poly_fit_bounds(self,
        dst: gdal.Dataset, 
        wkt_poly: str, 
        nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        Polygonを使用した最小の四角形でRasterを切り抜く
        Args:
            dst (gdal.Dataset): 切り抜くRasterData
            wkt_poly (str): 切り抜く範囲のWKT形式のポリゴン
            nodata (Any, optional): 出力RasterのNoData値. Defaults to np.nan.
        Returns:
            gdal.Dataset: 切り抜かれたRasterData
        Examples:
            >>> dst = gdal.Open('./raster.tif')
            >>> wkt_poly = 'POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'
            >>> gcrop = GdalCrop()
            >>> croped_dst = gcrop.with_wkt_poly_fit_bounds(dst, wkt_poly)
        """
        wkt_poly = shapely.from_wkt(wkt_poly).minimum_rotated_rectangle.wkt
        return self.with_wkt_poly(dst, wkt_poly, nodata)



gcrop = GdalCrop()