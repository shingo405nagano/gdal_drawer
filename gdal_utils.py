"""
# Summary
ラスターデータの切り抜きを行うモジュール

-------------------------------------------------------------------------------
## RasterDataのPlot
>>> fp = '.\\dchm.tif'
>>> dst = gdal.Open(fp)
>>> gutils = GdalUtils()
>>> fig, ax = plt.subplots()
>>> gutils.plot_raster(fig, ax, dst, cmap='terrain', colorbar=True, nodata=True, shrink=0.8, nodata_label_anchor=(1.25, 1.1))
>>> plt.show()

>>> fp = '.\\rgb.tif'
>>> dst = gdal.Open(fp)
>>> gutils = GdalUtils()
>>> fig, ax = plt.subplots()
>>> gutils.plot_raster(fig, ax, dst)
>>> plt.show()

-------------------------------------------------------------------------------
## RasterDataの保存
>>> dst = gdal.Open('input.tif')
>>> gutils = GdalUtils()
>>> gutils.save_dst(dst, '.\\output.tif')

-------------------------------------------------------------------------------
## メモリ上にラスターデータを作成
>>> fp = '.\\dchm.tif'
>>> dst = gdal.Open(fp)
>>> ary = dst.ReadAsArray()
>>> ary = ary * 2
>>> gutils = GdalUtils()
>>> new_dst = gutils.write_ary_to_mem(dst, ary)

-------------------------------------------------------------------------------
## gdal.Datasetをコピー
>>> dst = gdal.Open('input.tif')
>>> gutils = GdalUtils()
>>> new_dst = gutils.copy_dataset(dst)

-------------------------------------------------------------------------------
## NoDataをnanに変換
>>> fp = '.\\dchm.tif'
>>> dst = gdal.Open(fp)
>>> gutils = GdalUtils()
>>> ary = gutils.nodata_to_nan(dst)

>>> fp = '.\\dchm.tif'
>>> dst = gdal.Open(fp)
>>> gutils = GdalUtils()
>>> new_dst = gutils.nodata_to_nan(dst, return_dst=True)

-------------------------------------------------------------------------------
## 各セルの中心座標を計算
>>> fp = '.\\dchm.tif'
>>> dst = gdal.Open(fp)
>>> gutils = GdalUtils()
>>> centers = gutils.cells_center_coordinates(dst)

-------------------------------------------------------------------------------
## RasterDataのセル値をshapely.PointにしてGeoDataFrameに入力
>>> fp = '.\\dchm.tif'
>>> dst = gdal.Open(fp)
>>> gutils = GdalUtils()
>>> gdf = gutils.raster_to_geodataframe(dst)

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union

import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import numpy as np
from osgeo import gdal
import shapely

from gdal_projection import gprojection



@dataclass
class CenterCoordinates:
    """
    各セルの中心座標を格納するデータクラス
    X(np.ndarray): X座標の2次元配列
    Y(np.ndarray): Y座標の2次元配列
    """
    X: np.ndarray
    Y: np.ndarray
    

class GdalUtils(object):
    def plot_raster(self, 
        fig: Figure,
        ax: Axes, 
        dst: gdal.Dataset, 
        **kwargs
    ) -> None:
        """
        ラスターデータをプロットする
        Args:
            fig (Figure): Figure
            ax (Axes): Axes
            dst (gdal.Dataset): ラスターデータ
            **kwargs: 
                cmap (str, optional): カラーマップ. Defaults to 'terrain'.
                colorbar (bool, optional): カラーバーを表示するかどうか. Defaults to True.
                - shrink(float): カラーバーのサイズを調整する. Defaults to 0.8
                - nodata(bool): NoDataを表示するかどうか. Defaults to False
                - nodata_label_anchor(tuple): NoDataのラベルの位置. Defaults to (1.25, 1.1)
        Returns:
            List[Any]: [Figure, Axes]
        Examples:
            >>> fp = '.\\dchm.tif'
            >>> dst = gdal.Open(fp)
            >>> gutils = GdalUtils()
            >>> fig, ax = plt.subplots()
            >>> gutils.plot_raster(fig, ax, dst, cmap='terrain', colorbar=True, nodata=True, shrink=0.8, nodata_label_anchor=(1.25, 1.1))
            >>> plt.show()
            --------------------------------------------------
            >>> fp = '.\\rgb.tif'
            >>> dst = gdal.Open(fp)
            >>> gutils = GdalUtils()
            >>> fig, ax = plt.subplots()
            >>> gutils.plot_raster(fig, ax, dst)
            >>> plt.show()
        """
        scope = gprojection.dataset_bounds(dst)
        extent = [scope.x_min, scope.x_max, scope.y_min, scope.y_max]

        if dst.RasterCount == 1:
            # Bandが1つの場合は、NoDataをnanに変換
            img = self.nodata_to_nan(dst)
        else:
            # Bandが複数の場合は、matplotlibで表示できる様に配列形状を変更
            img = np.dstack(dst.ReadAsArray())

        if kwargs.get('nodata') and dst.RasterCount == 1:
            # NoDataを強調して表示
            nodata_ary = np.where(np.isnan(img), 255, 0)
            ax.imshow(nodata_ary, cmap='bwr', extent=extent)
            patches = [Patch(color='red', label="NoData")]
            nodata_label_anchor = kwargs.get('nodata_label_anchor', (1.25, 1.1))
            plt.legend(handles=patches, bbox_to_anchor=nodata_label_anchor)

        cmap = kwargs.get('cmap', 'terrain')
        cb = ax.imshow(img, cmap=cmap, extent=extent)
        colorbar = kwargs.get('colorbar', True)
        if colorbar and dst.RasterCount == 1:
            # カラーバーを表示
            shrink = kwargs.get('shrink', 0.8)
            fig.colorbar(cb, shrink=shrink)

    def save_dst(self, dst: gdal.Dataset, path: Path, fmt: str='GTiff') -> None:
        """
        gdal.Datasetを保存する
        Args:
            dst (gdal.Dataset): 保存するRasterData
            path (Path): 保存先のパス
            fmt (str, optional): 保存形式. Defaults to 'GTiff'.
        Returns:
            None
        Examples:
            >>> dst = gdal.Open('input.tif')
            >>> gutils = GdalUtils()
            >>> gutils.save_dst(dst, '.\\output.tif')
        """
        driver = gdal.GetDriverByName(fmt)
        _dst = driver.CreateCopy(path, dst)
        _dst.FlushCache()
        _dst = None

    def write_ary_to_mem(self,
        org_dst: gdal.Dataset,
        ary: np.ndarray,
        data_type: int=gdal.GDT_Float32,
        nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        メモリ上にラスターデータを作成する。
        この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。※配列のshapeはオリジナルの`gdal.Dataset`と同じである必要がある。
        Args:
            org_dst(gdal.Dataset): ラスターデータ
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            nodata(Any): NoData. Defaults to np.nan
        Returns:
            gdal.Dataset
        Examples:
            >>> fp = '.\\dchm.tif'
            >>> dst = gdal.Open(fp)
            >>> ary = dst.ReadAsArray()
            >>> ary = ary * 2
            >>> gutils = GdalUtils()
            >>> new_dst = gutils.write_ary_to_mem(dst, ary)
        """
        if org_dst.RasterCount == 1:
            return self._single_band_to_mem(org_dst, ary, data_type, nodata)
        else:
            return self._multi_band_to_mem(org_dst, ary, data_type, nodata)
    
    def _single_band_to_mem(self, 
        org_dst: gdal.Dataset, 
        ary: np.ndarray, 
        data_type: int=gdal.GDT_Float32, 
        nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        メモリ上にラスターデータを作成する。これは単一バンドのみを対象とする。
        この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。※配列のshapeはオリジナルの`gdal.Dataset`と同じである必要がある。
        Args:
            org_dst(gdal.Dataset): ラスターデータ
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            nodata(Any): NoData. Defaults to np.nan
        Returns:
            gdal.Dataset
        """
        if ary.shape != (org_dst.RasterYSize, org_dst.RasterXSize):
            raise ValueError('The shape of the array must be the same as the raster size')
        driver = gdal.GetDriverByName('MEM')
        driver.Register()
        new_dst = driver.Create(
            '',
            xsize=org_dst.RasterXSize,
            ysize=org_dst.RasterYSize,
            bands=1,
            eType=data_type
        )
        new_dst.SetGeoTransform(org_dst.GetGeoTransform())
        new_dst.SetProjection(org_dst.GetProjection())
        band = new_dst.GetRasterBand(1)
        band.WriteArray(ary)
        band.SetNoDataValue(nodata)
        return new_dst

    def _multi_band_to_mem(self,
        org_dst: gdal.Dataset, 
        ary: np.ndarray, 
        data_type: int=gdal.GDT_Float32, 
        nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        メモリ上にラスターデータを作成する。これは複数バンドを対象とする。
        この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。※配列のshapeはオリジナルの`gdal.Dataset`と同じである必要がある。
        Args:
            org_dst(gdal.Dataset): ラスターデータ
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            nodata(Any): NoData. Defaults to np.nan
        Returns:
            gdal.Dataset
        """
        if ary.shape != (org_dst.RasterCount, org_dst.RasterYSize, org_dst.RasterXSize):
            raise ValueError('The shape of the array must be the same as the raster size')
        driver = gdal.GetDriverByName('MEM')
        driver.Register()
        new_dst = driver.Create(
            '',
            xsize=org_dst.RasterXSize,
            ysize=org_dst.RasterYSize,
            bands=org_dst.RasterCount,
            eType=data_type
        )
        new_dst.SetGeoTransform(org_dst.GetGeoTransform())
        new_dst.SetProjection(org_dst.GetProjection())
        for i in range(org_dst.RasterCount):
            band = new_dst.GetRasterBand(i+1)
            band.WriteArray(ary[i])
            band.SetNoDataValue(nodata)
        return new_dst
                           
    def copy_dataset(self, dst: gdal.Dataset) -> gdal.Dataset:
        """
        gdal.Datasetをコピーする
        Args:
            dst(gdal.Dataset): コピーするRasterData
        Returns:
            (gdal.Dataset): コピーされたRasterData
        Examples:
            >>> dst = gdal.Open('input.tif')
            >>> gutils = GdalUtils()
            >>> new_dst = gutils.copy_dataset(dst)
        """
        driver = gdal.GetDriverByName('MEM')
        new_dst = driver.CreateCopy('', dst)
        return new_dst

    def nodata_to_nan(self,
        dst: gdal.Dataset,
        return_dst: bool=False,
    ) -> Union[np.array, gdal.Dataset]:
        """
        ラスターデータのNoDataをnanに変換し、配列を返す
        Args:
            dst (gdal.Dataset): ラスターデータ
            return_dst (bool, optional): ラスターデータを返すかどうか. Defaults to False.
        Returns:
            np.array | gdal.Dataset
        Examples:
            >>> fp = '.\\dchm.tif'
            >>> dst = gdal.Open(fp)
            >>> gutils = GdalUtils()
            >>> ary = gutils.nodata_to_nan(dst)
        """
        ary = dst.ReadAsArray()
        nodata = dst.GetRasterBand(1).GetNoDataValue()
        ary = np.where(ary == nodata, np.nan, ary)
        if return_dst:
            return self.write_ary_to_mem(dst, ary)
        return ary
    
    def cells_center_coordinates(self, dst: gdal.Dataset) -> CenterCoordinates:
        """
        各セルの中心座標を計算
        Args:
            dst(gdal.Dataset): RasterDataを読み込んだデータセット
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次元配列
        Examples:
            >>> fp = '.\\dchm.tif'
            >>> dst = gdal.Open(fp)
            >>> gutils = GdalUtils()
            >>> centers = gutils.cells_center_coordinates(dst)
        """
        transform = dst.GetGeoTransform()
        scope = gprojection.dataset_bounds(dst)
        # X方向のセルの中心座標を計算し、1次元配列に
        x_resol = transform[1]
        # 0.5はセルの中心を示す
        half = 0.5
        _X = np.arange(scope.x_min, scope.x_max, x_resol) + x_resol * half
        # Y方向のセルの中心座標を計算し、1次元配列に
        y_resol = transform[-1]
        _Y = np.arange(scope.y_max, scope.y_min, y_resol) + y_resol * half
        # 各セルの中心座標を計算
        X, Y = np.meshgrid(_X, _Y)
        return CenterCoordinates(X, Y)

    def raster_to_geodataframe(self, dst: gdal.Dataset) -> gpd.GeoDataFrame:
        """
        RasterDataのセル値をshapely.PointにしてGeoDataFrameに入力
        Args:
            dst(gdal.Dataset): RasterDataを読み込んだデータセット
        Returns:
            gpd.GeoDataFrame
                'x': X座標
                'y': Y座標
                'band_1': バンド1の値
                ...
        Examples:
            >>> fp = '.\\dchm.tif'
            >>> dst = gdal.Open(fp)
            >>> gutils = GdalUtils()
            >>> gdf = gutils.raster_to_geodataframe(dst)
        """
        dst_ = self.copy_dataset(dst)
        in_crs = dst_.GetProjection()
        out_crs = None

        # セルの中心座標を取得
        centers = self.cells_center_coordinates(dst_)
        data = {
            'x': centers.X.flatten(),
            'y': centers.Y.flatten(),
        }
        # 各バンドの値を取得
        dst_ = self.nodata_to_nan(dst_, True)
        for i in range(dst_.RasterCount):
            ary1d = dst_.GetRasterBand(i+1).ReadAsArray().flatten()
            data[f'band_{i+1}'] = ary1d
        # GeoDataFrameを作成
        geoms = gpd.points_from_xy(data['x'], data['y'])
        # 小数部の影響か、たまに範囲外の座標が生成されるので、範囲内のものだけを取得
        data, geoms = self._adjustment_of_length(dst_, geoms, data)
        return gpd.GeoDataFrame(data, geometry=geoms, crs=in_crs)
    
    def _adjustment_of_length(self,
        dst: gdal.Dataset,
        geoms: gpd.GeoSeries,
        data: Dict[str, np.ndarray]
    ) -> Any:
        """
        小数部の影響か、たまに範囲外の座標が生成されるので、範囲内のものだけを取得
        """
        if 1 < len(np.unique([len(ary1d) for ary1d in data.values()])):
            x = data.get('x')
            y = data.get('y')
            bounds = shapely.box(*gprojection.dataset_bounds(dst))
            idx = np.where(geoms.intersects(bounds))[0]
            data['x'] = x[idx]
            data['y'] = y[idx]
            geoms = geoms[idx]
            return data, geoms
        else:
            return data, geoms


gutils = GdalUtils()

