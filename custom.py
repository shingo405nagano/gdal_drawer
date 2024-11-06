from collections.abc import Iterable
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pandas as pd
import pyproj
import scipy.ndimage
import shapely
gdal.UseExceptions()

from gdal_utils import GdalUtils
from utils.exceptions import custom_gdal_exception
from kernels import kernels
gdal_utils = GdalUtils()


################################################################################
# --------------------------------- DataClass ----------------------------------
################################################################################
class Bounds(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class XY(NamedTuple):
    x: float
    y: float


class CellSize(NamedTuple):
    x_size: float
    y_size: float

@dataclass
class Coordinates:
    """
    各セルの座標を格納するデータクラス
    X(np.ndarray): X座標の2次元配列
    Y(np.ndarray): Y座標の2次元配列
    """
    X: np.ndarray
    Y: np.ndarray


################################################################################
# --------------------------------- Functions ----------------------------------
################################################################################

##############################################################################
# ------------------------------ Main class ----------------------------------
################################################################################
class CustomGdalDataset(object):
    def __init__(self, dataset):
        if not isinstance(dataset, gdal.Dataset):
            custom_gdal_exception.not_gdal_dataset_err()
        if dataset.GetProjection() == '':
            custom_gdal_exception.not_have_crs_err()
        self.dataset = self._copy_dataset(dataset)

    def __getattr__(self, module_name):
        return getattr(self.dataset, module_name)
     
    @staticmethod
    def __check_crs(crs_index: int, crs_arg_name: str):
        """
        ## Summary
            CRSが正しく指定されているかチェックするデコレータ
        Args:
            crs_index(int): CRSが指定されている位置引数のインデックス
            crs_arg_name(str): CRSが指定されている引数名
        """
        def decorator(func: Callable):
            def wrapper(self, *args, **kwargs):
                def convert_crs(crs: Any) -> str:
                    """CRSをWkt形式に変換する"""
                    if isinstance(crs, str) or isinstance(crs, int):
                        return pyproj.CRS(crs).to_wkt()
                    elif isinstance(crs, pyproj.CRS):
                        return crs.to_wkt()
                    elif crs is None:
                        return None
                    else:
                        custom_gdal_exception.unknown_crs_err()
                # CRSが指定されているかチェック
                crs = None
                in_args = True
                if crs_index < len(args):
                    crs = args[crs_index]
                elif crs_arg_name in kwargs:
                    crs = kwargs[crs_arg_name]
                    in_args = False
                else:
                    in_args = False
                # CRSをWkt形式に変換
                crs = convert_crs(crs)
                if in_args:
                    args = list(args)
                    args[crs_index] = crs
                else:
                    kwargs[crs_arg_name] = crs
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def __check_datum(func):
        """
        ## Summary
            データムが正しく指定されているかチェックするデコレータ
        """
        def wrapper(self, *args, **kwargs):
            datum_name = kwargs.get('datum_name', 'JGD2011')
            try:
                _ = pyproj.CRS(datum_name).to_authority()
            except pyproj.exceptions.CRSError:
                custom_gdal_exception.unknown_datum_err()
            return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def __is_iterable_of_ints(arg_index: int, arg_name: str):
        """
        ## Summary
            引数がint型またはint型のイテラブルであるかチェックするデコレータ
        Args:
            arg_index(int): イテラブルが指定されている位置引数のインデックス
            arg_name(str): イテラブルが指定されている引数名
        """
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if arg_index < len(args):
                    value = args[arg_index]
                elif arg_name in kwargs:
                    value = kwargs[arg_name]
                else:
                    value = None
                
                if isinstance(value, Iterable):
                    if all(isinstance(item, int) for item in value):
                        return func(self, *args, **kwargs)
                elif isinstance(value, int):
                    return func(self, *args, **kwargs)
                elif value is None:
                    return func(self, *args, **kwargs)
                else:
                    custom_gdal_exception.get_band_number_err()
            return wrapper
        return decorator
    
    @staticmethod
    def __wkt_geometry_check(arg_index: int, arg_name: str) -> str:
        """
        ## Summary
            ジオメトリがWKT形式であるかチェックする。shapely.geometryだった場合はWKT形式に変換する
        Args:
            geom_index(int): ジオメトリが指定されている位置引数のインデックス
            geom_name(str): ジオメトリが指定されている引数名
        """
        def decorator(func: Callable):
            def wrapper(self, *args, **kwargs):
                # geometryの取得
                geom = None
                in_args = True
                if arg_index < len(args):
                    geom = args[arg_index]
                elif arg_name in kwargs:
                    geom = kwargs[arg_name]
                    in_args = False
                else:
                    raise ValueError('The geometry argument was not found.')
                # WKT形式に変換
                if isinstance(geom, str):
                    try:
                        geom = shapely.from_wkt(geom)
                    except:
                        custom_gdal_exception.load_wkt_geometry_err()
                elif isinstance(geom, shapely.geometry.base.BaseGeometry):
                    using = ['POINT', 'MULTIPOINT', 
                             'LINESTRING', 'MULTILINESTRING', 
                             'LINERGING', 'MULTILINERGING',
                             'POLYGON', 'MULTIPOLYGON']
                    if geom.geom_type.upper() not in using:
                        raise ValueError('The geometry type is not supported.')
                # geometryを引数にセット
                if in_args:
                    args = list(args)
                    args[arg_index] = geom.wkt
                else:
                    kwargs[arg_name] = geom.wkt
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def __band_check(count: int):
        """
        ## Summary
            datasetのBand数が指定された数と一致するかチェックする.
        """
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if self.RasterCount == count:
                    return func(self, *args, **kwargs)
                else:
                    custom_gdal_exception.band_count_err(count)
            return wrapper
        return decorator
        
    @property
    def x_resolution(self):
        """
        ## Summary
            X方向の解像度を取得する
        Returns:
            (float): X方向の解像度
        """
        return self.GetGeoTransform()[1]
    
    @property
    def y_resolution(self):
        """
        ## Summary
            Y方向の解像度を取得する
        Returns:
            (float): Y方向の解像度
        """
        return self.GetGeoTransform()[-1]

    @__is_iterable_of_ints(0, 'band_numbers')
    def array(self, band_numbers: int | Iterable[int]=None) -> np.ndarray:
        """
        ## Summary
            gdal.Datasetから配列を取得する。この関数はFloat型のNoDataをnp.nanに変換し、バンドのNoDataも書き換える。
        Args:
            band_numbers(int | Iterable[int]): バンド番号。指定しない場合は全バンドを取得する。
        Returns:
            (np.ndarray): Float型はNoDataをnp.nanに変換した配列を返す。この際、バンドのNoDataも書き換える。
        Examples:
            >>> ary = dst.array()
            >>> ary = dst.array(1)
            >>> ary = dst.array([1, 2, 3])
            >>> ary = dst.array([3, 2, 1])
        """
        if band_numbers is None:
            # バンド番号が指定されている場合
            return self._get_all_ary()
        elif isinstance(band_numbers, int):
            return self._get_selected_ary(band_numbers)
        elif isinstance(band_numbers, Iterable):
            return self._get_selected_arys(band_numbers)
        else:
            custom_gdal_exception.get_band_err()
        
    def _get_all_ary(self) -> np.ndarray:
        """
        ## Summary
            全バンドの配列を取得し、Float型ならばNoDataをnp.nanに変換する。
        Returns:
            (np.ndarray): Float型はNoDataをnp.nanに変換した配列を返す。この際、バンドのNoDataも書き換える。
        """
        arys = []
        for band in self._band_generator:
            data_type = gdal.GetDataTypeName(band.DataType)
            if data_type in ['Float32', 'Float64']:
                arys.append(self._get_float_ary(band))
            else:
                arys.append(band.ReadAsArray())
        ary = np.array(arys)
        if ary.shape[0] == 1:
            return ary[0]
        return ary
    
    def _get_selected_ary(self, band_nums: int) -> np.ndarray:
        """
        ## Summary
            指定したバンドの配列を取得する
        Args:
            band_nums(int): バンド番号
        Returns:
            (np.ndarray): Float型はNoDataをnp.nanに変換した配列を返す。この際、バンドのNoDataも書き換える。
        """
        band = self.GetRasterBand(band_nums)
        data_type = gdal.GetDataTypeName(band.DataType)
        if data_type in ['Float32', 'Float64']:
            return self._get_float_ary(band)
        return band.ReadAsArray()

    def _get_selected_arys(self, band_nums: List[int]) -> np.ndarray:
        """
        ## Summary
            Listで指定したバンドの配列を取得する
        Args:
            band_nums(List[int]): バンド番号のリスト
        Returns:
            (np.ndarray): Float型はNoDataをnp.nanに変換した配列を返す。この際、バンドのNoDataも書き換える。
        """
        arys = []
        for band_num in band_nums:
            band = self.GetRasterBand(band_num)
            data_type = gdal.GetDataTypeName(band.DataType)
            if data_type in ['Float32', 'Float64']:
                arys.append(self._get_float_ary(band))
            else:
                arys.append(band.ReadAsArray())
        ary = np.array(arys)
        if ary.shape[0] == 1:
            return ary[0]
        return ary

    @property
    def _band_generator(self) -> Generator:
        """
        ## Summary
            gdal.Bandのジェネレータを取得する
        Returns:
            Generator: gdal.Band
        Examples:
            >>> for band in dst._band_generator:
            >>>     ary = band.ReadAsArray()
            >>>     nodata = band.GetNoDataValue()
        """
        for i in range(1, self.dataset.RasterCount + 1):
            yield self.dataset.GetRasterBand(i)
    
    def _get_float_ary(self, band: gdal.Band) -> np.array:
        """
        ## Summary
            Nodataを全てnp.nanに変換した配列を取得し、BandのNodataを書き換える。
        Args:
            band(gdal.Band):
        Returns:
            (np.ndarray): NoDataをnp.nanに変換した配列
        """
        ary = band.ReadAsArray()
        ary = np.where(ary == band.GetNoDataValue(), np.nan, ary)
        ary = np.where(np.isnan(ary), np.nan, ary)
        ary = np.where(np.isinf(ary), np.nan, ary)
        band.SetNoDataValue(np.nan)
        return ary
        
    #######################################################################
    # -------------------- Methods for create dataset. --------------------
    def copy_dataset(self) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            `gdal.Dataset`のコピーを作成する。
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        Examples:
            >>> new_dst: gdal.Dataset = dst.copy_dataset()
        """
        driver = gdal.GetDriverByName('MEM')
        new_dst = driver.CreateCopy('', self.dataset)
        return CustomGdalDataset(new_dst)
    
    def _copy_dataset(self, 
        dst: gdal.Dataset
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            `gdal.Dataset`のコピーを作成する。
        Args:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        Returns:
            (gdal.Dataset):
        Examples:
            >>> new_dst: gdal.Dataset = dst._copy_dataset()
        """
        driver = gdal.GetDriverByName('MEM')
        new_dst = driver.CreateCopy('', dst)
        return new_dst

    def save_dst(self, file_path: Path, fmt: str='GTiff') -> None:
        """
        ## Summary
            gdal.Datasetを保存する
        Args:
            path (Path): 保存先のパス
            fmt (str, optional): 保存形式. Defaults to 'GTiff'.
        Returns:
            None
        Examples:
            >>> new_file_path = r'.\\new_raster.tif''
            >>> dst.save_dst(new_file_path)
        """
        driver = gdal.GetDriverByName(fmt)
        _dst = driver.CreateCopy(file_path, self.dataset)
        _dst.FlushCache()
        _dst = None

    def write_ary_to_mem(self,
        ary: np.ndarray,
        data_type: int=gdal.GDT_Float32,
        out_nodata: Any=np.nan,
        raster_count: int=1,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            メモリ上に新しい配列を書き込んだ`gdal.Dataset`を作成する。
            この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ新しい`gdal.Dataset`を作成する。
        Args:
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. 
                [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            nodata(Any): NoData. Defaults to np.nan
            raster_count(int): バンド数
            **kwargs: データセットのパラメータ
                - transform(list): GeoTransform
                - projection(str): Projection
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        Examples:
            >>> file_path = r'.\\raster.tif''
            >>> dst = CustomGdalDataset(file_path)
            >>> ary = np.random.rand(dst.RasterYSize, dst.RasterXSize)
            >>> new_dst: gdal.Dataset = dst.write_ary_to_mem(ary)
        """
        kwargs['count'] = raster_count
        if raster_count == 1:
            kwargs['xsize'] = ary.shape[1]
            kwargs['ysize'] = ary.shape[0]
            return self._single_band_to_mem(ary, data_type, out_nodata, **kwargs)
        else:
            kwargs['xsize'] = ary.shape[2]
            kwargs['ysize'] = ary.shape[1]
            return self._multi_band_to_mem(ary, data_type, out_nodata, **kwargs)

    def _single_band_to_mem(self, 
        ary: np.ndarray, 
        data_type: int=gdal.GDT_Float32, 
        out_nodata: Any=np.nan,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ新しい`gdal.Dataset`を作成する。
        Args:
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. 
                [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            out_nodata(Any): NoData. Defaults to np.nan
            **kwargs: データセットのパラメータ
                - transform(list): GeoTransform
                - projection(str): Projection
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        """
        # メモリ上に新しいラスターデータを作成する
        new_dst = self.__create_dataset(data_type, **kwargs)
        band = new_dst.GetRasterBand(1)
        ary = self._nodata_to(ary, band.GetNoDataValue(), out_nodata)
        band.WriteArray(ary)
        band.SetNoDataValue(out_nodata)
        return CustomGdalDataset(new_dst)

    def _multi_band_to_mem(self,
        ary: np.ndarray, 
        data_type: int=gdal.GDT_Float32, 
        out_nodata: Any=np.nan,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。
        Args:
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. 
                [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            out_nodata(Any): NoData. Defaults to np.nan
            **kwargs: データセットのパラメータ
                - xsize(int): X方向のサイズ
                - ysize(int): Y方向のサイズ
                - count(int): バンド数
                - transform(list): GeoTransform
                - projection(str): Projection
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        """
        # メモリ上に新しいラスターデータを作成する
        new_dst = self.__create_dataset(data_type, **kwargs)
        for i in range(self.RasterCount):
            band = new_dst.GetRasterBand(i+1)
            band.WriteArray(ary[i])
            band.SetNoDataValue(out_nodata)
        return CustomGdalDataset(new_dst)
    
    def _nodata_to(self, 
        ary: np.ndarray, 
        in_nodata: Any, 
        out_nodata: Any
    ) -> gdal.Dataset:
        ary = np.where(ary == in_nodata, out_nodata, ary)
        ary = np.where(np.isnan(ary), out_nodata, ary)
        ary = np.where(np.isinf(ary), out_nodata, ary)
        return ary

    def __create_dataset(self, data_type: int, **kwargs) -> gdal.Driver:
        """
        ## Summary
            メモリ上に新しい`gdal.Dataset`を作成する。
        Args:
            data_type(int): データ型
            **kwargs: データセットのパラメータ
                - xsize(int): X方向のサイズ
                - ysize(int): Y方向のサイズ
                - count(int): バンド数
                - transform(list): GeoTransform
                - projection(str): Projection
        Returns:
            (gdal.Dataset): 新しい`gdal.Dataset`
        """
        driver = gdal.GetDriverByName('MEM')
        driver.Register()
        dst = driver.Create(
            '',
            xsize=kwargs.get('xsize', self.RasterXSize),
            ysize=kwargs.get('ysize', self.RasterYSize),
            bands=kwargs.get('count', self.RasterCount),
            eType=data_type
        )
        dst.SetGeoTransform(kwargs.get('transform', self.GetGeoTransform()))
        dst.SetProjection(kwargs.get('projection', self.GetProjection()))
        return dst

    def fill_nodata(self, 
        max_search_distance: int, 
        smoothing: int=10
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            NoDataを埋める
        Args:
            max_search_distance(int): 最大探索距離
            smoothing(int, optional): スムージングの回数. Defaults to 10.
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        """
        if self.RasterCount == 1:
            return self._fill_nodata_of_single_band(max_search_distance, smoothing)
        return self._fill_nodata_of_multi_band(max_search_distance, smoothing)
    
    def _fill_nodata_of_single_band(self,
        max_search_distance: int,
        smoothing: int
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            SingleBandのNoDataを埋める
        Args:
            max_search_distance(int): 最大探索距離
            smoothing(int, optional): スムージングの回数. Defaults to 10.
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        """
        # 入力されるデータセットの作成
        write_dst = self.copy_dataset()
        # マスクに使用するデータセット
        mask_dst = self.copy_dataset()
        mask_ary = mask_dst.array()
        mask_ary = np.where(np.isnan(mask_ary), False, True)
        mask_band = mask_dst.GetRasterBand(1)
        mask_band.WriteArray(mask_ary)
        write_band = write_dst.GetRasterBand(1)
        gdal.FillNodata(
            write_band, 
            mask_band, 
            maxSearchDist=max_search_distance, 
            smoothingIterations=smoothing
        )
        return CustomGdalDataset(write_dst.dataset)
    
    def _fill_nodata_of_multi_band(self,
        max_search_distance: int,
        smoothing: int
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            MultiBandの`gdal.Dataset`のNoDataを埋める※ MultiBandの場合は int型を使用しているのでnp.nanは使用できない。修正が必要？
        Args:
            max_search_distance(int): 最大探索距離
            smoothing(int, optional): スムージングの回数. Defaults to 10.
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        """
        # 入力されるデータセットの作成
        write_dst = self.copy_dataset()
        # マスクに使用するデータセット
        mask_dst = self.copy_dataset()
        # 穴埋め箇所をFalse, それ以外をTrueに変換
        mask_ary = mask_dst.array()[0]
        mask_ary = np.where(np.isnan(mask_ary), False, True)
        for write_band, mask_band in zip(write_dst._band_generator, mask_dst._band_generator):
            mask_band.WriteArray(mask_ary)
            gdal.FillNodata(
                write_band, 
                mask_band, 
                maxSearchDist=max_search_distance, 
                smoothingIterations=smoothing
            )
        return CustomGdalDataset(write_dst.dataset)

    def expansion_dst(self, 
        vertical: int, 
        horizontal: int
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            ラスターデータを拡張する。拡張後のサイズは、(rows + vertical, cols + horizontal)となる。これは畳み込み処理を行う際等に使用する。
        Args:
            vertical(int): 上下方向の拡張数
            horizontal(int): 左右方向の拡張数
        Returns:
            CustomGdalDataset(gdal.Dataset): 拡張された gdal.Dataset
        """
        # 拡張後のサイズを計算
        x_expa = self.x_resolution * horizontal
        y_expa = abs(self.y_resolution * vertical)
        transform = list(self.GetGeoTransform())
        transform[0] -= x_expa
        transform[3] += y_expa
        # 拡張後の配列を作成
        ary = self.array()
        nodata = self.GetRasterBand(1).GetNoDataValue()
        v_ary = np.zeros((vertical, ary.shape[1]))
        v_ary[:] = nodata
        ary = np.vstack((v_ary, ary, v_ary))
        h_ary = np.zeros((ary.shape[0], horizontal))
        h_ary[:] = nodata
        ary = np.hstack((h_ary, ary, h_ary))
        # メモリ上に新しいラスターデータを作成
        if 2 < len(ary.shape):
            count = ary.shape[2]
        else:
            count = 1
        kwargs = {
            'transform': transform, 
            'projection': self.GetProjection()
        }
        new_dst = self.write_ary_to_mem(ary, raster_count=count, **kwargs)
        return new_dst.fill_nodata(max([vertical, horizontal]) + 3, 10)

    ##########################################################################
    # -------------- Methods for calculation data coordinates. --------------
    def bounds(self) -> Bounds:
        """
        ## Summary
            `gdal.Dataset`の範囲を取得する。
        Returns:
            Bounds(NamedTuple): (x_min, y_min, x_max, y_max)
        Examples:
            >>> bounds: Bounds = dst.bounds
        """
        transform = self.GetGeoTransform()
        x_min = transform[0]
        y_max = transform[3]
        rows = self.RasterYSize
        cols = self.RasterXSize
        x_max = x_min + cols * self.x_resolution
        y_min = y_max + rows * self.y_resolution
        return Bounds(x_min, y_min, x_max, y_max)
    
    @__check_crs(0, 'out_crs')
    def reprojected_bounds(self, 
        out_crs: Optional[Union[str, int, pyproj.CRS]]
    ) -> Bounds:
        """
        ## Summary
            投影変換した後の範囲を取得する。
        Args:
            out_wkt_crs(str | int | pyproj.CRS): 出力先のCRS
        Returns:
            Bounds(NamedTuple): 投影変換後の範囲。(x_min, y_min, x_max, y_max)
        Examples:
            >>> out_wkt_crs = pyproj.CRS(6690).to_wkt()
            >>> bounds: Bounds = dst.reprojected_bounds(out_wkt_crs)
        """
        bounds = self.bounds()
        in_crs = self.GetProjection()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        xs, ys = gdal_utils.reproject_xy(
            [bounds.x_min, bounds.x_max], 
            [bounds.y_min, bounds.y_max], 
            in_crs, out_crs
        )
        return Bounds(xs[0], ys[0], xs[1], ys[1])
    
    @__check_crs(0, 'out_crs')
    def center(self, out_crs: Optional[Union[str, int, pyproj.CRS]]=None) -> XY:
        """
        ## Summary
            `gdal.Dataset`の中心座標を取得する。
        Args:
            out_crs(str | int | pyproj.CRS): 出力先のCRS
        Returns:
            XY(NamedTuple): (x, y)
        Examples:
            >>> center: XY = dst.center()
            >>> x: float = center.x
            >>> y: float = center.y
        """
        bounds = self.bounds()
        x = (bounds.x_min + bounds.x_max) / 2
        y = (bounds.y_min + bounds.y_max) / 2
        if not out_crs is None:
            return gdal_utils.reproject_xy(x, y, self.GetProjection(), out_crs)
        return XY(x, y)

    def cells_center_coordinates(self) -> Coordinates:
        """
        ## Summary
            各セルの中心座標を計算。戻り値はX座標とY座標の2次元配列であり、各数値がセルの中心座標を示す。
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次元配列
        """
        bounds = self.bounds()
        # X方向のセルの中心座標を計算し、1次元配列に
        x_resol = self.x_resolution
        # 0.5はセルの中心を示す
        half = 0.5
        X = np.arange(bounds.x_min, bounds.x_max, x_resol) + x_resol * half
        # Y方向のセルの中心座標を計算し、1次元配列に
        y_resol = self.y_resolution
        Y = np.arange(bounds.y_max, bounds.y_min, y_resol) + y_resol * half
        # 各セルの中心座標を計算
        return Coordinates(*np.meshgrid(X, Y))
    
    def cells_upper_left_corner_coordinates(self) -> Coordinates:
        """
        ## Summary
            各セルの左上の座標を計算。戻り値はX座標とY座標の2次元配列であり、各数値がセルの左上の座標を示す。
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次
        """
        bounds = self.bounds()
        # X方向のセルの中心座標を計算し、1次元配列に
        x_resol = self.x_resolution
        X = np.arange(bounds.x_min, bounds.x_max, x_resol)
        # Y方向のセルの中心座標を計算し、1次元配列に
        y_resol = self.y_resolution
        Y = np.arange(bounds.y_max, bounds.y_min, y_resol)
        # 各セルの左上の座標を計算
        return Coordinates(*np.meshgrid(X, Y))

    def cells_upper_right_corner_coordinates(self) -> Coordinates:
        """
        ## Summary
            各セルの右上の座標を計算。戻り値はX座標とY座標の2次元配列であり、各数値がセルの右上の座標を示す。
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次元配列
        """
        bounds = self.bounds()
        # X方向のセルの中心座標を計算し、1次元配列に
        x_resol = self.x_resolution
        X = np.arange(bounds.x_min, bounds.x_max, x_resol) + x_resol
        # Y方向のセルの中心座標を計算し、1次元配列に
        y_resol = self.y_resolution
        Y = np.arange(bounds.y_max, bounds.y_min, y_resol)
        # 各セルの右上の座標を計算
        return Coordinates(*np.meshgrid(X, Y))
    
    def cells_lower_left_corner_coordinates(self) -> Coordinates:
        """
        ## Summary
            各セルの左下の座標を計算。戻り値はX座標とY座標の2次元配列であり、各数値がセルの左下の座標を示す。
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次元配列
        """
        bounds = self.bounds()
        # X方向のセルの中心座標を計算し、1次元配列に
        x_resol = self.x_resolution
        X = np.arange(bounds.x_min, bounds.x_max, x_resol)
        # Y方向のセルの中心座標を計算し、1次元配列に
        y_resol = self.y_resolution
        Y = np.arange(bounds.y_max, bounds.y_min, y_resol) + y_resol
        # 各セルの左下の座標を計算
        return Coordinates(*np.meshgrid(X, Y))
    
    def cells_lower_right_corner_coordinates(self) -> Coordinates:
        """
        ## Summary
            各セルの右下の座標を計算。戻り値はX座標とY座標の2次元配列であり、各数値がセルの右下の座標を示す。
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次元配列
        """
        bounds = self.bounds()
        # X方向のセルの中心座標を計算し、1次元配列に
        x_resol = self.x_resolution
        X = np.arange(bounds.x_min, bounds.x_max, x_resol) + x_resol
        # Y方向のセルの中心座標を計算し、1次元配列に
        y_resol = self.y_resolution
        Y = np.arange(bounds.y_max, bounds.y_min, y_resol) + y_resol
        # 各セルの右下の座標を計算
        return Coordinates(*np.meshgrid(X, Y))
    
    def to_geodataframe_xy(self, **kwargs) -> gpd.GeoDataFrame:
        """
        ## Summary
            RasterDataのセル値をshapely.PointにしてGeoDataFrameに入力。バンド数に応じて列数が増える。
        Args:
            **kwargs:
                - position(str): セルの位置。'center' or 'upper_left' or 'upper_right' or 'lower_left' or 'lower_right'. Defaults to 'center'.
        Returns:
            gpd.GeoDataFrame
                x(float): X座標
                y(float): Y座標
                band_1(Any): バンド1の値
                ...
                geometry: shapely.Point
        Examples:
            >>> gdf: geopandas.GeoDataFrame = dst.to_geodataframe()
        """
        # セルの座標を取得
        position = kwargs.get('position', 'center').lower()
        if position == 'upper_left':
            cds = self.cells_upper_left_corner_coordinates()
        elif position == 'upper_right':
            cds = self.cells_upper_right_corner_coordinates()
        elif position == 'lower_left':
            cds = self.cells_lower_left_corner_coordinates()
        elif position == 'lower_right':
            cds = self.cells_lower_right_corner_coordinates()
        else:
            cds = self.cells_center_coordinates()
        data = {
            'x': cds.X.flatten(),
            'y': cds.Y.flatten(),
        }
        # 各バンドの値を取得
        band = self.GetRasterBand(1)
        ary = self._nodata_to(self.ReadAsArray(), band.GetNoDataValue(), np.nan)
        data_type = band.DataType
        dst = self.write_ary_to_mem(ary, data_type, np.nan)
        for i in range(self.RasterCount):
            ary1d = dst.GetRasterBand(i+1).ReadAsArray().ravel()
            data[f'band_{i+1}'] = ary1d
        # GeoDataFrameを作成
        geoms = gpd.points_from_xy(data['x'], data['y'])
        # 小数部の影響か、たまに範囲外の座標が生成されるので、範囲内のものだけを取得
        data, geoms = self._adjustment_of_length(geoms, data)
        return gpd.GeoDataFrame(data, geometry=geoms, crs=self.GetProjection())

    def to_pandas_xy(self, digit=9, **kwargs) -> pd.DataFrame:
        """
        ## Summary
            RasterDataのセル値をshapely.PointにしてDataFrameに入力。これは強制的にEPSG:4326に投影変換する。
        Args:
            digit(int): 小数点以下の桁数
            **kwargs:
                - position(str): セルの位置。'center' or 'upper_left' or 'upper_right' or 'lower_left' or 'lower_right'. Defaults to 'center'.
        Returns:
            pd.DataFrame
                x(float): X座標
                y(float): Y座標
                band_1(Any): バンド1の値
                ...
                geometry: Wkt形式のPoint
        Examples:
            >>> df: pandas.DataFrame = dst.to_pandas()
        """
        gdf = self.to_geodataframe_xy(**kwargs)
        geoms = gdf.geometry
        df = gdf.drop('geometry', axis=1)
        epsg = gdf.crs.to_epsg()
        if epsg != 4326:
            geoms = geoms.to_crs(epsg=4326)
        wkt_points = []
        for geom in geoms:
            x, y = geom.x, geom.y
            wkt_points.append(f'POINT ({x:.{digit}f} {y:.{digit}f})')
        df['geometry'] = wkt_points
        return df

    def _adjustment_of_length(self,
        geoms: gpd.GeoSeries,
        data: Dict[str, np.ndarray]
    ) -> Any:
        """
        ## Summary
            小数部の影響か、たまに範囲外の座標が生成されるので、範囲内のものだけを取得
        Args:
            geoms(gpd.GeoSeries): ジオメトリ
            data(Dict[str, np.ndarray]): データ
        Returns:
            Tuple[Dict[str, np.ndarray], gpd.GeoSeries]
        Examples:
            >>> data, geoms = self._adjustment_of_length(geoms, data)
            >>> gdf = gpd.GeoDataFrame(data, geometry=geoms)
        """
        if 1 < len(np.unique([len(ary1d) for ary1d in data.values()])):
            x = data.get('x')
            y = data.get('y')
            bounds = shapely.box(*self.bounds)
            idx = np.where(geoms.intersects(bounds))[0]
            data['x'] = x[idx]
            data['y'] = y[idx]
            geoms = geoms[idx]
            return data, geoms
        else:
            return data, geoms

    def check_crs_is_metre(self) -> bool:
        """
        ## Summary
            投影法がメートル法かどうかを判定する。
        Returns:
            (bool): メートル法の場合はTrue、それ以外はFalse
        Examples:
            >>> is_metre: bool = dst.check_crs_is_metre()
            False
            >>> file_path = r'.\\raster_epsg6690.tif''
            >>> dst = CustomGdalDataset(file_path)
            >>> is_metre: bool = dst.check_crs_is_metre()
            True
        """
        crs = pyproj.CRS(self.GetProjection())
        return crs.axis_info[0].unit_name == 'metre'

    @__check_datum
    def estimate_utm_crs(self, **kwargs) -> str:
        """
        ## Summary
            UTMのCRSを推定する。日本の場合は "datum_name='JGD2011'" を指定する。
        Args:
            kwargs:
                datum_name: 'WGS 84', 'JGD2011' ...  default='JGD2011'
        Returns:
            (str): WKT-CRS
        Examples:
            >>> file_path = r'.\\raster.tif''
            >>> dst = CustomGdalDataset(file_path)
            >>> wkt_crs: str = dst.estimate_utm_crs()
        """
        datum_name = kwargs.get('datum_name', 'JGD2011')
        org_crs = pyproj.CRS(self.GetProjection())
        center = self.center()
        if org_crs.to_epsg() != 4326:
            # 座標系がWGS84でない場合は、一時的にWGS84に変換する
            wgs_crs = pyproj.CRS(4326).to_wkt()
            center = gdal_utils.reproject_xy(center.x, center.y, org_crs.to_wkt(), wgs_crs)
        # UTMのCRSを推定する
        aoi = pyproj.aoi.AreaOfInterest(
            west_lon_degree=center.x,
            south_lat_degree=center.y,
            east_lon_degree=center.x,
            north_lat_degree=center.y,
        )
        utm_crs_lst = pyproj.database.query_utm_crs_info(
            datum_name=datum_name,
            area_of_interest=aoi
        )
        return pyproj.CRS.from_epsg(utm_crs_lst[0].code).to_wkt()

    def cell_size_in_metre(self, digit=4) -> CellSize:
        """
        ## Summary
            セルサイズをMetreで取得する。
        Args:
            digit(int): 小数点以下の桁数
        Returns:
            CellSize: セルサイズ
                - x(float): X方向のセルサイズ
                - y(float): Y方向のセルサイズ
        """
        x_resol = self.x_resolution
        y_resol = self.y_resolution
        if self.check_crs_is_metre():
            return CellSize(round(x_resol, digit+1), round(y_resol, digit+1))
        utm_crs = self.estimate_utm_crs()
        bounds = self.reprojected_bounds(utm_crs)
        return self._cell_size_from_bounds(bounds, digit)
    
    def cell_size_in_degree(self, digit=9) -> CellSize:
        """
        ## Summary
            セルサイズをDegreeで取得する。
        Args:
            digit(int): 小数点以下の桁数
        Returns:
            CellSize: セルサイズ
                - x(float): X方向のセルサイズ
                - y(float): Y方向のセルサイズ
        """
        x_resol = self.x_resolution
        y_resol = self.y_resolution
        if self.check_crs_is_metre():
            wkt_crs = pyproj.CRS(4326).to_wkt()
            bounds = self.reprojected_bounds(wkt_crs)
            return self._cell_size_from_bounds(bounds, digit)
        return CellSize(round(x_resol, digit+1), round(y_resol, digit+1))
    
    def _cell_size_from_bounds(self, bounds: Bounds, digit: int) -> CellSize:
        """
        ## Summary
            範囲の座標からセルサイズを計算する
        Args:
            bounds(Bounds): 範囲
            digit(int): 小数点以下の桁数
        Returns:
            CellSize: セルサイズ
                - x(float): X方向のセルサイズ
                - y(float): Y方向のセルサイズ
        """
        x_resol = abs(bounds.x_max - bounds.x_min) / self.RasterXSize
        y_resol = abs(bounds.y_max - bounds.y_min) / self.RasterYSize
        return CellSize(round(x_resol, digit+1), round(y_resol, digit+1))

    ############################################################################
    # ----------------- Methods for projection transform. -----------------
    @__check_crs(0, 'out_crs')
    def reprojected_dataset(self, 
        out_crs: str
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            'gdal.Dataset'の投影変換。
        Args:
            out_crs(str | int | pyproj.CRS): 出力のCRS. WKT形式、EPSGコード、pyproj.CRSのいずれか
        Returns:
            CustomGdalDataset(gdal.Dataset): 投影変換後のラスターデータ
        Examples:
            >>> dst = CustomGdalDataset(file_path) #EPSG:4326
            >>> out_crs = pyproj.CRS(6690).to_wkt()
            >>> reprojected_dataset: gdal.Dataset = dst.reprojected_dataset(out_crs)
        """
        # 投影変換の為のオプションを設定
        ops = gdal.WarpOptions(
            format='MEM',
            srcSRS=self.GetProjection(),
            dstSRS=out_crs,
            outputBounds=self.reprojected_bounds(out_crs),
            width=self.RasterXSize,
            height=self.RasterYSize,
            resampleAlg=gdal.GRA_CubicSpline
        )
        dst = gdal.Warp('', self.dataset, options=ops)
        return CustomGdalDataset(dst)

    def estimate_utm_and_reprojected_dataset(self,
        datum_name: str='JGD2011'
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            UTMのCRSを推定して投影変換を行う。日本で使用する場合は "datum_name='JGD2011'" を指定する。
        Args:
            dst(gdal.Dataset): ラスターデータ
            datum_name: 'WGS 84', 'JGD2011' など
        Returns:
            CustomGdalDataset(gdal.Dataset): 投影変換後のラスターデータ
        Examples:
            >>> new_dst: gdal.Dataset = dst.estimate_utm_and_reprojected_dataset('JGD2011')
        """
        out_wkt_crs = self.estimate_utm_crs(datum_name)
        return self.reprojected_dataset(out_wkt_crs)

    ############################################################################
    # ----------------- Methods for resampling dataset. -----------------
    def _resample_option_template_with_resol_spec(self,
        x_resolution: float,
        y_resolution: float,
        resample_algorithm: int=gdal.GRA_CubicSpline
    ) -> gdal.WarpOptions:
        """
        ## Summary
            分解能を指定するためのオプションテンプレートを作成する。
        Args:
            x_resolution(float): X方向の解像度
            y_resolution(float): Y方向の解像度
            resample_algorithm(int, optional): 補間方法. Defaults to gdal.GRA_CubicSpline. [gdal.GRA_NearestNeighbour, gdal.GRA_Bilinear, gdal.GRA_Cubic, gdal.GRA_CubicSpline, gdal.GRA_Lanczos]
        Returns:
            gdal.WarpOptions:
        """
        return gdal.WarpOptions(
            format='MEM',
            xRes=x_resolution,
            yRes=y_resolution,
            resampleAlg=resample_algorithm,
            outputBounds=self.bounds()
        )

    def _resample_option_temppate_with_cells_spec(self,
        x_cells: int,
        y_cells: int,
        resample_algorithm: int=gdal.GRA_CubicSpline
    ) -> gdal.WarpOptions:
        """
        ## Summary
            セル数を指定するためのオプションテンプレートを作成する
        Args:
            x_cells(int): X方向のセル数
            y_cells(int): Y方向のセル数
            resample_algorithm(int, optional): 補間方法. Defaults to gdal.GRA_CubicSpline. [gdal.GRA_NearestNeighbour, gdal.GRA_Bilinear, gdal.GRA_Cubic, gdal.GRA_CubicSpline, gdal.GRA_Lanczos]
        Returns:
            gdal.WarpOptions:
        """
        return gdal.WarpOptions(
            format='MEM',
            width=x_cells,
            height=y_cells,
            resampleAlg=resample_algorithm,
            outputBounds=self.bounds()
        )

    def resample_with_resol_spec(self,
        x_resolution: float,
        y_resolution: float,
        resample_algorithm: int=gdal.GRA_CubicSpline,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            分解能を指定してラスターデータをリサンプリングする。このメソッドでは EPSG:4326 のデータセットでも分解能をメートル単位で指定することができる。
        Args:
            x_resolution(float): X方向の分解能
            y_resolution(float): Y方向の分解能
            resample_algorithm(int, optional): 
                補間方法.\n 
                Defaults to gdal.GRA_CubicSpline. \n
                [gdal.GRA_NearestNeighbour, gdal.GRA_Bilinear, gdal.GRA_Cubic, gdal.GRA_CubicSpline, gdal.GRA_Lanczos]
            forced_metre_system(bool): 
                Degree単位のデータセットもメートル単位で指定するかどうか.\n 
                Defaults to True.\n
                Trueの場合は、メートル単位で指定した解像度をDegree単位に変換した後でリサンプリングを行う。
            datum_name(str): 'WGS 84', 'JGD2011' など. Defaults to 'JGD2011'
        Returns:
            CustomGdalDataset(gdal.Dataset): リサンプリング後のラスターデータ
        Examples:
            ### メートル単位で指定する場合
            >>> dst = CustomGdalDataset(file_path) #EPSG:4326
            >>> new_dst: gdal.Dataset = dst.resample_with_resol_spec(5, 5)
            ------------------------------------------------------------
            
            ### 指定した分解能そのままで指定する場合
            >>> new_dst: gdal.Dataset = dst.resample_with_resol_spec(0.0001, 0.0001, forced_metre_system=False)
        """
        if kwargs.get('forced_metre_system', True) and not self.check_crs_is_metre():
            # Degree単位のデータセットもメートル単位で指定する場合
            center = self.center()
            datum_name = kwargs.get('datum_name', 'JGD2011')
            wkt_crs = self.estimate_utm_crs(datum_name=datum_name)
            # x_resolutionをMetreからDegree単位に変換
            x_resolution = gdal_utils.degree_from_metre(
                x_resolution, center.x, center.y, 
                wkt_crs, x_direction=True
            )
            # y_resolutionをMetreからDegree単位に変換
            y_resolution = gdal_utils.degree_from_metre(
                y_resolution, center.x, center.y, 
                wkt_crs, x_direction=False
            )
        ops = self._resample_option_template_with_resol_spec(
            x_resolution, y_resolution, resample_algorithm)
        return CustomGdalDataset(gdal.Warp('', self.dataset, options=ops))

    def resample_with_cells_spec(self,
        x_cells: int,
        y_cells: int,
        resample_algorithm: int=gdal.GRA_CubicSpline
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            XYのセル数を指定してラスターデータをリサンプリングする。
        Args:
            x_cells(int): X方向のセル数
            y_cells(int): Y方向のセル数
            resample_algorithm(int, optional): 
                補間方法.\n 
                Defaults to gdal.GRA_CubicSpline. \n
                [gdal.GRA_NearestNeighbour, gdal.GRA_Bilinear, gdal.GRA_Cubic, gdal.GRA_CubicSpline, gdal.GRA_Lanczos]
        Returns:
            CustomGdalDataset(gdal.Dataset): リサンプリング後のラスターデータ
        Examples:
            >>> new_dst: gdal.Dataset = dst.resample_with_cells(100, 100)
        """
        ops = self._resample_option_temppate_with_cells_spec(
            x_cells, y_cells, resample_algorithm)
        return CustomGdalDataset(gdal.Warp('', self.dataset, options=ops))

    ############################################################################
    # ----------------- Methods for clipping dataset. -----------------
    def _clip_option_template_with_wkt_poly_spec(self,
        wkt_poly: str,
        fmt: str='MEM',
        nodata: Any=np.nan,
        **kwargs
    ) -> gdal.Dataset:
        """
        ## Summary
            gdal.WarpでRasterを切り抜く為のオプションテンプレートを作成する
        Args:
            dst (gdal.Dataset): 切り抜くRaster
            wkt_poly (str): 切り抜く範囲のWKT形式のポリゴン
            fmt (str, optional): 出力形式. Defaults to 'MEM'.
            nodata (Any, optional): 出力RasterのNoData値. Defaults to np.nan.
        Returns:
            gdal.WarpOptions: gdal.WarpでRasterを切り抜く為のオプション
        """
        poly_crs = kwargs.get('poly_crs', self.GetProjection())
        if poly_crs != self.GetProjection() and not isinstance(poly_crs, str):
            poly_crs = poly_crs.to_wkt()

        return gdal.WarpOptions(
            format=fmt,
            cutlineWKT=wkt_poly,
            cropToCutline=True,
            cutlineSRS=poly_crs,
            srcSRS=self.GetProjection(),
            dstNodata=nodata,
            srcNodata=self.GetRasterBand(1).GetNoDataValue()
        )
    
    @__wkt_geometry_check(0, 'wkt_poly')
    def clip_by_wkt_poly(self, 
        wkt_poly: str | shapely.Polygon, 
        nodata: Any=np.nan,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            ポリゴンでラスターデータをクリップする。このメソッドは、ポリゴンの投影法が異なる場合にも使用できる。
        Args:
            wkt_poly(str): WKT形式のポリゴン
            nodata(Any, optional): NoData. Defaults to np.nan
            kwargs:
                poly_crs(str): ポリゴンの投影法. Defaults to None. これを指定すれば Raster と異なる投影法のポリゴンを使用できる。
        Returns:
            CustomGdalDataset(gdal.Dataset): クリップ後のラスターデータ
        Examples:
            >>> wkt_poly = 'POLYGON ((x1 y1, x2 y2, x3 y3, x4 y4, x1 y1))'
            >>> new_dst: gdal.Dataset = dst.clip_by_wkt_poly(wkt_poly)
        """
        # ポリゴンの投影法が異なり、かつ指定されている場合は、ポリゴンを投影変換する
        options = self._clip_option_template_with_wkt_poly_spec(
            wkt_poly, 
            nodata=nodata, 
            poly_crs=kwargs.get('poly_crs', self.GetProjection())
        )
        return CustomGdalDataset(gdal.Warp('', self.dataset, options=options))
    
    @__wkt_geometry_check(0, 'wkt_poly')
    def clip_by_bounds(self, 
        wkt_poly: str | shapely.Polygon,
        nodata: Any=np.nan,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            Polygonのバウンディングボックスでラスターデータをクリップする。このメソッドは、ポリゴンの投影法が異なる場合にも使用できる。
        Args:
            wkt_poly(str): WKT形式のポリゴン
            nodata(Any, optional): NoData. Defaults to np.nan
            kwargs:
                poly_crs(str): ポリゴンの投影法. Defaults to None. これを指定すれば Raster と異なる投影法のポリゴンを使用できる。
        Returns:
            CustomGdalDataset(gdal.Dataset): クリップ後のラスターデータ
        """
        if isinstance(wkt_poly, str):
            wkt_poly = shapely.from_wkt(wkt_poly).envelope.wkt
        elif isinstance(wkt_poly, shapely.geometry.base.BaseGeometry):
            wkt_poly = wkt_poly.envelope.wkt
        else:
            custom_gdal_exception.load_wkt_geometry_err()
        poly_crs = kwargs.get('poly_crs', self.GetProjection())
        return self.clip_by_wkt_poly(wkt_poly, nodata, poly_crs=poly_crs)

    @__wkt_geometry_check(0, 'wkt_poly')
    def clip_by_fit_bounds(self, 
        wkt_poly: str | shapely.Polygon, 
        nodata: Any=np.nan, 
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            ポリゴンの最小外接矩形でラスターデータをクリップする。このメソッドは、ポリゴンの投影法が異なる場合にも使用できる。
        Args:
            wkt_poly(str): WKT形式のポリゴン
            nodata(Any, optional): NoData. Defaults to np.nan
            kwargs:
                poly_crs(str): ポリゴンの投影法. Defaults to None. これを指定すれば Raster と異なる投影法のポリゴンを使用できる。
        Returns:
            CustomGdalDataset(gdal.Dataset): クリップ後のラスターデータ
        """
        if isinstance(wkt_poly, str):
            wkt_poly = shapely.from_wkt(wkt_poly).minimum_rotated_rectangle.wkt
        elif isinstance(wkt_poly, shapely.geometry.base.BaseGeometry):
            wkt_poly = wkt_poly.minimum_rotated_rectangle.wkt
        else:
            custom_gdal_exception.load_wkt_geometry_err()
        poly_crs = kwargs.get('poly_crs', self.GetProjection())
        return self.clip_by_wkt_poly(wkt_poly, nodata, poly_crs=poly_crs)

    ############################################################################
    # ----------------- Methods for mask dataset. -----------------
    @__wkt_geometry_check(0, 'wkt_geom')
    @__check_crs(1, 'in_wkt_crs')
    def get_masked_array(self, 
        wkt_geom: str,
        in_wkt_crs: str,
        masked_value: Any,
        bands: List[int] = [1],
        all_touched: bool=True,
        inverse: bool=False
    ) -> np.ndarray:
        """
        ## Summary
            Geometryでラスターデータをマスクする。このメソッドは、Geometryの投影法が異なる場合にも使用できる。
        Args:
            wkt_geom(str): WKT形式のGeometry
            in_wkt_crs(str): 入力のWKT-CRS
            masked_value(Any): マスクする値
            bands(List[int], optional): バンド番号. Defaults to [1].
            all_touched(bool, optional): マスクするセルの条件. Defaults to True.
            inverse(bool, optional): マスクを反転するかどうか. Defaults to False.
        Returns:
            np.ndarray: マスク後のラスターデータ
        Examples:
            >>> wkt_geom = 'POLYGON ((x1 y1, x2 y2, x3 y3, x4 y4, x1 y1))'
            >>> masked_value = np.nan
            >>> ary = dst.get_masked_array(wkt_geom, masked_value)
        """
        data_source = self._create_ogr_lyr(wkt_geom, in_wkt_crs)
        mask_lyr = data_source.GetLayer(0)
        mask_dst = self.copy_dataset()
        ops = dict(ALL_TOUCHED=all_touched)
        gdal.RasterizeLayer(
            mask_dst, bands, mask_lyr, 
            burn_values=[masked_value], options=ops
        )
        ary = mask_dst.ReadAsArray()
        mask_dst = None
        if inverse:
            # Optionsの'INVERSE'が機能しないため、ここでマスクを反転させる
            if np.isnan(masked_value):
                inversed_ary = np.where(np.isnan(ary), ary, masked_value)
            else:
                inversed_ary = np.where(ary == masked_value, ary, masked_value)
            return inversed_ary
        else:
            return ary
    
    def _create_ogr_lyr(self,
        wkt_geom: str, 
        in_wkt_crs: str,
    ) -> ogr.DataSource:
        """
        ## Summary
            WKT-PolygonをOGRレイヤーに変換する。これは`gdal.Dataset`をmaskする際に使用するので、in_wkt_crsとgdal.Datasetの投影法が異なる場合は、WKT-Polygonを投影変換する。
        Args:
            wkt_geom(str): WKT形式のGeometry
            in_wkt_crs(str): 入力のWKT-CRS
        Returns:
            ogr.Layer: OGRレイヤー
        Examples:
            >>> vector_dst = self._create_poly_lyr(wkt_geom, in_wkt_crs)
            >>> lyr = vector_dst.GetLayer(0)
        """
        if in_wkt_crs != self.GetProjection():
            wkt_geom = gdal_utils.reprojection_geometry(
                wkt_geometry=wkt_geom, 
                in_wkt_crs=in_wkt_crs, 
                out_wkt_crs=self.GetProjection()
            )
            in_wkt_crs = self.GetProjection()
        in_geom_type = shapely.from_wkt(wkt_geom).geom_type.upper()
        # GeoemtryTypes
        geom_types = {
            'POINT': ogr.wkbPoint,
            'MULTIPOINT': ogr.wkbMultiPoint,
            'LINESTRING': ogr.wkbLineString,
            'MULTILINESTRING': ogr.wkbMultiLineString,
            'LINEARRING': ogr.wkbLinearRing,
            'MULTILINEARRING': ogr.wkbMultiLineString,
            'POLYGON': ogr.wkbPolygon,
            'MULTIPOLYGON': ogr.wkbMultiPolygon
        }
        geom_type = geom_types.get(in_geom_type)
        if geom_type is not None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(in_wkt_crs)
            vector_dst = ogr.GetDriverByName('Memory').CreateDataSource('out')
            lyr = vector_dst.CreateLayer('', srs, geom_type)
            feature_defn = lyr.GetLayerDefn()
            feature = ogr.Feature(feature_defn)
            feature.SetGeometry(ogr.CreateGeometryFromWkt(wkt_geom))
            lyr.CreateFeature(feature)
            return vector_dst
        else:
            raise ValueError('Invalid geometry type')

    ############################################################################
    # ----------------- Statistical methods for dataset. -----------------
    def normalized_array(self) -> np.ndarray:
        """
        ## Summary
            ラスターデータを正規化する
        Returns:
            np.ndarray: 正規化後のラスターデータ
        """
        ary = self.ReadAsArray()
        min_ = np.nanmin(ary)
        max_ = np.nanmax(ary)
        return (ary - min_) / (max_ - min_)
    
    def outlier_treatment_array_by_std(self, sigma: float=2) -> np.ndarray:
        """
        ## Summary
            ラスターデータの外れ値を標準偏差で処理する
        Args:
            threshold(float, optional): 標準偏差の倍数. Defaults to 2.
        Returns:
            np.ndarray: 外れ値処理後のラスターデータ
        """
        ary = self.ReadAsArray()
        mean = np.nanmean(ary)
        std = np.nanstd(ary)
        upper = mean + sigma * std
        lower = mean - sigma * std
        return np.where(upper < ary, upper, np.where(ary < lower, lower, ary))
    
    def outlier_treatment_array_by_quantile(self, threshold: float=1.5) -> np.ndarray:
        """
        ## Summary
            ラスターデータの外れ値を四分位範囲で処理する
        Args:
            threshold(float, optional): 四分位範囲の倍数. Defaults to 1.5.
        Returns:
            np.ndarray: 外れ値処理後のラスターデータ
        """
        ary = self.ReadAsArray()
        q1 = np.nanpercentile(ary, 25)
        q3 = np.nanpercentile(ary, 75)
        iqr = q3 - q1
        upper = q3 + threshold * iqr
        lower = q1 - threshold * iqr
        return np.where(upper < ary, upper, np.where(ary < lower, lower, ary))

    ############################################################################
    # ----------------- DEM processing methods for dathaset. -----------------
    @__band_check(count=1)
    def hillshade(self, 
        azimuth: int=315, 
        altitude: int=45,
        z_factor: float=1,
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            陰影起伏図を作成する。このメソッドは、DEM（DTM)の処理に使用される。
        Args:
            azimuth(int, optional): 方位角. Defaults to 315.
            altitude(int, optional): 高度. Defaults to 45.
            z_factor(float, optional): Zファクター. Defaults to 1.
            kwargs:
                - alg(str): アルゴリズム. Horn | ZevenbergenThorne. Defaults to 'Horn'.
                - scale(float, optional): Ratio of vertical units to horizontal. If the horizontal unit of the source DEM is degrees (e.g Lat/Long WGS84 projection), you can use scale=111120 if the vertical units are meters (or scale=370400 if they are in feet)
                - combined(bool): combined shading, a combination of slope and oblique shading. Defaults to False.
                - multiDirectional(bool): multidirectional shading, a combination of hillshading illuminated from 225 deg, 270 deg, 315 deg, and 360 deg azimuth.
                - return_array(bool): If True, return the result as a numpy array. Defaults to False.
        Returns:
            CustomGdalDataset(gdal.Dataset): 陰影起伏図の`gdal.Dataset`
        """
        # return_arrayが指定されている場合は、その値を取得して削除する
        return_array = kwargs.get('return_array', False)
        if 'return_array' in kwargs:
            del kwargs['return_array']

        options = {
            'destName': '',
            'srcDS': self.dataset,
            'processing': 'hillshade',
            'format': 'MEM',
            'azimuth': azimuth,
            'altitude': altitude,
            'zFactor': z_factor,
        }
        options.update(kwargs)
        if 'multiDirectional' in options:
            if options.get('multiDirectional') == True:
                del options['combined']
                del options['azimuth']
        new_dst = gdal.DEMProcessing(**options)
        if return_array:
            # return_arrayが指定されている場合は、numpy配列で返す
            hillshade_ary = new_dst.ReadAsArray()
            new_dst = None
            return hillshade_ary
        return CustomGdalDataset(new_dst)

    @__band_check(count=1)
    def slope(self, **kwargs) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            勾配を計算する。このメソッドは、DEM（DTM)の処理に使用される。
        Args:
            kwargs:
                - alg(str): アルゴリズム. ZevenbergenThorne | Horn. Defaults to 'Horn'.
                - percent(bool): 勾配をパーセントで出力するかどうか. Defaults to False.
        Returns:
            CustomGdalDataset(gdal.Dataset): 勾配の`gdal.Dataset`.
        """
        # return_arrayが指定されている場合は、その値を取得して削除する
        return_array = kwargs.get('return_array', False)
        if 'return_array' in kwargs:
            del kwargs['return_array']

        options = {
            'destName': '',
            'srcDS': self.dataset,
            'processing': 'slope',
            'format': 'MEM',
        }
        options.update(kwargs)
        options_list = None
        if 'percent' in options:
            if options.get('percent') == True:
                del options['percent']
                options_list = ['-p']
        if options_list is None:
            new_dst = gdal.DEMProcessing(**options)
        else:
            new_dst = gdal.DEMProcessing(**options, options=options_list)
        # Nodataが-9999の場合は、nanに変換し書き換える。
        ary = new_dst.ReadAsArray()
        band = new_dst.GetRasterBand(1)
        band.WriteArray(np.where(ary == -9999, np.nan, ary))
        band.SetNoDataValue(np.nan)
        if return_array:
            # return_arrayが指定されている場合は、numpy配列で返す
            slope_ary = new_dst.ReadAsArray()
            new_dst = None
            return slope_ary
        return CustomGdalDataset(new_dst)

    @__band_check(count=1)
    def aspect(self, 
        zero_for_flat=True, 
        **kwargs
    ) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            傾斜方位を計算する。このメソッドは、DEM（DTM)の処理に使用される。
        Args:
            zero_for_flat(bool, optional): 平坦部を0度にするかどうか、Falseならば-9999になる. Defaults to True.
            kwargs:
                - alg(str): アルゴリズム. ZevenbergenThorne | Horn. Defaults to 'Horn'.
        Returns:
            CustomGdalDataset(gdal.Dataset): 傾斜方位の`gdal.Dataset`.
        """
        # return_arrayが指定されている場合は、その値を取得して削除する
        return_array = kwargs.get('return_array', False)
        if 'return_array' in kwargs:
            del kwargs['return_array']

        options = {
            'destName': '',
            'srcDS': self.dataset,
            'processing': 'aspect',
            'format': 'MEM',
        }
        options.update(kwargs)
        if zero_for_flat:
            options_list = ['-zero_for_flat']
            new_dst = gdal.DEMProcessing(**options, options=options_list)
        else:
            new_dst = gdal.DEMProcessing(**options)
        if return_array:
            # return_arrayが指定されている場合は、numpy配列で返す
            aspect_ary = new_dst.ReadAsArray()
            new_dst = None
            return aspect_ary
        return CustomGdalDataset(new_dst)

    @__band_check(count=1)
    def tri(self, **kwargs) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            TRI（Topographic Roughness Index）を計算する。このメソッドは、DEM（DTM)の処理に使用される。
        Args:
            kwargs:
                alg(str): アルゴリズムの指定。Wilson | Riley. Defaults to 'Wilson'.
        Returns:
            CustomGdalDataset(gdal.Dataset): TRIの`gdal.Dataset`.
        """
        # return_arrayが指定されている場合は、その値を取得して削除する
        options = {
            'destName': '',
            'srcDS': self.dataset,
            'processing': 'TRI',
            'format': 'MEM',
            'alg': kwargs.get('alg', 'Wilson')
        }
        new_dst = gdal.DEMProcessing(**options)
        if kwargs.get('return_array', False):
            # return_arrayが指定されている場合は、numpy配列で返す
            tri_ary = new_dst.ReadAsArray()
            new_dst = None
            return tri_ary
        return CustomGdalDataset(new_dst)
    
    @__band_check(count=1)
    def tpi(self, **kwargs) -> Union['CustomGdalDataset', gdal.Dataset]:
        """
        ## Summary
            TPI（Topographic Position Index）を計算する。このメソッドは、DEM（DTM)の処理に使用される。
        Args:
            kwargs:
                - alg(str): アルゴリズムの指定。Horn | ZevenbergenThorne. Defaults to 'Horn'.
                - kernel(2D-array): 畳み込み用のカーネルを指定する。
                - outlier_treatment(float | False): 外れ値処理の倍数. Defaults to 1.5
                - return_array(bool): If True, return the result as a numpy array. Defaults to False.
        Returns:
            CustomGdalDataset(gdal.Dataset): TPIの`gdal.Dataset`.
        """
        if 'kernel' in kwargs:
            # 'kernel'が指定されている場合は、指定したカーネルで畳み込み処理を行う
            kernel = kwargs.get('kernel')
            rows, cols = kernel.shape
            # 畳み込み処理を端まで行うために、ラスターデータを拡張する
            _dst = self.expansion_dst(vertical=rows, horizontal=cols)
            ary = _dst.array()
            conved_ary = (
                scipy
                .ndimage
                .convolve(ary, kernel, mode='constant')
            )
            tpi_ary = ary - conved_ary
            # 端の部分を削除
            tpi_ary = tpi_ary[rows:-rows, cols:-cols]
        else:
            # 'kernel'が指定されていない場合は、DEMProcessingを使用
            options = {
                'destName': '',
                'srcDS': self.dataset,
                'processing': 'TPI',
                'format': 'MEM',
                'alg': kwargs.get('alg', 'Horn')
            }
            _new_dst = gdal.DEMProcessing(**options)
            tpi_ary = _new_dst.ReadAsArray()
            _new_dst = None
        outlier_treatment = kwargs.get('outlier_treatment', 1.5)
        if outlier_treatment:
            # 外れ値処理
            tpi_ary = self._outlier_treatment(
                ary=tpi_ary, 
                threshold=outlier_treatment
            )
        if kwargs.get('return_array', False):
            # return_arrayが指定されている場合は、numpy配列で返す
            return tpi_ary
        return self.write_ary_to_mem(tpi_ary)
    
    def _outlier_treatment(self, ary: np.ndarray, threshold: float) -> np.ndarray:
        """
        ## Summary
            外れ値処理を行う
        Args:
            ary(np.ndarray): ラスターデータ
            threshold(float): 外れ値処理の倍数
        Returns:
            np.ndarray: 外れ値処理後のラスターデータ
        """
        q1 = np.nanpercentile(ary, 25)
        q3 = np.nanpercentile(ary, 75)
        iqr = q3 - q1
        upper = q3 + threshold * iqr
        lower = q1 - threshold * iqr
        return np.where(upper < ary, upper, np.where(ary < lower, lower, ary))

    ############################################################################
    # --------------------- Methods for create kernels. ---------------------
    def mean_kernel_from_distance(self, 
        distance: float, 
        metre: bool=True
    ) -> np.ndarray:
        """
        ## Summary
            作成したい辺の長さを元に平均カーネルを作成する。
            作成したカーネルは、`scipy.ndimage.convolve`で使用する。
        Args:
            distance(int): カーネルの距離
            metre(bool, optional): メートル単位で指定するかどうか. Defaults to True.Falseの場合はそのままの値を使用する。
        Returns:
            np.ndarray: 平均カーネル
        """
        if metre:
            x_resol, y_resol = self.cell_size_in_metre(5)
        else:
            x_resol = self.x_resolution
            y_resol = self.y_resolution
        kernel_size = kernels.distance_to_kernel_size(distance, x_resol, y_resol)
        return kernels.mean_kernel(kernel_size.x, kernel_size.y)
    
    def doughnut_kernel_from_distance(self, 
        distance: float, 
        metre: bool=True
    ) -> np.ndarray:
        """
        ## Summary  
            作成したい辺の長さを元にドーナツカーネルを作成する。
            作成したカーネルは、`scipy.ndimage.convolve`で使用する。
        Args:
            distance(float): カーネルの距離
            metre(bool, optional): メートル単位で指定するかどうか. Defaults to True.Falseの場合はそのままの値を使用する。
        Returns:
            np.ndarray: ドーナツカーネル
        """
        if metre:
            x_resol, y_resol = self.cell_size_in_metre(5)
        else:
            x_resol = self.x_resolution
            y_resol = self.y_resolution
        kernel_size = kernels.distance_to_kernel_size(distance, x_resol, y_resol)
        return kernels.doughnut_kernel(kernel_size.x, kernel_size.y)

    def gaussian_kernel_from_distance(self,
        distance: float,
        metre: bool=True,
        coef: float=None
    ) -> np.ndarray:
        """
        ## Summary
            作成したい辺の長さを元にガウシアンカーネルを作成する。
            作成したカーネルは、`scipy.ndimage.convolve`で使用する。
        Args:
            distance(float): カーネルの距離
            metre(bool, optional): メートル単位で指定するかどうか. Defaults to True.Falseの場合はそのままの値を使用する。
            coef(float, optional): ガウシアンカーネルの係数. Defaults to None.
        Returns:
            np.ndarray: ガウシアンカーネル
        """
        if metre:
            x_resol, y_resol = self.cell_size_in_metre(5)
        else:
            x_resol = self.x_resolution
            y_resol = self.y_resolution
        kernel_size = kernels.distance_to_kernel_size(distance, x_resol, y_resol)
        return kernels.gaussian_kernel_from_size(kernel_size.x, kernel_size.y, coef)

    def inverse_gaussian_kernel_from_distance(self, 
        distance: float, 
        metre: bool=True,
        coef: float=None
    ) -> np.ndarray:
        """
        ## Summary
            作成したい辺の長さを元に逆ガウシアンカーネルを作成する。
            作成したカーネルは、`scipy.ndimage.convolve`で使用する。
        Args:
            distance(float): カーネルの距離
            metre(bool, optional): メートル単位で指定するかどうか. Defaults to True.Falseの場合はそのままの値を使用する。
            coef(float, optional): 逆ガウシアンカーネルの係数. Defaults to None.
        Returns:
            np.ndarray: 逆ガウシアンカーネル
        """
        if metre:
            x_resol, y_resol = self.cell_size_in_metre(5)
        else:
            x_resol = self.x_resolution
            y_resol = self.y_resolution
        kernel_size = kernels.distance_to_kernel_size(distance, x_resol, y_resol)
        return kernels.inverse_gaussian_kernel_from_size(kernel_size.x, kernel_size.y, coef)
    
    ############################################################################
    # ----------------- Methods for set colormap to 2D-Array. -----------------

    #############################################################################
    # ----------------- Methods for plotting. -----------------
    def plot_raster(self, 
        fig: Figure,
        ax: Axes,
        **kwargs
    ) -> None:
        """
        ## Summary
            ラスターデータをプロットする。
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
        """
        scope = self.bounds()
        extent = [scope.x_min, scope.x_max, scope.y_min, scope.y_max]

        if self.RasterCount == 1:
            # Bandが1つの場合は、NoDataをnanに変換
            img = self.array()
        else:
            # Bandが複数の場合は、matplotlibで表示できる様に配列形状を変更
            img = np.dstack(self.array())

        if kwargs.get('nodata') and self.RasterCount == 1:
            # NoDataを強調して表示
            nodata_ary = np.where(np.isnan(img), 255, 0)
            ax.imshow(nodata_ary, cmap='bwr', extent=extent)
            patches = [Patch(color='red', label="NoData")]
            nodata_label_anchor = kwargs.get('nodata_label_anchor', (1.25, 1.1))
            plt.legend(handles=patches, bbox_to_anchor=nodata_label_anchor)

        cmap = kwargs.get('cmap', 'terrain')
        cb = ax.imshow(img, cmap=cmap, extent=extent)
        colorbar = kwargs.get('colorbar', True)
        if colorbar and self.RasterCount == 1:
            # カラーバーを表示
            shrink = kwargs.get('shrink', 0.8)
            fig.colorbar(cb, shrink=shrink)



def gdal_open(file_path: Path) -> CustomGdalDataset:
    """
    ## Summary
        ラスターデータを開いて、`CustomGdalDataset`を返す。CustomGdalDatasetは、gdal.Datasetの拡張クラス。
    Args:
        file_path(Path): ラスターデータのパス
    Returns:
        CustomGdalDataset: gdal.Dataset の拡張クラス
    """
    dst = gdal.Open(file_path)
    new_dst = CustomGdalDataset(dst)
    dst = None
    return new_dst


if __name__ == '__main__':
    file = r"D:\Repositories\ProcessingRaster\datasets\test\DTM__R10__EPSG4326.tif"
    gdal_dst = gdal_open(file)
    import os
    
    