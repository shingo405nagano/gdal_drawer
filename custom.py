from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import NamedTuple
from typing import Tuple

import geopandas as gpd
import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
import shapely
gdal.UseExceptions()

from gdal_utils import GdalUtils
from utils.exceptions import custom_gdal_exception
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
class CenterCoordinates:
    """
    各セルの中心座標を格納するデータクラス
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
class CustomGdalDataset:
    def __init__(self, dataset):
        if not isinstance(dataset, gdal.Dataset):
            custom_gdal_exception.not_gdal_dataset_err()
        if dataset.GetProjection() == '':
            custom_gdal_exception.not_have_crs_err()
        self.dataset = self._copy_dataset(dataset)

    def __getattr__(self, module_name):
        return getattr(self.dataset, module_name)
    
    @staticmethod
    def __check_crs(func):
        def wrapper(self, out_wkt_crs=None, *args, **kwargs):
            if out_wkt_crs is None:
                return func(self, *args, **kwargs)
            try:
                _ = pyproj.CRS(out_wkt_crs)
            except:
                custom_gdal_exception.unknown_crs_err()
            return func(self, out_wkt_crs, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def __check_datum(func):
        def wrapper(self, *args, **kwargs):
            datum_name = kwargs.get('datum_name', 'JGD2011')
            try:
                _ = pyproj.CRS(datum_name).to_authority()
            except pyproj.exceptions.CRSError:
                custom_gdal_exception.unknown_datum_err()
            return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def __is_iterable_of_ints(func):
        def wrapper(self, obj, *args, **kwargs):
            if isinstance(obj, Iterable):
                if all(isinstance(item, int) for item in obj):
                    return func(self, obj, *args, **kwargs)
            custom_gdal_exception.get_band_number_err()
        return wrapper

    @staticmethod
    def __pyproj_to_wkt_crs(func):
        def wrapper(self, obj=None, *args, **kwargs):
            if obj is None:
                return func(self, obj, *args, **kwargs)
            elif isinstance(obj, pyproj.CRS):
                return func(self, obj.to_wkt(), *args, **kwargs)
            return func(self, obj, *args, **kwargs)
        return wrapper

    @staticmethod
    def __wkt_geometry_check(func: Callable) -> str:
    # ジオメトリがWKT形式であるかチェックする。shapely.geometryだった場合はWKT形式に変換する
        def wrapper(self, *args, **kwargs):
            args_ = False
            if 'wkt_poly' not in kwargs.keys():
                poly = args[0]
                args_ = True
            else:
                poly = kwargs.get('wkt_poly')
            if isinstance(poly, str):
                try:
                    shapely.from_wkt(poly)
                except shapely.errors.WKTReadingError:
                    custom_gdal_exception.load_wkt_geometry_err()
                else:
                    return func(self, *args, **kwargs)
            elif isinstance(poly, shapely.geometry.base.BaseGeometry):
                if args_:
                    lst = list(args)
                    lst[2] = poly.wkt
                    args = tuple(lst)
                else:
                    kwargs['wkt_poly'] = poly.wkt
                return func(self, *args, **kwargs)
            else:
                custom_gdal_exception.load_wkt_geometry_err()
        return wrapper

    @property
    def x_resolution(self):
        """
        X方向の解像度を取得する
        Returns:
            (float): X方向の解像度
        """
        return self.GetGeoTransform()[1]
    
    @property
    def y_resolution(self):
        """
        Y方向の解像度を取得する
        Returns:
            (float): Y方向の解像度
        """
        return self.GetGeoTransform()[-1]

    def array(self, band_numbers: int | Iterable[int]=None) -> np.ndarray:
        """
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

    @__is_iterable_of_ints
    def _get_selected_arys(self, band_nums: List[int]) -> np.ndarray:
        """
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
    def copy_dataset(self) -> gdal.Dataset:
        """
        `gdal.Dataset`のコピーを作成する。
        Returns:
            (gdal.Dataset):
        Examples:
            >>> new_dst: gdal.Dataset = dst.copy_dataset()
        """
        driver = gdal.GetDriverByName('MEM')
        new_dst = driver.CreateCopy('', self.dataset)
        return CustomGdalDataset(new_dst)
    
    def _copy_dataset(self, dst: gdal.Dataset) -> gdal.Dataset:
        """
        `gdal.Dataset`のコピーを作成する。
        Args:
            dst(gdal.Dataset): コピー先のDataset
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
        out_nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        メモリ上に新しい配列を書き込んだ`gdal.Dataset`を作成する。
        この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。※配列のshapeはオリジナルの`gdal.Dataset`と同じである必要がある。
        Args:
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. 
                [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            nodata(Any): NoData. Defaults to np.nan
        Returns:
            gdal.Dataset
        Examples:
            >>> file_path = r'.\\raster.tif''
            >>> dst = CustomGdalDataset(file_path)
            >>> ary = np.random.rand(dst.RasterYSize, dst.RasterXSize)
            >>> new_dst: gdal.Dataset = dst.write_ary_to_mem(ary)
        """
        if self.RasterCount == 1:
            return self._single_band_to_mem(ary, data_type, out_nodata)
        else:
            return self._multi_band_to_mem(ary, data_type, out_nodata)

    def _single_band_to_mem(self, 
        ary: np.ndarray, 
        data_type: int=gdal.GDT_Float32, 
        out_nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。※配列のshapeはオリジナルの`gdal.Dataset`と同じである必要がある。
        Args:
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. 
                [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            out_nodata(Any): NoData. Defaults to np.nan
        Returns:
            gdal.Dataset
        """
        if ary.shape != self.__original_ary_shape:
            # ラスターサイズと配列のサイズが一致しない場合はエラーを発生させる
            custom_gdal_exception.shape_err(self.__original_ary_shape, ary.shape)
        # メモリ上に新しいラスターデータを作成する
        driver = gdal.GetDriverByName('MEM')
        driver.Register()
        new_dst = self.__create_dataset(data_type)
        band = new_dst.GetRasterBand(1)
        ary = nodata_to(ary, band.GetNoDataValue(), out_nodata)
        band.WriteArray(ary)
        band.SetNoDataValue(out_nodata)
        return CustomGdalDataset(new_dst)

    def _multi_band_to_mem(self,
        ary: np.ndarray, 
        data_type: int=gdal.GDT_Float32, 
        out_nodata: Any=np.nan
    ) -> gdal.Dataset:
        """
        この関数はオリジナルの`gdal.Dataset`のメタデータを引き継ぎ、新たな配列を書き込んだ
        新しい`gdal.Dataset`を作成する。※配列のshapeはオリジナルの`gdal.Dataset`と同じである必要がある。
        Args:
            ary(np.ndarray): ラスターデータの配列
            data_type(int): データ型. Defaults to gdal.GDT_Float32. 
                [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]
            out_nodata(Any): NoData. Defaults to np.nan
        Returns:
            gdal.Dataset
        """
        if ary.shape != self.__original_ary_shape:
            # ラスターサイズと配列のサイズが一致しない場合はエラーを発生させる
            custom_gdal_exception.shape_err(self.__original_ary_shape, ary.shape)
        # メモリ上に新しいラスターデータを作成する
        new_dst = self.__create_dataset(data_type)
        for i in range(self.RasterCount):
            band = new_dst.GetRasterBand(i+1)
            band.WriteArray(ary[i])
            band.SetNoDataValue(out_nodata)
        return CustomGdalDataset(new_dst)
    
    def __create_dataset(self, data_type: int) -> gdal.Driver:
        """
        メモリ上に新しい`gdal.Dataset`を作成する。
        Args:
            data_type(int): データ型
        Returns:
            (gdal.Dataset): 新しい`gdal.Dataset`
        """
        driver = gdal.GetDriverByName('MEM')
        driver.Register()
        dst = driver.Create(
            '',
            xsize=self.RasterXSize,
            ysize=self.RasterYSize,
            bands=self.RasterCount,
            eType=data_type
        )
        dst.SetGeoTransform(self.GetGeoTransform())
        dst.SetProjection(self.GetProjection())
        return dst

    @property
    def __original_ary_shape(self) -> tuple:
        """
        オリジナルの`gdal.Dataset`に記録されている配列の形状を取得する
        Returns:
            (tuple): (rows, cols) or (bands, rows, cols)
        """
        if 1 < self.RasterCount:
            return (self.RasterCount, self.RasterYSize, self.RasterXSize)
        return (self.RasterYSize, self.RasterXSize)

    def fill_nodata(self, max_search_distance: int, smoothing: int=10) -> gdal.Dataset:
        if self.RasterCount == 1:
            return self._fill_nodata_of_single_band(max_search_distance, smoothing)
        return self._fill_nodata_of_multi_band(max_search_distance, smoothing)
    
    def _fill_nodata_of_single_band(self,
        max_search_distance: int,
        smoothing: int
    ) -> gdal.Dataset:
        """
        SingleBandのNoDataを埋める
        Args:
            max_search_distance(int): 最大探索距離
            smoothing(int, optional): スムージングの回数. Defaults to 10.
        Returns:
            gdal.Dataset
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
    ) -> gdal.Dataset:
        """
        MultiBandの`gdal.Dataset`のNoDataを埋める
        ※ MultiBandの場合は int型を使用しているのでnp.nanは使用できない。修正が必要？
        Args:
            max_search_distance(int): 最大探索距離
            smoothing(int, optional): スムージングの回数. Defaults to 10.
        Returns:
            gdal.Dataset
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

    ##########################################################################
    # -------------- Methods for calculation data coordinates. --------------
    def bounds(self) -> Bounds:
        """
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
    
    @__pyproj_to_wkt_crs
    @__check_crs
    def reprojected_bounds(self, out_wkt_crs: str) -> Bounds:
        """
        投影変換した後の範囲を取得する。
        Args:
            out_wkt_crs(str): 出力先のWKT-CRS
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
            in_crs, out_wkt_crs
        )
        return Bounds(xs[0], ys[0], xs[1], ys[1])
    
    @__pyproj_to_wkt_crs
    @__check_crs
    def center(self, out_wkt_crs: str=None) -> XY:
        """
        `gdal.Dataset`の中心座標を取得する。
        Args:
            kwargs:
                - out_wkt_crs(str): 出力先のWKT-CRS
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
        if not out_wkt_crs is None:
            return gdal_utils.reproject_xy(x, y, self.GetProjection(), out_wkt_crs)
        return XY(x, y)

    def cells_center_coordinates(self) -> CenterCoordinates:
        """
        各セルの中心座標を計算。戻り値はX座標とY座標の2次元配列であり、各数値がセルの中心座標を示す。
        Returns:
            CenterCoordinates(dataclass):
                X(np.ndarray): X座標の2次元配列
                Y(np.ndarray): Y座標の2次元配列
        Examples:
            >>> center_coords = dst.cells_center_coordinates
            >>> X: np.ndarray = center_coords.X
            >>> Y: np.ndarray = center_coords.Y
            >>> points = [(x, y) for x, y in zip(X.ravel(), Y.ravel())] #参照配列
            or ...
            >>> points = [(x, y) for x, y in zip(X.flatten(), Y.flatten())] #新規配列
        """
        transform = self.GetGeoTransform()
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
        return CenterCoordinates(*np.meshgrid(X, Y))
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        RasterDataのセル値をshapely.PointにしてGeoDataFrameに入力。バンド数に応じて列数が増える。
        Args:
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
        # セルの中心座標を取得
        centers = self.cells_center_coordinates()
        data = {
            'x': centers.X.flatten(),
            'y': centers.Y.flatten(),
        }
        # 各バンドの値を取得
        band = self.GetRasterBand(1)
        ary = nodata_to(self.ReadAsArray(), band.GetNoDataValue(), np.nan)
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

    def to_pandas(self, digit=9) -> pd.DataFrame:
        """
        RasterDataのセル値をshapely.PointにしてDataFrameに入力。これは強制的にEPSG:4326に投影変換する。
        Args:
            digit(int): 小数点以下の桁数
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
        gdf = self.to_geodataframe()
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

    def cell_size_in_metre(self, digit=4):
        """
        セルサイズをMetreで取得する
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
    
    def cell_size_in_degree(self, digit=9):
        """
        セルサイズをDegreeで取得する
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
    @__pyproj_to_wkt_crs
    @__check_crs
    def reprojected_dataset(self, out_wkt_crs: str) -> gdal.Dataset:
        """
        'gdal.Dataset'の投影変換
        Args:
            out_crs(str): 出力のWKT-CRS
        Returns:
            (gdal.Dataset): 投影変換後のラスターデータ
        Examples:
            >>> dst = CustomGdalDataset(file_path) #EPSG:4326
            >>> out_crs = pyproj.CRS(6690).to_wkt()
            >>> reprojected_dataset: gdal.Dataset = dst.reprojected_dataset(out_crs)
        """
        # 投影変換の為のオプションを設定
        ops = gdal.WarpOptions(
            format='MEM',
            srcSRS=self.GetProjection(),
            dstSRS=out_wkt_crs,
            outputBounds=self.reprojected_bounds(out_wkt_crs),
            width=self.RasterXSize,
            height=self.RasterYSize,
            resampleAlg=gdal.GRA_CubicSpline
        )
        dst = gdal.Warp('', self.dataset, options=ops)
        return CustomGdalDataset(dst)

    def estimate_utm_and_reprojected_dataset(self,
        datum_name: str='JGD2011'
    ) -> gdal.Dataset:
        """
        UTMのCRSを推定して投影変換を行う。日本で使用する場合は "datum_name='JGD2011'" を指定する。
        Args:
            dst(gdal.Dataset): ラスターデータ
            datum_name: 'WGS 84', 'JGD2011' など
        Returns:
            (gdal.Dataset): 投影変換後のラスターデータ
        Examples:
            >>> new_dst: gdal.Dataset = dst.estimate_utm_and_reprojected_dataset('JGD2011')
        """
        out_wkt_crs = self.estimate_utm_crs(datum_name)
        return self.reprojected_dataset(out_wkt_crs)

    ############################################################################
    # ----------------- Methods for resampling dataset. -----------------
    def _option_template_with_resol_spec(self,
        x_resolution: float,
        y_resolution: float,
        resample_algorithm: int=gdal.GRA_CubicSpline
    ) -> gdal.WarpOptions:
        """
        ## Summary
        分解能を指定するためのオプションテンプレートを作成する
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

    def _option_temppate_with_cells_spec(self,
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
    ) -> gdal.Dataset:
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
            (gdal.Dataset): リサンプリング後のラスターデータ
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
        ops = self._option_template_with_resol_spec(
            x_resolution, y_resolution, resample_algorithm)
        return CustomGdalDataset(gdal.Warp('', self.dataset, options=ops))

    def resample_with_cells_spec(self,
        x_cells: int,
        y_cells: int,
        resample_algorithm: int=gdal.GRA_CubicSpline
    ) -> gdal.Dataset:
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
            (gdal.Dataset): リサンプリング後のラスターデータ
        Examples:
            >>> new_dst: gdal.Dataset = dst.resample_with_cells(100, 100)
        """
        ops = self._option_temppate_with_cells_spec(
            x_cells, y_cells, resample_algorithm)
        return CustomGdalDataset(gdal.Warp('', self.dataset, options=ops))

    ############################################################################
    # ----------------- Methods for clipping dataset. -----------------
    @__wkt_geometry_check
    def _option_template_with_wkt_poly_spec(self,
        wkt_poly: str,
        fmt: str='MEM',
        nodata: Any=np.nan,
        **kwargs
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
            srcNodata=dst.GetRasterBand(1).GetNoDataValue()
        )
    
    def clip_by_wkt_poly(self, 
        wkt_poly: str | shapely.Polygon, 
        nodata: Any=np.nan,
        **kwargs
    ) -> gdal.Dataset:
        """
        ## Summary
        ポリゴンでラスターデータをクリップする。このメソッドは、ポリゴンの投影法が異なる場合にも使用できる。
        Args:
            wkt_poly(str): WKT形式のポリゴン
            nodata(Any, optional): NoData. Defaults to np.nan
            kwargs:
                poly_crs(str): ポリゴンの投影法. Defaults to None. これを指定すれば Raster と異なる投影法のポリゴンを使用できる。
        Returns:
            gdal.Dataset: クリップ後のラスターデータ
        Examples:
            >>> wkt_poly = 'POLYGON ((x1 y1, x2 y2, x3 y3, x4 y4, x1 y1))'
            >>> new_dst: gdal.Dataset = dst.clip_by_wkt_poly(wkt_poly)
        """
        # ポリゴンの投影法が異なり、かつ指定されている場合は、ポリゴンを投影変換する
        options = self._option_template_with_wkt_poly_spec(
            wkt_poly, 
            nodata=nodata, 
            poly_crs=kwargs.get('poly_crs', self.GetProjection())
        )
        return CustomGdalDataset(gdal.Warp('', self.dataset, options=options))
    
    def clip_by_bounds(self, 
        wkt_poly: str | shapely.Polygon,
        nodata: Any=np.nan,
        **kwargs
    ) -> gdal.Dataset:
        """
        ## Summary
        Polygonのバウンディングボックスでラスターデータをクリップする。このメソッドは、ポリゴンの投影法が異なる場合にも使用できる。
        Args:
            wkt_poly(str): WKT形式のポリゴン
            nodata(Any, optional): NoData. Defaults to np.nan
            kwargs:
                poly_crs(str): ポリゴンの投影法. Defaults to None. これを指定すれば Raster と異なる投影法のポリゴンを使用できる。
        Returns:
            gdal.Dataset: クリップ後のラスターデータ
        """
        if isinstance(wkt_poly, str):
            wkt_poly = shapely.from_wkt(wkt_poly).envelope.wkt
        elif isinstance(wkt_poly, shapely.geometry.base.BaseGeometry):
            wkt_poly = wkt_poly.envelope.wkt
        else:
            custom_gdal_exception.load_wkt_geometry_err()
        poly_crs = kwargs.get('poly_crs', self.GetProjection())
        return self.clip_by_wkt_poly(wkt_poly, nodata, poly_crs=poly_crs)

    def clip_by_fit_bounds(self, 
        wkt_poly: str | shapely.Polygon, 
        nodata: Any=np.nan, 
        **kwargs
    ) -> gdal.Dataset:
        """
        ## Summary
        ポリゴンの最小外接矩形でラスターデータをクリップする。このメソッドは、ポリゴンの投影法が異なる場合にも使用できる。
        Args:
            wkt_poly(str): WKT形式のポリゴン
            nodata(Any, optional): NoData. Defaults to np.nan
            kwargs:
                poly_crs(str): ポリゴンの投影法. Defaults to None. これを指定すれば Raster と異なる投影法のポリゴンを使用できる。
        Returns:
            gdal.Dataset: クリップ後のラスターデータ
        """
        if isinstance(wkt_poly, str):
            wkt_poly = shapely.from_wkt(wkt_poly).minimum_rotated_rectangle.wkt
        elif isinstance(wkt_poly, shapely.geometry.base.BaseGeometry):
            wkt_poly = wkt_poly.minimum_rotated_rectangle.wkt
        else:
            custom_gdal_exception.load_wkt_geometry_err()
        poly_crs = kwargs.get('poly_crs', self.GetProjection())
        return self.clip_by_wkt_poly(wkt_poly, nodata, poly_crs=poly_crs)

    #############################################################################
    # ----------------- Methods for obtaining cells statistics. -----------------
    def plot_raster(self, 
        fig: Any,
        ax: Any, 
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
        """
        scope = self.dataset_bounds(dst)
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

