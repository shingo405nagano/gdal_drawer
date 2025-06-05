"""
python gdal has difficulty controlling memory.
Methods that cannot free memory are
    1. ``concurrent.futures.ProcessPoolExecutor``
    2. ``concurrent.futures.ThreadPoolExecutor``
This module was created because it is necessary to use one of the following.
"""

import gc
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Union

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

from gdal_drawer.utils.config import CRS, crs_checker
from gdal_drawer.utils.raster_info import (
    crs_unit_name,
    estimate_utm_crs_from_datasets,
    get_bounds,
    get_reprojected_bounds,
    resolution_from_dataset,
)


@dataclass
class TempFileDataset:
    temp_file: tempfile.NamedTemporaryFile
    dataset: gdal.Dataset

    def close(self):
        """
        ## Summary:
            Close and delete the temporary file.
        ## Examples:
            >>> temp_dataset = TempDataset(temp_file, dataset)
            >>> temp_dataset.close()
        """
        self.dataset = None
        self.temp_file.close()
        os.unlink(self.temp_file.name)
        gc.collect()


def copy_temppfile(
    dst: gdal.Dataset | str, prefix: str = "GDAL_CUSTOM_", **kwargs
) -> TempFileDataset:
    """
    ## Summary:
        Create a temporary file and copy the gdal.Dataset to it.
    Args:
        dst (gdal.Dataset): gdal.Dataset
        prefix (str): Prefix for the temporary file name.
        kwargs(dict):
            - xsize (int): Width of the dataset. Default is dst.RasterXSize.
            - ysize (int): Height of the dataset. Default is dst.RasterYSize.
            - count (int): Number of bands in the dataset. Default is dst.RasterCount.
            - eType (int): Data type of the dataset. Default is dst.GetRasterBand(1).DataType.
            - projection (str): Projection of the dataset. Default is dst.GetProjection().
            - transform (list): GeoTransform of the dataset. Default is dst.GetGeoTransform().
    Returns:
        TempDataset(dataclass): Temporary file and gdal.Dataset
            - temp_file(tempfile.NamedTemporaryFile)
            - dataset(gdal.Dataset)
    Examples:
        >>> temp_dataset = copy_temppfile(dst)
        >>> temp_dataset.close()
    """
    driver = gdal.GetDriverByName("GTiff")
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".tif", delete=False)
    options = {
        "xsize": kwargs.get("xsize", dst.RasterXSize),
        "ysize": kwargs.get("ysize", dst.RasterYSize),
        "bands": kwargs.get("bands", dst.RasterCount),
        "eType": kwargs.get("eType", dst.GetRasterBand(1).DataType),
    }
    temp_dst = driver.Create(temp_file.name, **options)
    temp_dst.SetProjection(kwargs.get("projection", dst.GetProjection()))
    temp_dst.SetGeoTransform(kwargs.get("transform", dst.GetGeoTransform()))
    for i in range(dst.RasterCount):
        band = dst.GetRasterBand(i + 1)
        temp_band = temp_dst.GetRasterBand(i + 1)
        temp_band.SetNoDataValue(band.GetNoDataValue())
        data = band.ReadAsArray()
        temp_band.WriteArray(data)
    temp_dst.FlushCache()
    return TempFileDataset(temp_file, temp_dst)


def copy_memory(dst: gdal.Dataset, **kwargs) -> gdal.Dataset:
    """
    ## Summary:
        Create a copy of the gdal.Dataset in memory.
    Args:
        dst (gdal.Dataset): gdal.Dataset
    Returns:
        gdal.Dataset: gdal.Dataset in memory
    """
    driver = gdal.GetDriverByName("MEM")
    options = {
        "xsize": kwargs.get("xsize", dst.RasterXSize),
        "ysize": kwargs.get("ysize", dst.RasterYSize),
        "bands": kwargs.get("bands", dst.RasterCount),
        "eType": kwargs.get("eType", dst.GetRasterBand(1).DataType),
    }
    new_dst = driver.Create("", **options)
    new_dst.SetProjection(kwargs.get("projection", dst.GetProjection()))
    new_dst.SetGeoTransform(kwargs.get("transform", dst.GetGeoTransform()))
    for i in range(dst.RasterCount):
        band = dst.GetRasterBand(i + 1)
        new_band = new_dst.GetRasterBand(i + 1)
        new_band.SetNoDataValue(band.GetNoDataValue())
        data = band.ReadAsArray()
        new_band.WriteArray(data)
    new_dst.FlushCache()
    return new_dst


def nodata_to_nan(dst: gdal.Dataset) -> gdal.Dataset:
    """
    ## Summary:
        Convert NoData values to NaN in a gdal.Dataset. Value type must be float.
    Args:
        dst (gdal.Dataset): gdal.Dataset
    Returns:
        gdal.Dataset: gdal.Dataset with NoData values converted to NaN
    Examples:
        >>> dst = gdal.Open(file_path)
        >>> dst = nodata_to_nan(dst)
        >>> band = dst.GetRasterBand(1)
        >>> print(band.GetNoDataValue())
        np.nan
    """
    for i in range(dst.RasterCount):
        band = dst.GetRasterBand(i + 1)
        nodata_value = band.GetNoDataValue()
        data_type = gdal.GetDataTypeName(band.DataType)
        if data_type not in ["Float32", "Float64"]:
            # Skip if the data type is not float
            continue

        ary = band.ReadAsArray()
        ary = np.where(ary == nodata_value, np.nan, ary)
        ary = np.where(np.isinf(ary), np.nan, ary)
        band.WriteArray(ary)
        band.SetNoDataValue(np.nan)
    return dst


def fill_nodata(
    dst: gdal.Dataset,
    nodata_nan: bool = True,
    max_search: int = 10,
    smoothing: int = 1,
    return_array: bool = True,
    release_memory: bool = True,
) -> Union[gdal.Dataset, np.array]:
    """
    ## Summary:
        Fill in the value of NoData using the surrounding values.
        Use ``ThreadPoolExecutor`` to prevent memory leaks.
        Note that if return_array is False, memory cannot be freed.
    Args:
        dst (gdal.Dataset):
            gdal.Dataset
        nodata_nan (bool):
            Convert NoData values to NaN. Default is True.
        max_search (int):
            Maximum search distance for filling NoData values. Default is 10.
        smoothing (int):
            Number of smoothing iterations. Default is 1.
        return_array (bool):
            Return a numpy array instead of a gdal.Dataset. Default is True.
            Note that it is harder to control memory usage in the case of gdal.Dataset.
        release_memory (bool):
            Whether to free Dataset memory before processing. Default is True.
    Returns:
        gdal.Dataset:
            gdal.Dataset with NoData values filled in.
        np.array:
            Numpy array with NoData values filled in if return_array is True.
    Examples:
        >>> dst = gdal.Open(file_path)
        >>> filled_dst = fill_nodata(dst, nodata_nan=True, max_search=10, smoothing=1, return_array=False)
        >>> print(type(filled_dst))
        <class 'osgeo.gdal.Dataset'>
        >>> filled_array = fill_nodata(dst, nodata_nan=True, max_search=10, smoothing=1, return_array=True)
        >>> print(type(filled_array))
        <class 'numpy.ndarray'>
    """
    file_path = dst.GetDescription()

    if release_memory:
        # Release memory before processing
        dst = None

    if return_array:
        with ProcessPoolExecutor() as executor:
            future = executor.submit(
                _fill_nodata, file_path, nodata_nan, max_search, smoothing, return_array
            )
        result = future.result()
    else:
        result = _fill_nodata(file_path, nodata_nan, max_search, smoothing, return_array)
    return result


def _fill_nodata(
    file_path: str,
    nodata_nan: bool = True,
    max_search: int = 10,
    smoothing: int = 1,
    return_array: bool = True,
) -> Union[gdal.Dataset, np.array]:
    """
    ## Summary:
        Fill in the value of NoData using the surrounding values.
    Args:
        file_path (str):
            Path to the raster file.
        nodata_nan (bool):
            Convert NoData values to NaN. Default is True.
        max_search (int):
            Maximum search distance for filling NoData values. Default is 10.
        smoothing (int):
            Number of smoothing iterations. Default is 1.
        return_array (bool):
            Return a numpy array instead of a gdal.Dataset. Default is True.
            Note that it is harder to control memory usage in the case of gdal.Dataset.
    Returns:
        gdal.Dataset:
            gdal.Dataset with NoData values filled in.
        np.array:
            Numpy array with NoData values filled in if return_array is True.
    """
    # Create a temporary file to write the filled data.
    dst = gdal.Open(file_path)
    write_temp_dataset = copy_temppfile(dst)
    if nodata_nan:
        # Convert NoData values to NaN in the dataset.
        write_temp_dataset.dataset = nodata_to_nan(write_temp_dataset.dataset)
    # Create a temporary file to use as a mask.
    mask_temp_dataset = copy_temppfile(write_temp_dataset.dataset)
    for i in range(dst.RasterCount):
        # Fill in the NoData values for each band.
        write_band = write_temp_dataset.dataset.GetRasterBand(i + 1)
        mask_band = mask_temp_dataset.dataset.GetRasterBand(i + 1)
        gdal.FillNodata(
            write_band,
            mask_band,
            maxSearchDist=max_search,
            smoothingIterations=smoothing,
        )
    write_temp_dataset.dataset.FlushCache()
    mask_temp_dataset.close()
    if return_array:
        # Read the filled data into a numpy array.
        result_array = write_temp_dataset.dataset.ReadAsArray()
        write_temp_dataset.close()
        return result_array
    return write_temp_dataset.dataset


def expansion_dst(
    dst: gdal.Dataset,
    vertical_cells: int,
    horizontal_cells: int,
    datum_name: str = "JGD2011",
    memory: bool = False,
    return_array: bool = True,
) -> Union[gdal.Dataset, np.array]:
    """
    ## Summary:
        Expand the size of a gdal.Dataset by adding cells to the top, bottom, left, and right.
    Args:
        dst (gdal.Dataset):
            gdal.Dataset to be expanded.
        vertical_cells (int):
            Number of cells to add vertically.
        horizontal_cells (int):
            Number of cells to add horizontally.
        datum_name (str):
            Name of the datum. Default is 'JGD2011'.
        memory (bool):
            Whether to use memory for the new dataset. Default is False.
        return_array (bool):
            Return a numpy array instead of a gdal.Dataset. Default is True.
    Returns:
        gdal.Dataset or np.array:
            Expanded data of gdal.Dataset or numpy array.
    """
    # Calculate the resolution of the dataset
    new_transform = _calc_expanded_transform(
        dst, vertical_cells, horizontal_cells, datum_name=datum_name
    )
    # Create a new array with the expanded size
    expanded_ary = _get_expanded_array(dst, vertical_cells, horizontal_cells)
    # Set the NoData value to NaN
    if return_array:
        return expanded_ary
    # Create a new dataset with the expanded size
    options = {
        "xsize": expanded_ary.shape[1],
        "ysize": expanded_ary.shape[0],
        "bands": dst.RasterCount,
        "eType": dst.GetRasterBand(1).DataType,
    }
    if memory:
        new_dst = copy_memory(dst, **options)
    else:
        temp_file = copy_temppfile(dst, **options)
        new_dst = temp_file.dataset
    # Set the new transform and write the array to the new dataset
    new_dst.SetGeoTransform(new_transform)
    new_dst.GetRasterBand(1).WriteArray(expanded_ary)
    new_dst.FlushCache()
    return new_dst


def _calc_expanded_transform(
    dst: gdal.Dataset,
    vertical_cells: int,
    horizontal_cells: int,
    datum_name: str = "JGD2011",
) -> list:
    unit_name = crs_unit_name(dst.GetProjection())
    digit = 3 if unit_name == "metre" else 9
    resolution = resolution_from_dataset(
        dst, unit=unit_name, digit=digit, datum_name=datum_name
    )
    # Calculate the expanded size of the dataset
    x_direction_expantion = resolution.x * horizontal_cells
    y_direction_expantion = resolution.y * vertical_cells
    dst_transform = list(dst.GetGeoTransform())
    dst_transform[0] -= x_direction_expantion
    dst_transform[3] += y_direction_expantion
    return dst_transform


def _get_expanded_array(
    dst: gdal.Dataset, vertical_cells: int, horizontal_cells: int
) -> np.array:
    # 配列の次元数分繰り返す処理に変更したい
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    org_ary = dst.ReadAsArray()
    upper_ary = org_ary[0, :]
    upper_ary = np.hstack(
        [
            np.zeros(horizontal_cells) + upper_ary[0],
            upper_ary,
            np.zeros(horizontal_cells) + upper_ary[-1],
        ]
    )
    upper_ary = np.vstack([upper_ary for _ in range(vertical_cells)])
    lower_ary = org_ary[-1, :]
    lower_ary = np.hstack(
        [
            np.zeros(horizontal_cells) + lower_ary[0],
            lower_ary,
            np.zeros(horizontal_cells) + lower_ary[-1],
        ]
    )
    lower_ary = np.vstack([lower_ary for _ in range(vertical_cells)])
    left_ary = np.hstack([org_ary[:, :1] for _ in range(horizontal_cells)])
    right_ary = np.hstack([org_ary[:, -1:] for _ in range(horizontal_cells)])
    result_ary = np.hstack([left_ary, org_ary, right_ary])
    result_ary = np.vstack([upper_ary, result_ary, lower_ary])
    return result_ary


crs_checker(index=0, kward="out_crs")


def reprojected_dataset(
    dst: gdal.Dataset,
    out_crs: CRS,
    reversible: bool = True,
    algorithm: int = gdal.GRA_CubicSpline,
) -> gdal.Dataset:
    """
    ## Summary:
        Projected return of Dataset.
    Args:
        dst (gdal.Dataset):
            gdal.Dataset to be reprojected.
        out_crs (CRS):
            Output CRS. CRS is a pyproj.CRS object or EPSG code.
        reversible (bool):
            Whether the projection is reversible. Default is True.
        algorithm (int):
            1: gdal.GRA_Bilinear
            2: gdal.GRA_Cubic
            3: gdal.GRA_CubicSpline
    Returns:
        gdal.Dataset:
            Projected gdal.Dataset.
    """
    ops = dict(
        format="MEM",
        srcSRS=dst.GetProjection(),
        dstSRS=out_crs,
        width=dst.RasterXSize,
        height=dst.RasterYSize,
        resampleAlg=algorithm,
    )
    if reversible:
        ops["outputBounds"] = get_reprojected_bounds(dst, out_crs)
    new_dst = gdal.Warp("", dst, options=gdal.WarpOptions(**ops))
    return new_dst


def estimate_utm_and_reprojected_dataset(
    dst: gdal.Dataset,
    datum_name: str = "JGD2011",
    reversible: bool = True,
    algorithm: int = gdal.GRA_CubicSpline,
) -> gdal.Dataset:
    """
    ## Summary:
        Estimate the UTM CRS from the dataset and reproject it.
    Args:
        dst (gdal.Dataset):
            gdal.Dataset to be reprojected.
        datum_name (str):
            Name of the datum. Default is 'JGD2011'.
        reversible (bool):
            Whether the projection is reversible. Default is True.
        algorithm (int):
            1: gdal.GRA_Bilinear
            2: gdal.GRA_Cubic
            3: gdal.GRA_CubicSpline
    Returns:
        gdal.Dataset:
            Reprojected gdal.Dataset.
    """
    utm_crs = estimate_utm_crs_from_datasets(dst, datum_name=datum_name)
    return reprojected_dataset(dst, utm_crs, reversible=reversible, algorithm=algorithm)


def _resample(
    dst: gdal.Dataset,
    resample_algorithm: int = gdal.GRA_CubicSpline,
    reversible: bool = True,
    out_crs: CRS = None,
    return_array: bool = True,
    **kwargs,
) -> gdal.Dataset:
    """
    ## Summary:
        Function to resample gdal.Dataset.
    Args:
        dst(gdal.Dataset):
            gdal.Dataset to be resampled.
        resample_algorithm(int):
            Resampling algorithm to be used.
            1: gdal.GRA_Bilinear
            2: gdal.GRA_Cubic
            3: gdal.GRA_CubicSpline
        reversible(bool):
            Whether to specify the coordinates after resampling.
        out_crs(int | str | pyproj.CRS):
            Output CRS. If None, the UTM CRS will be estimated from the dataset.
        return_array(bool):
            If True, return a numpy array instead of a gdal.Dataset. Default is True.
        kwargs:
            - x_resolution(float): X resolution of the resampled dataset.
            - y_resolution(float): Y resolution of the resampled dataset.
            - width(int): Width of the resampled dataset.
            - height(int): Height of the resampled dataset.
            - forced_metre_system(bool): If True, the output CRS will be in metres. Default is False.
            - datum_name(str): Name of the datum. Default is 'JGD2011'.
    """

    @crs_checker(0, "crs")
    def crs_check(crs: CRS) -> bool:
        return crs

    if out_crs is None:
        forced_metre_system = kwargs.get("forced_metre_system", False)
        datum_name = kwargs.get("datum_name", "JGD2011")
        if forced_metre_system:
            # If forced_metre_system is True, estimate UTM CRS
            out_crs = estimate_utm_crs_from_datasets(dst, datum_name=datum_name)
    elif out_crs is not None:
        # If out_crs is specified, check if it is a valid CRS
        out_crs = crs_check(out_crs)

    # Make the output options
    ops = dict(
        format="MEM",
        resampleAlg=resample_algorithm,
    )
    if kwargs.get("x_resolution") is not None:
        # To specify the resolution after resample.
        ops["xRes"] = kwargs["x_resolution"]
        ops["yRes"] = kwargs.get("y_resolution", kwargs["x_resolution"])
    elif (kwargs.get("width") is not None) and (kwargs.get("height") is not None):
        # To specify the width and height after resample.
        ops["width"] = kwargs["width"]
        ops["height"] = kwargs["height"]
    else:
        raise ValueError(
            "Either 'x_resolution' and 'y_resolution' or 'width' and 'height' must be specified."
        )
    if out_crs is not None:
        # When performing a projective transformation, specify the coordinate system of the output
        ops["srcSRS"] = dst.GetProjection()
        ops["dstSRS"] = out_crs.to_wkt()
    if reversible:
        # If reversible is True, specify the output bounds
        if out_crs is None:
            # If no projection transformation is performed.
            ops["outputBounds"] = get_bounds(dst)
        else:
            ops["outputBounds"] = get_reprojected_bounds(dst, out_crs)
    ops = gdal.WarpOptions(**ops)
    if return_array:
        return None
    return gdal.Warp("", dst, options=ops)


def resample_with_resolution_spec(
    dst: gdal.Dataset,
    x_resolution: float,
    y_resolution: float = None,
    resample_algorithm: int = gdal.GRA_CubicSpline,
    reversible: bool = True,
    out_crs: CRS = None,
    return_array: bool = True,
    forced_metre_system: bool = False,
    datum_name: str = "WGS 84",
) -> gdal.Dataset:
    """
    ## Summary:
        地上分解能を指定して`gdal.Dataset`のリサンプリングを行う関数。
    Args:
        dst (gdal.Dataset):
            リサンプル対象の`gdal.Dataset`オブジェクト
        x_resolution (float):
            リサンプル完了後に期待するX方向の地上分解能
        y_resolution (float):
            リサンプル完了後に期待するY方向の地上分解能。Noneの場合、x_resolutionと
            同じ値が使用されます。
        resample_algorithm (int):
            リサンプリングアルゴリズム。デフォルトは`gdal.GRA_CubicSpline`。
            他の選択肢としては、`gdal.GRA_Bilinear`や`gdal.GRA_Cubic`などがあります。
        reversible (bool):
            リサンプリング後の座標を元の座標になるべく近づけるかどうか。
            Trueの場合、元の座標に近い座標になるようにリサンプリングされます。
            Falseの場合、元の座標に近づけることは保証されません。
        out_crs (CRS):
            投影変換を同時に行う場合の出力座標系。
            `pyproj.CRS`オブジェクトまたはEPSGコードを指定します。
        return_array (bool):
            Trueの場合、リサンプリング後のデータを`numpy`配列として返します。
            Falseの場合、`gdal.Dataset`オブジェクトとして返します。
        forced_metre_system (bool):
            Trueの場合、出力座標系をUTM座標系に強制します。
            Falseの場合、出力座標系は指定された`out_crs`に従います。
        datum_name (str):
            出力座標系の基準楕円体の名前。デフォルトは'WGS 84'。他の選択肢としては、
            'JGD2011'などがあります。
    Returns:
        gdal.Dataset | np.array:
            リサンプリング後の`gdal.Dataset`オブジェクトまたは`numpy`配列。
    """
    return _resample(
        dst=dst,
        resample_algorithm=resample_algorithm,
        reversible=reversible,
        out_crs=out_crs,
        return_array=return_array,
        x_resolution=x_resolution,
        y_resolution=y_resolution if y_resolution is not None else x_resolution,
        forced_metre_system=forced_metre_system,
        datum_name=datum_name,
    )


def resample_with_cell_size_spec(
    dst: gdal.Dataset,
    x_cells: int,
    y_cells: int,
    resample_algorithm: int = gdal.GRA_CubicSpline,
    reversible: bool = True,
    out_crs: CRS = None,
    return_array: bool = True,
    forced_metre_system: bool = False,
    datum_name: str = "WGS 84",
) -> gdal.Dataset:
    """
    ## Summary:
        リサンプル後のセル数を指定して`gdal.Dataset`のリサンプリングを行う関数。
    Args:
        dst (gdal.Dataset):
            リサンプル対象の`gdal.Dataset`オブジェクト
        x_cells (int):
            リサンプル完了後に期待するX方向のセル数
        y_cells (int):
            リサンプル完了後に期待するY方向のセル数
        resample_algorithm (int):
            リサンプリングアルゴリズム。デフォルトは`gdal.GRA_CubicSpline`。
            他の選択肢としては、`gdal.GRA_Bilinear`や`gdal.GRA_Cubic`などがあります。
        reversible (bool):
            リサンプリング後の座標を元の座標になるべく近づけるかどうか。
            Trueの場合、元の座標に近い座標になるようにリサンプリングされます。
            Falseの場合、元の座標に近づけることは保証されません。
        out_crs (CRS):
            投影変換を同時に行う場合の出力座標系。
            `pyproj.CRS`オブジェクトまたはEPSGコードを指定します。
        return_array (bool):
            Trueの場合、リサンプリング後のデータを`numpy`配列として返します。
            Falseの場合、`gdal.Dataset`オブジェクトとして返します。
        forced_metre_system (bool):
            Trueの場合、出力座標系をUTM座標系に強制します。
            Falseの場合、出力座標系は指定された`out_crs`に従います。
        datum_name (str):
            出力座標系の基準楕円体の名前。デフォルトは'WGS 84'。他の選択肢としては、
            'JGD2011'などがあります。
    Returns:
        gdal.Dataset | np.array:
            リサンプリング後の`gdal.Dataset`オブジェクトまたは`numpy`配列。
    """
    return _resample(
        dst=dst,
        resample_algorithm=resample_algorithm,
        reversible=reversible,
        out_crs=out_crs,
        return_array=return_array,
        width=x_cells,
        height=y_cells,
        forced_metre_system=forced_metre_system,
        datum_name=datum_name,
    )
