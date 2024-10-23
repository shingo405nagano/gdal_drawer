"""
# Summary
ラスターデータの畳み込みを行うモジュール

"""
from typing import NamedTuple

import numpy as np
from osgeo import gdal
import scipy.ndimage

from gdal_projection import gprojection


class KernelSize(NamedTuple):
    x: int
    y: int

class Kernels(object):
    def _distance_to_kernel_size(self, one_side_distance: float, cell_size: float) -> int:
        """
        カーネルのサイズを計算する。
        Args:
            one_side_distance(int): 中心からの距離
            cell_size(float): セルのサイズ
        Returns:
            cells(int): カーネルのサイズ
        """
        divd, remainder = divmod(one_side_distance, abs(cell_size))
        if 0 < remainder:
            divd += 1
        return int(divd * 2) + 1
    
    def distance_to_kernel_size(self, 
        x_dist: float, 
        x_cell_size: float, 
        **kwargs
    ) -> KernelSize:
        """
        中心からの距離からカーネルサイズを計算する。
        Args:
            x_dist(float): X方向の中心からの距離
            x_cell_size(float): X方向のセルのサイズ
            Kwargs:\n
                y_dist(float): Y方向の中心からの距離\n
                y_cell_size(float): Y方向のセルのサイズ
        Returns:
            KernelSize: カーネルのサイズ
                - x(int): X方向のカーネルのサイズ（X方向の辺の長さ）
                - y(int): Y方向のカーネルのサイズ（Y方向の辺の長さ）
        """
        x_size = self._distance_to_kernel_size(x_dist, x_cell_size)
        y_dist = kwargs.get('y_dist', x_dist)
        y_size = kwargs.get('y_cell_size', x_cell_size)
        y_size = self._distance_to_kernel_size(y_dist, y_size)
        return KernelSize(x_size, y_size)

    def kernel_size_from_dst_unit(self, 
        dst: gdal.Dataset, 
        x_dist: float,
        metre: bool=True,
        **kwargs
    ) -> KernelSize:
        """
        メートル単位の距離をカーネルのサイズに変換する。この関数では`dst: gdal.Dataset`
        の投影法が Degree か Metre かを判定し、メートル単位の距離をカーネルのサイズに変換する。
        Args:
            dst(gdal.Dataset): ラスターデータ
            x_dist(float): 中心からの距離
            Kwargs:\n
                y_dist(float): Y方向の中心からの距離
        Returns:
            int: カーネルのサイズ
        """
        cell_sizes = self.dst_cell_size_of_metre(dst)
        return self.distance_to_kernel_size(
            x_dist=one_side_metres, x_cell_size=cell_sizes.x, 
            y_dist=one_side_metres, y_cell_size=cell_sizes.y
        )

    def cell_to_kernel_size(one_side_cells: int) -> int:
        """
        カーネルのサイズを計算する。
        Args:
            one_side_distance(int): 中心からの距離
        Returns:
            cells(int): カーネルのサイズ
        """
        return int(one_side_cells * 2) + 1
    
    def simple_kernel(distance: int) -> np.ndarray:
        """
        畳み込みに使用するシンプルなカーネルを生成する。
        Args:
            distance(int): カーネルのサイズ
        Returns:
            np.ndarray
        Examples:
            >>> simple_kernel(3)
            array([[0.11111111, 0.11111111, 0.11111111],
                   [0.11111111, 0.11111111, 0.11111111],
                   [0.11111111, 0.11111111, 0.11111111]])
        """
        shape = (distance, distance)
        cells = distance * distance
        return np.ones(shape) / cells
    


class GdalConvolution(object):
    def mean_filter(self, dst: gdal.Dataset, kernel: np.ndarray) -> gdal.Dataset:
        """
        平均値フィルタを適用する
        """
        pass

    def gaussian_filter(self, dst: gdal.Dataset, sigma: float) -> gdal.Dataset:
        """
        ガウシアンフィルタを適用する
        """
        pass

    def maximum_filter(self, dst: gdal.Dataset, kernel: np.ndarray) -> gdal.Dataset:
        """
        最大値フィルタを適用する
        """
        pass

    def minimum_filter(self, dst: gdal.Dataset, kernel: np.ndarray) -> gdal.Dataset:
        """
        最小値フィルタを適用する
        """
        pass

    def median_filter(self, dst: gdal.Dataset, kernel: np.ndarray) -> gdal.Dataset:
        """
        中央値フィルタを適用する
        """
        pass

    def std_filter(self, dst: gdal.Dataset, kernel: np.ndarray, sigma: float) -> gdal.Dataset:
        """
        標準偏差フィルタを適用する
        """
        pass

    def sobel_filter(self, dst: gdal.Dataset) -> gdal.Dataset:
        """
        Sobelフィルタを適用する
        """
        pass



gconvolution = GdalConvolution()
kernels = Kernels()

if __name__ == '__main__':
    fp = r"D:\Repositories\ProcessingRaster\datasets\TEST_GDAL_DRAER\DTM__R0_5__EPSG4326.tif"
    dst = gdal.Open(fp)
    print(kernels.unit_metre_to_kernel_size(dst, )))