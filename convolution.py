"""
# Summary
ラスターデータの畳み込みを行うモジュール

"""
import numpy as np
from osgeo import gdal
import scipy.ndimage


class RasterConvolution(object):
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



raster_convolution = RasterConvolution()