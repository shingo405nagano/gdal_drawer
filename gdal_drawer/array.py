from typing import Any, Optional

import numpy as np
from PIL import Image

from .info import GdalInfo

Image.MAX_IMAGE_PIXELS = None  # Disable the safety check


class Array:
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"Loading {file_path} ...")
        self._info = GdalInfo(file_path)
        self._img = np.array(Image.open(file_path))

    def nodata_value(self, band: Optional[int] = None) -> Any:
        if band is not None:
            return self._info.nodata_values[band]  # type: ignore
        else:
            return self._info.nodata_values

    @property
    def band_count(self) -> int:
        """
        ## Summary:
            ラスターデータのバンド数を返す
        """
        shape = self._img.shape
        if len(shape) == 2:
            return 1
        return shape[-1]

    def array(self, band: Optional[int] = None) -> np.ndarray:
        """
        ## Summary:
            ラスターデータをNumPy配列として返す。配列の形状は(band, height, width)となる。
        Args:
            band (int, optional):
                取得するバンド番号。1始まりなので注意。Noneの場合は全バンドを返す。
        """
        if self.band_count == 1:
            return self._img
        else:
            ary = np.array([self._img[:, :, i] for i in range(self.band_count)])
            if band is not None:
                return ary[band - 1]
            else:
                return ary

    def height(self) -> int:
        """
        ## Summary:
            ラスターデータの高さを返す
        """
        return self._img.shape[0]

    def width(self) -> int:
        """
        ## Summary:
            ラスターデータの幅を返す
        """
        return self._img.shape[1]

    def nodata_mask(self, band: int) -> np.ndarray:
        """
        ## Summary:
            NoData値のマスクを返す。NoData値の位置がTrueとなる。
        """
        nodata_value = self.nodata_value(band)
        if nodata_value == "No setting":
            return np.zeros(self.array(band).shape, dtype=bool)
        ary = self.array(band)
        mask = ary == nodata_value
        mask = np.where(np.isnan(ary), False, mask)  # type: ignore
        mask = np.where(np.isinf(ary), False, mask)  # type: ignore
        return mask

    def overwrite_nodata(self, band: int, overwrite_value: Any) -> np.ndarray:
        """
        ## Summary:
            指定したバンドのNoData値を指定した値で上書きした配列を返す。
        Args:
            band (int):
                上書きするバンド番号。1始まりなので注意。
            overwrite_value (Any):
                上書きする値。
        """
        ary = self.array(band).copy()
        mask = self.nodata_mask(band)
        if np.sum(mask) == 0:
            return ary
        else:
            ary[mask] = overwrite_value
            return ary
