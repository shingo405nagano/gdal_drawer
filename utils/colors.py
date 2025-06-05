from typing import Any, Iterable, Union

import numpy as np
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    to_hex,
    to_rgb,
    to_rgba,
)

UniqueIterable = Union[list, tuple, np.ndarray]


def dimensional_count(value: Any) -> int:
    """
    ## Summary:
        引数の次元数を測定する関数。
    Arguments:
        value (Any):
            次元数を測定したい値
    Returns:
        int: 次元数
            - 0: 文字列型や数値型などの、次元数を測定できない値。
            - 1: 値がリストやタプル、NumPy配列などの一次元のイテラブルである。
            - 2: 値がリストのリストやタプルのタプル、NumPy配列の二次元配列である。
            - ...
    Examples:
        >>> dimensional_measurement(1)
        0
        >>> dimensional_measurement('a')
        0
        >>> dimensional_measurement([1, 2, 3])
        1
        >>> dimensional_measurement([[1, 2, 3], [4, 5, 6]])
        2
        >>> dimensional_measurement([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        3
    """
    if isinstance(value, UniqueIterable):
        try:
            value = value.tolist()
        except:  # noqa: E722
            try:
                value = value.tolist()
            except:  # noqa: E722
                pass
        return 1 + max(dimensional_count(item) for item in value) if value else 1
    else:
        return 0


class Converter(object):
    def __init__(self, color_list: UniqueIterable, return_type: str = "rgba"):
        """
        ## Summary:
            色のリストを指定された形式に変換するクラス。色には、RGB、RGBA、Hexのいずれかの形式を指定できる。
        Arguments:
            color_list (list | tuple | np.ndarray):
                色のリスト。RGB、RGBA、Hexのいずれかの形式で指定する。
            return_type (str):
                変換後の色の形式。'hex', 'rgb', 'rgba'のいずれかを指定する。
        """
        self._is_alpha = False
        self._color_list = self._check_color_list(color_list)
        self._return_type = self._check_return_type(return_type)

    def _check_color_list(self, color_list: UniqueIterable) -> list[str]:
        """
        ## Summary:
            `color_list`の型をチェックしHexの形式に変換する。
        Args:
            color_list (list | tuple | np.ndarray):
        Returns:
            list[str]:
                Hex形式の色のリスト
        """
        if not isinstance(color_list, UniqueIterable):
            raise TypeError(
                f"color_list must be a list, tuple, or np.ndarray, not {type(color_list)}"
            )
        if len(color_list) == 0:
            # リストが空の場合はエラーを返す
            raise ValueError("color_list cannot be empty")
        if all(isinstance(item, str) for item in color_list):
            # すべての要素が文字列の場合、Hex形式に変換可能かどうかをチェック
            try:
                color_list = [to_hex(item) for item in color_list]
            except ValueError as e:
                raise ValueError("All elements in the list must be in Hex format.") from e
            else:
                self._is_alpha = False
                return color_list

        length_list = [len(item) for item in color_list]
        if all(length == 3 for length in length_list):
            # リストの要素がRGB形式の場合、正規化してHex形式に変換
            color_list = self._normalize(color_list)
            try:
                color_list = [to_hex(item) for item in color_list]
            except ValueError as e:
                raise ValueError("All elements in the list must be in RGB format.") from e
            else:
                self._is_alpha = False
                return color_list
        elif all(length == 4 for length in length_list):
            # リストの要素がRGBA形式の場合、正規化してHex形式に変換
            color_list = self._normalize(color_list)
            try:
                color_list = [to_hex(item, keep_alpha=True) for item in color_list]
            except ValueError as e:
                raise ValueError(
                    "All elements in the list must be in RGBA format."
                ) from e
            else:
                self._is_alpha = True
                return color_list
        else:
            raise ValueError("All items in the list need to be RGB or RGBA or Hex.")

    def _normalize(self, color_list: UniqueIterable) -> list[float]:
        # RGBまたはRGBAの場合、matplotlibで処理しやすいように正規化する
        array = np.array(color_list)
        max_ = np.max(array)
        min_ = np.min(array)
        if (0 <= min_) and (max_ <= 1):
            return color_list
        elif (0 <= min_) and (max_ <= 255):
            return (array / 255).tolist()
        elif (0 <= min_) and (max_ <= 65535):
            return (array / 65535).tolist()
        else:
            return color_list

    def _check_return_type(self, return_type: str) -> str:
        """
        ## Summary:
            `return_type`の型をチェックし、指定された形式に変換する。
        Args:
            return_type (str):
                変換後の色の形式。'hex', 'rgb', 'rgba'のいずれかを指定する。
        Returns:
            str:
                変換後の色の形式。'hex', 'rgb', 'rgba'のいずれかを返す。
        """
        if not isinstance(return_type, str):
            raise TypeError("return_type must be a string")
        return_type = return_type.lower()
        if return_type not in ["hex", "rgb", "rgba"]:
            raise ValueError("return_type must be one of ['hex', 'rgb', 'rgba']")
        return return_type


class CustomCmap(object):
    """
    ## Summary
        自作のColorMapを作成するクラス。
    """

    def to_mpl_color_list(
        self,  #
        color_list: UniqueIterable,
        rgba: bool = True,
    ) -> list[list[float, float, float]]:
        """
        ## Summary
            matplotlibで使用するための色のリストを作成する。
        Args:
            color_list (list | tuple | np.ndarray):
                色のリスト。RGB、RGBA、Hexのいずれかの形式で指定する。
            rgba (bool):
                RGBA形式で返すかどうか。デフォルトはTrue。
        Returns:
            list[list[float, float, float]]:
                色のリスト。RGBA形式で返す場合は4つの要素を持つリスト、
                RGB形式で返す場合は3つの要素を持つリスト。
                値は0から1の範囲に正規化されている。
        """
        converter = Converter(color_list, return_type="rgba" if rgba else "rgb")
        if rgba:
            return [to_rgba(c) for c in converter._color_list]
        else:
            return [to_rgb(c) for c in converter._color_list]

    def __create_position(self, colors: list[Any]) -> list[float]:
        """
        ## Summary
            色のリストから位置のリストを作成する。
        """
        length = len(colors)
        return [i / (length - 1) for i in range(length)]

    def __check_position(self, positions: list[float]) -> list[float]:
        """
        ## Summary
            位置のリストをチェックし、0から1の範囲に正規化する。
        """
        positions[0] = 0.0
        positions[-1] = 1.0
        positions.sort()
        if all([0 <= p <= 1 for p in positions]):
            return positions
        return self.__create_position(positions)

    def color_list_to_linear_cmap(
        self,  #
        color_list: UniqueIterable,
        **kwargs,
    ) -> Union["LinearColorMap", LinearSegmentedColormap]:
        """
        ## Summary
            色のリストから線形カラーマップを作成する。
        Args:
            color_list (list | tuple | np.ndarray):
                色のリスト。RGB、RGBA、Hexのいずれかの形式で指定する。
            **kwargs:
                - positions (list[float]): 色の位置を指定するリスト。デフォルトは自動生成される。
                - name (str): カラーマップの名前。デフォルトは"custom_cmap"。
        Returns:
            LinearColorMap:
                作成された線形カラーマップ。
        """
        positions = kwargs.get("positions", self.__create_position(color_list))
        positions = self.__check_position(positions)
        # カラーマップを作成
        color_list = [
            (position, color)
            for position, color in zip(positions, color_list, strict=False)
        ]
        name = kwargs.get("name", "custom_cmap")
        cmap = LinearSegmentedColormap.from_list(name, color_list, N=256)
        return LinearColorMap(cmap)


class LinearColorMap(object):
    """
    ## Summary
        matplotlibの線形カラーマップをラップするクラス。
    """

    def __init__(self, cmap: LinearSegmentedColormap | ListedColormap):
        if isinstance(cmap, ListedColormap):
            colors = cmap(np.linspace(0, 1, 10))
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        if not isinstance(cmap, LinearSegmentedColormap):
            raise ValueError("cmap must be a LinearSegmentedColormap")
        self.cmap = cmap

    def __call__(self, position: int):
        return self.cmap(position)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.cmap, name)

    def get(self, position: int, return_type: str = "rgba") -> Any:
        """
        ## Summary
            このメソッドは、作成されたカラーマップの指定された位置の色を返します。
            RGB, RGBA, Hex, int あるいは inta 形式で返します。
        Args:
            position (int | Iterable[int]):
                'position'は、カラーマップから取得する色の位置を指定します。
                指定は整数で、0から255の範囲でなければなりません。
            return_type (str):
                'return_type'は、返される色の形式を指定します。
                'rgb', 'rgba', 'hex', 'int', 'inta'のいずれかを指定できます。
                - rgb: RGB形式の色を返します。
                - rgba: RGBA形式の色を返します。
                - hex: 16進数形式の色コードを返します。
                - int: RGB形式の整数値を返します。
                - inta: RGBA形式の整数値を返します。
        Returns:
            Any:
                指定された形式での色を返します。
                - rgb: RGB形式の色を返します。
                - rgba: RGBA形式の色を返します。
                - hex: 16進数形式の色コードを返します。
                - int: RGB形式の整数値を返します。
                - inta: RGBA形式の整数値を返します。
        Examples:
            >>> # Get Index color.
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> cmap.get(0)
            (1.0, 1.0, 1.0)
            >>> # Get the color of the Index array.
            >>> cmap.get([0, 128, 255], 'rgba')
            [(1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)]
        """
        if isinstance(position, (int, np.integer)):
            return self._get(position, return_type)
        elif isinstance(position, Iterable):
            result = []
            for pos in position:
                if isinstance(pos, (int, np.integer)):
                    result.append(self._get(pos, return_type))
                else:
                    try:
                        result.append(self._get_in_list(pos, return_type))
                    except TypeError as e:
                        raise TypeError(
                            f"position must be int or Iterable[int], not {type(pos)}"
                        ) from e
            return np.array(result)

    def _get(self, position: int, return_type: str = "rgb") -> Any:
        """
        ## Summary
            ColorMapから指定された位置の色を取得します。
        Args:
            position (int):
                カラーマップから取得する色の位置を指定します。
                0から255の範囲でなければなりません。
            return_type (str):
                返される色の形式を指定します。
                'rgb', 'rgba', 'hex', 'int', 'inta'のいずれかを指定できます。
        """
        transparency = False
        if position < 0 or 255 < position:
            # Return transparent color if out of range.
            transparency = True
        pattern = ["rgb", "rgba", "hex", "int", "inta"]
        return_type = return_type.lower()
        if return_type not in pattern:
            # return_type must be one of pattern
            raise ValueError(f"return_type must be one of {pattern}")
        # Get the color at the given position
        color = (1.0, 1.0, 1.0, 0) if transparency else self.cmap(position)
        if return_type == "rgb":
            # If RGB is specified.
            if len(color) == 4:
                return color[:-1]
            else:
                return color[:-1]
        elif return_type == "rgba":
            # If RGBA is specified.
            if len(color) == 3:
                return color + (1.0,)
            else:
                return color
        elif return_type == "hex":
            # If Hex is specified.
            return to_hex(color)
        elif return_type == "int" and len(color) == 4:
            # If int is specified.
            if len(color) == 4:
                return tuple(int(c * 255) for c in color[:-1])
            else:
                return tuple(int(c * 255) for c in color)
        elif return_type == "inta":
            # If inta is specified.
            if len(color) == 3:
                return tuple(int(c * 255) for c in color) + (255,)
            else:
                return tuple(int(c * 255) for c in color)

    def _get_in_list(
        self,  #
        positions: Iterable[int],
        return_type: str = "rgb",
    ) -> UniqueIterable:
        """
        ## Summary
            ColorMapから指定された位置の色を取得します。
        Args:
            positions (Iterable[int]):
                カラーマップから取得する色の位置を指定します。
                0から255の範囲でなければなりません。
            return_type (str): The
                返される色の形式を指定します。
                'rgb', 'rgba', 'hex', 'int', 'inta'のいずれかを指定できます。
        Returns:
            List:
                - rgb: RGB形式の色を返します。
                - rgba: RGBA形式の色を返します。
                - hex: 16進数形式の色コードを返します。
                - int: RGB形式の整数値を返します。
                - inta: RGBA形式の整数値を返します。
        """
        return [self._get(pos, return_type) for pos in positions]

    def get_registered_color(self, return_type: str = "rgb") -> UniqueIterable:
        """
        ## Summary
            cmapに登録されているすべての色をListで取得します。 cmapには256色が登録さ
            れています。
        Args:
            return_type (str):
                返される色の形式を指定します。
                'rgb', 'rgba', 'hex', 'int', 'inta'のいずれかを指定できます。
        Returns:
            List:
                長さは256で、各要素は以下のいずれかの形式で格納される。
                - rgb: List of Tuple of 3 float
                - rgba: List of Tuple of 4 float
                - hex: List of Hexadecimal color code
                - int: List of Tuple of 3 integers
                - inta: List of Tuple of 4 integers
        Examples:
            >>> custom_cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> custom_cmap.get_registered_color('rgba')
            [(1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), ...]
        """
        index = list(range(256))
        return [self.get(i, return_type) for i in index]

    def generate_idx_for_retrieval(
        self,
        values: Iterable[float] | Iterable[int],
        in_nodata_value: Any = np.nan,
        out_nodata_index: int = -1,
    ) -> Iterable[int]:
        """
        ## Summary
            ColorMapから取得するIndexを生成します。例えばこれは、連続値の配列に色を
            割り当てるために使用されます。
        Args:
            values (Iterable[float] | Iterable[int]):
            Indexに変換する値の配列。これは、0-255の範囲に収まるように正規化される、
            ただし、'in_nodata_value'、nan、infの値は'out_nodata_index'に置き換えられる。
            配列は1次元または2次元のnp.ndarrayでなければならない。
        Returns:
            Iterable[int]:
            0-255の範囲に正規化された値のIndexの配列。nanやinfの値は、'out_nodata_index'
            に置き換えられる。
        Examples:
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> values = np.random.normal(0, 1, 100).reshape(10, 10)
            >>> indices = cmap.generate_idx_for_retrieval(values)
        """
        if not isinstance(values, np.ndarray) and (2 <= dimensional_count(values) <= 3):
            values = np.array(values)
        if not isinstance(values, np.ndarray):
            raise ValueError("values must be an Iterable")
        # Convert np.inf or NoData values to nan.
        values = np.where(values == in_nodata_value, np.nan, values)
        values = np.where(np.isinf(values), np.nan, values)
        # Get the index of the nan.
        nan_idx = np.isnan(values)
        # Normalize to the range 0-255.
        max_ = np.nanmax(values)
        min_ = np.nanmin(values)
        mean_ = np.nanmean(values)
        values[nan_idx] = mean_
        index_ary = np.round((values - min_) / (max_ - min_) * 255).astype(int)
        # If nan was entered, replace with out_nodata_index.
        index_ary[nan_idx] = out_nodata_index
        return index_ary

    def values_to_img(
        self,
        values: Iterable[Iterable[float]],
        nodata_value: Any = -1,
        return_type: str = "inta",
    ) -> np.ndarray:
        """
        ## Summary
            連続値の配列からRGB画像を作成する。
        Args:
            values (Iterable[Iterable[float]]):
                連続値の配列。1次元または2次元のnp.ndarrayでなければならない。
            return_type (str):
                返される画像の形式を指定します。
                'rgb', 'rgba', 'int', 'inta'のいずれかを指定できます。
        Returns:
            np.ndarray:
                RGB Image
        Examples:
            >>> # Create an RGB image using a color map set up from a two-dimensional array of continuous values.
            >>> custom_cmap = CustomColorMap()
            >>> cmap = custom_cmap.color_list_to_linear_cmap(['red', 'green', 'blue'])
            >>> values = np.random.normal(0, 1, 100).reshape(10, 10)
            >>> img = cmap.values_to_img(values)
            >>> plt.imshow(img)
            >>> plt.show()
            >>> #--------------------------------
            >>> # Create RGB images using matplotlib colormaps.
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> img = cmap.values_to_img(values)
            >>> plt.imshow(img)
            >>> plt.show()
        """
        pattern = ["rgb", "rgba", "int", "inta"]
        return_type = return_type.lower()
        if return_type not in pattern:
            # return_type must be one of pattern
            raise ValueError(f"return_type must be one of {pattern}")
        # Get Index of NoData
        indices = self.generate_idx_for_retrieval(values)
        nodata_idxs = indices == nodata_value
        colors = np.array(self.get_registered_color(return_type))
        img = colors[indices]
        # Convert Index of NoData to transparent color
        if img.shape[-1] == 4:
            img[nodata_idxs] = [255, 255, 255, 0]
        else:
            img[nodata_idxs] = [255, 255, 255]
        return img
