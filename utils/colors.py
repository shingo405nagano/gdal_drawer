"""
Do not use doctest.

このモジュールでは、matplotlibのカラーマップを拡張するためのクラスを提供しています。

# Example
-------------------------------------------------------------------------------
## 連続値を使用したカスタムColorMapの作成
これは`matplotlib.colors.LinearSegmentedColormap`の拡張クラスを返す。
```python
>>> colors = ['red', 'green', 'blue']
>>> custom_cmap = CustomCmap()
>>> cmap = custom_cmap.color_list_to_linear_cmap(colors)
>>> type(cmap)
<class '__main__.LinearColorMap'>
```
-------------------------------------------------------------------------------
## 作成したカスタムColorMapからIndexを指定して色を取得する
```python
>>> # Indexの色を取得する
>>> cmap.get(0)
(1.0, 0.0, 0.0)
>>> # rgbaの色を取得する
>>> cmap.get(0, 'rgba')
(1.0, 0.0, 0.0, 1.0)
>>> # Hexの色を取得する
>>> cmap.get(0, 'hex')
'#ff0000'
>>> # intの色を取得する
>>> cmap.get(0, 'int')
(255, 0, 0)
>>> # intaの色を取得する
>>> cmap.get(0, 'inta')
(255, 0, 0, 255)
>>> # Index配列で色を取得する。2次元でも可
>>> idx = [0, 128, 255]
>>> cmap.get(idx, 'rgba')
[(1.0, 0.0, 0.0, 1.0), (0.0, 0.5019607843137255, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
```
-------------------------------------------------------------------------------
## カスタムColorMapに登録された色を全て取得する
```python
>>> rgba_list = cmap.get_registered_color('rgba')
[(1.0, 0.0, 0.0, 1.0), (0.0, 0.5019607843137255, 0.0, 1.0), ...]
>>> len(rgba_list)
256
```
-------------------------------------------------------------------------------
## 連続値の配列からRGB画像を作成する
```python
>>> values = np.random.normal(0, 1, 100).reshape(10, 10)
>>> img = cmap.values_to_img(values, 'inta')
>>> plt.imshow(img)
>>> plt.show()
```
"""
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

from matplotlib.colors import to_hex
from matplotlib.colors import to_rgba
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

class CustomCmap(object):
    @staticmethod
    def __to_rgba(arg_index: int, arg_name: str) -> Callable:
        """
        Listの要素をTuple(float)に変換するデコレータ。
        引数として想定しているのは、List[str(Hex)]か、あるいはList[int(0-255)]。
        """
        def decorator(func: Callable) -> str:
            def wrapper(self, *args, **kwargs) -> str:
                is_args = True
                if arg_index < len(args):
                    colors = args[arg_index]
                elif arg_name in kwargs:
                    colors = kwargs[arg_name]
                    is_args = False
                else:
                    raise ValueError(f'{arg_name} is required.')
                # Convert to rgba color
                converted_colors = []
                for color in colors:
                    if isinstance(color, int):
                        converted_colors.append(color / 255)
                    elif isinstance(color, str):
                        converted_colors.append(to_rgba(color))
                    elif isinstance(color, float):
                        converted_colors.append(color)
                    else:
                        raise ValueError(f'Invalid {arg_name} type.')
                if is_args:
                    args = list(args)
                    args[arg_index] = converted_colors
                    return func(self, *args, **kwargs)
                else:
                    kwargs[arg_name] = converted_colors
                    return func(self, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def __round_values(digits: int):
        """戻り値のリストの要素を小数点第4位まで丸めるデコレータ。"""
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                if isinstance(result, list):
                    return [[round(c, digits) for c in col] for col in result]
                return result
            return wrapper
        return decorator

    @staticmethod
    def __check_float_values(arg_index: int, arg_name: str):
        """Listの要素がfloatであるかどうかをチェックするデコレータ。"""
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                # データの取得
                is_args = True
                if arg_index < len(args):
                    colors = args[arg_index]
                elif arg_name in kwargs:
                    colors = kwargs[arg_name]
                    is_args = False
                else:
                    raise ValueError(f'{arg_name} is required.')
                # listの中身が変換可能かどうかをチェック
                colors = self.__contents_is_mpl_color_lst_or_not(colors)
                # 再格納
                if is_args:
                    args = list(args)
                    args[arg_index] = colors
                    return func(self, *args, **kwargs)
                else:
                    kwargs[arg_name] = colors
                    return func(self, *args, **kwargs)
            return wrapper
        return decorator

    def __contents_is_mpl_color_lst_or_not(self, 
        colors: List[Any]
    ) -> List[Tuple[float]]:
        is_ok = False
        try:
            # Listの中身が[0, 1]の範囲内が格納されたTupleであるかどうかをチェック
            is_floats = [all(isinstance(v, float) for v in color) for color in colors]
            defined_range = []
            for color in colors:
                if all([0 <= v <= 1 for v in color]):
                    defined_range.append(True)
                else:
                    defined_range.append(False)
            is_ok = all(is_floats) and all(defined_range)
            if is_ok:
                return colors
        except:
            pass
        try:
            # Listの中身がHexであるかどうかをチェック
            colors = [to_rgba(color) for color in colors]
            return colors
        except:
            pass
        try:
            colors = [color / 255 for color in colors]
            return colors
        except:
            raise ValueError("Invalid color type. Please check the type of 'colors' in arguments.")

    @__to_rgba(0, 'colors')
    @__round_values(digits=5)
    def to_mpl_color_list(self, 
        colors: List[str | int], 
        rgba: bool=True
    ) -> List[float]:
        """
        いくつかの色をリストで受け取り、RGBAのリストに変換する。色の指定方法は、Hex または0-255の整数で指定する。
        Args:
            colors (List[str | int]): 色のリスト
            rgba (bool, optional): RGBAのリストに変換するかどうか. Defaults to True. FalseならばRGBのリストに変換する。
        Returns:
            List[float]: RGBAのリスト
        Examples:
            >>> colors = Colors()
            >>> colors.to_mpl_color_list(['red', 'green', 'blue'], False)
            [(1.0, 0.0, 0.0), (0.0, 0.5019607843137255, 0.0), (0.0, 0.0, 1.0)]
        """
        if rgba:
            return colors
        else:
            return [c[: 3] for c in colors]

    @__check_float_values(0, 'colors')
    def color_list_to_linear_cmap(self,
            colors: List[Tuple[float]] | List[Tuple[int] | List[str]],
            **kwargs
        ) -> Union['LinearColorMap', LinearSegmentedColormap]:
        """
        色のリストをカラーマップに変換する。
        Args:
            colors (List[Tuple[float]] | List[str] | List[Tuple[int]]): 色のリスト\n
                - List[Tuple[0.0 ~ 1.0]]:
                - List[str]: Hexカラーコードのリスト
                - List[Tuple[int]]: 0-255の整数のリスト
            **kwargs:
                - positions (List[float]): 色の位置を指定する。デフォルトはNone。指定する場合は、Colorsと同じ長さのList[float]を指定する。floatは0.0 ~ 1.0の範囲内で指定。
                - name (str): カラーマップの名前を指定する。デフォルトは'custom_cmap'。
        Returns:
            LinearSegmentedColormap: カラーマップ
        """
        # 色を配置する位置を指定
        positions = kwargs.get('positions', self.__create_position(colors))
        positions = self.__check_position(positions)
        # カラーマップを作成
        colors = [(position, color) for position, color in zip(positions, colors)]
        name = kwargs.get('name', 'custom_cmap')
        cmap = LinearSegmentedColormap.from_list(name, colors, N=256)
        return LinearColorMap(cmap)

    def __create_position(self, colors: List[Any]) -> List[float]:
        length = len(colors)
        return [i / (length - 1) for i in range(length)]
    
    def __check_position(self, positions: List[float]) -> List[float]:
        positions[0] = 0.0
        positions[-1] = 1.0
        positions.sort()
        if all([0 <= p <= 1 for p in positions]):
            return positions
        return self.__create_position(positions)



class LinearColorMap(object):
    def __init__(self, cmap):
        if not isinstance(cmap, LinearSegmentedColormap):
            raise ValueError('cmap must be a LinearSegmentedColormap')
        self.cmap = cmap
    
    def __call__(self, position: int):
        return self.cmap(position)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.cmap, name)
    
    def get(self, position: int | Iterable[int], return_type: str = 'rgba'):
        """
        ## Summary
            このメソッドは、作成したcolormapの指定された位置の色をRGB、RGBA、Hex、int、intaのいずれかの形式で返します。
        Args:
            position (int | Iterable[int]): カラーマップ内の色の位置。整数または整数の配列。
            return_type (str): The type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            Any:
                - rgb: Tuple of 3 float
                - rgba: Tuple of 4 float
                - str: Hexadecimal color code
                - int: Tuple of 3 integers
                - inta: Tuple of 4 integers
        Examples:
            >>> # Indexの色を取得する
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> cmap.get(0)
            (1.0, 1.0, 1.0)
            >>> # Index配列の色を取得する
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
                    result.append(self._get_in_list(pos, return_type))
            return np.array(result)
            
    def _get(self, position: int, return_type: str = 'rgb'):
        """
        ## Summary
            colormapの指定された位置の色を取得する。
        Args:
            position (int): カラーマップ内の色の位置
            return_type (str): The type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            Any:
                - rgb: Tuple of 3 float
                - rgba: Tuple of 4 float
                - str: Hexadecimal color code
                - int: Tuple of 3 integers
                - inta: Tuple of 4 integers
        """
        transparency = False
        if position < 0 or 255 < position:
            # 範囲外の場合は透明色を返す
            transparency = True
        pattern = ['rgb', 'rgba', 'hex', 'int', 'inta']
        return_type = return_type.lower()
        if return_type not in pattern:
            # return_type must be one of pattern
            raise ValueError(f'return_type must be one of {pattern}')
        # Get the color at the given position
        color = (1., 1., 1., 0) if transparency else self.cmap(position)
        if return_type == 'rgb':
            # RGBが指定された場合
            if len(color) == 4:
                return color[:-1]
            else:
                return color[:-1]
        elif return_type == 'rgba':
            # RGBAが指定された場合
            if len(color) == 3:
                return color + (1.0,)
            else:
                return color
        elif return_type == 'hex':
            # Hexが指定された場合
            return to_hex(color)
        elif return_type == 'int' and len(color) == 4:
            # intが指定された場合
            if len(color) == 4:
                return tuple(int(c * 255) for c in color[:-1])
            else:
                return tuple(int(c * 255) for c in color)
        elif return_type == 'inta':
            # intaが指定された場合
            if len(color) == 3:
                return tuple(int(c * 255) for c in color) + (255,)
            else:
                return tuple(int(c * 255) for c in color)
    
    def _get_in_list(self, 
        positions: Iterable[int], 
        return_type: str = 'rgb'
    ) -> List[Tuple[float]]:
        """
        ## Summary
            カラーマップ内から指定したの色のリストを取得してListに格納する。
        Args:
            positions (Iterable[int]): カラーマップ内の色の位置のリスト
            return_type (str): The type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            List:
                - rgb: List of Tuple of 3 float
                - rgba: List of Tuple of 4 float
                - str: List of Hexadecimal color code
                - int: List of Tuple of 3 integers
                - inta: List of Tuple of 4 integers
        """
        return [self._get(pos, return_type) for pos in positions]

    def get_registered_color(self, 
        return_type: str = 'rgb'
    ) -> List[Tuple[float]]:
        """
        ## Summary
            cmapに登録された色をListで全て取得する。cmapには256個の色が格納されている。
        Args:
            return_type (str): The type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            List:
                長さは256で、各要素は以下のいずれかの形式で格納されている。
                - rgb: List of Tuple of 3 float
                - rgba: List of Tuple of 4 float
                - str: List of Hexadecimal color code
                - int: List of Tuple of 3 integers
                - inta: List of Tuple of 4 integers
        Examples:
            >>> custom_cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> custom_cmap.get_registered_color('rgba')
            [(1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), ...]
        """
        index = list(range(256))
        return [self.get(i, return_type) for i in index]

    def generate_idx_for_retrieval(self, 
        values: Iterable[float] | Iterable[int],
        in_nodata_value: Any=np.nan,
        out_nodata_index: int=-1
    ) -> Iterable[int]:
        """
        ## Summary
            ある値の配列から、カラーマップのインデックスを生成する。これは、カラーマップから色を取得するために使用される。
        Args:
            values (Iterable[float] | Iterable[int]): Indexに変換したい値の配列。これは、0-255の範囲に収まるように正規化されるが、'in_nodata_value'や nan、infの値は'out_nodata_index'に置換される。
            配列は、1次元または2次元のnp.ndarrayである必要がある。
        Returns:
            Iterable[int]: The index for retrieval
        Examples:
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> values = np.random.normal(0, 1, 100).reshape(10, 10)
            >>> indices = cmap.generate_idx_for_retrieval(values)
        """
        if not isinstance(values, np.ndarray) and isinstance(values, Iterable):
            values = np.array(values)
        if not isinstance(values, np.ndarray):
            raise ValueError('values must be an Iterable')
        # np.inf や NoDataの値をnanに変換
        values = np.where(values == in_nodata_value, np.nan, values)
        values = np.where(np.isinf(values), np.nan, values)
        # nanのIndexを取得しておく
        nan_idx = np.isnan(values)
        # 0-255の範囲に正規化
        max_ = np.nanmax(values)
        min_ = np.nanmin(values)
        mean_ = np.nanmean(values)
        values[nan_idx] = mean_
        index_ary = np.round((values - min_) / (max_ - min_) * 255).astype(int)
        # nanが入力されていた場合は、out_nodata_indexに置換
        index_ary[nan_idx] = out_nodata_index
        return index_ary

    def values_to_img(self, 
        values: Iterable[Iterable[float]], 
        nodata_value: Any = -1,
        return_type: str='inta'
    ) -> np.ndarray:
        """
        ## Summary
            連続値の配列からRGB画像を作成する。
        Args:
            values (Iterable[Iterable[float]]): 連続値の配列
            return_type (str): The type of the return value. Can be 'rgb' or 'rgba' or 'int' or 'inta'
        Returns:
            np.ndarray: RGB画像
        Examples:
            >>> # 連続値の2次元配列から設定したカラーマップを使ってRGB画像を作成する。
            >>> custom_cmap = CustomColorMap()
            >>> cmap = custom_cmap.color_list_to_linear_cmap(['red', 'green', 'blue'])
            >>> values = np.random.normal(0, 1, 100).reshape(10, 10)
            >>> img = cmap.values_to_img(values)
            >>> plt.imshow(img)
            >>> plt.show()
            >>> #--------------------------------
            >>> # matplotlibのカラーマップを使ってRGB画像を作成する。
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> img = cmap.values_to_img(values)
            >>> plt.imshow(img)
            >>> plt.show()
        """
        pattern = ['rgb', 'rgba', 'int', 'inta']
        return_type = return_type.lower()
        if return_type not in pattern:
            # return_type must be one of pattern
            raise ValueError(f'return_type must be one of {pattern}')
        # NoDataのIndexを取得
        indices = self.generate_idx_for_retrieval(values)
        nodata_idxs = indices == nodata_value
        colors = np.array(self.get_registered_color(return_type))
        img = colors[indices]
        # NoDataのIndexを透明色に変換
        if img.shape[-1] == 4:
            img[nodata_idxs] = [0, 0, 0, 0]
        else:
            img[nodata_idxs] = [255, 255, 255]
        return img