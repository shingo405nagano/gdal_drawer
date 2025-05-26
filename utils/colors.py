from typing import Any, Iterable, Union

from matplotlib.colors import (
    to_hex, to_rgb, to_rgba, 
    LinearSegmentedColormap,
    ListedColormap
)
import numpy as np

UniqueIterable = Union[list, tuple, np.ndarray]


def dimensional_count(value: UniqueIterable) -> int:
    """
    ## Summary:
        Recursively determine the dimensionality of a list.
    Arguments:
        value (tuple | list | np.ndarray):
            The list to be measured.
    Returns:
        int: The dimensionality of the list.
            - 0: The value is not a list. (str, int, float, etc.)
            - 1: The value is a list.
            - 2: The value is a list of lists.
            - 3: The value is a list of lists of lists.
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
        except:
            try:
                value = value.tolist()
            except:
                pass
        return 1 + max(dimensional_count(item) for item in value) if value else 1
    else:
        return 0
    

class Converter(object):
    def __init__(self, color_list: UniqueIterable, return_type: str = 'rgba'):
        """
        ## Summary:
            Convert a list of colors to a specified format.
        Arguments:
            color_list (list | tuple | np.ndarray):
                The list of colors to be converted.
            return_type (str):
                The format to convert the colors to. Options are 'hex', 'rgb', or 'rgba'.
        """
        self._is_alpha = False
        self._color_list = self._check_color_list(color_list)
        self._return_type = self._check_return_type(return_type)
        
    def _check_color_list(self, color_list: UniqueIterable) -> list[str]:
        """
        ## Summary:
            Check the color list and convert it to a valid format.
        Args:
            color_list (list | tuple | np.ndarray): 
                The list of colors to be checked and converted.
        Returns:
            list[str]:
                The converted color list in hex format.
        """
        if not isinstance(color_list, UniqueIterable):
            raise TypeError("color_list must be a list, "
                            f"tuple, or np.ndarray, not {type(color_list)}")
        # Check if the color_list is empty
        if len(color_list) == 0:
            raise ValueError("color_list cannot be empty")
        if all(isinstance(item, str) for item in color_list):
            # Check if all items are strings
            try:
                color_list = [to_hex(item) for item in color_list]
            except ValueError:
                raise ValueError("Invalid color string in color_list."
                                 " Must be a valid color name or hex code.")
            else:
                self._is_alpha = False
                return color_list
        length_list = [len(item) for item in color_list]
        if all(length == 3 for length in length_list):
            color_list = self._normalize(color_list)
            try:
                color_list = [to_hex(item) for item in color_list]
            except ValueError:
                raise ValueError("All elements in the list must be in RGB format.")
            else:
                self._is_alpha = False
                return color_list
        elif all(length == 4 for length in length_list):
            color_list = self._normalize(color_list)
            try:
                color_list = [to_hex(item, keep_alpha=True) for item in color_list]
            except ValueError:
                raise ValueError("All elements in the list must be in RGBA format.")
            else:
                self._is_alpha = True
                return color_list
        else:
            raise ValueError("All items in the list need to be RGB or RGBA or Hex.")

    def _normalize(self, color_list: UniqueIterable) -> list[float]:
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
            Check the return type and convert it to a valid format.
        Args:
            return_type (str): 
                The format to convert the colors to. Options are 'hex', 'rgb', or 'rgba'.
        Returns:
            str:
                The converted return type.
        """
        if not isinstance(return_type, str):
            raise TypeError("return_type must be a string")
        return_type = return_type.lower()
        if return_type not in ['hex', 'rgb', 'rgba']:
            raise ValueError("return_type must be one of ['hex', 'rgb', 'rgba']")
        return return_type



class CustomCmap(object):
    def to_mpl_color_list(
        self, 
        color_list: UniqueIterable, 
        rgba: bool = True
    ) -> list[list[float, float, float]]:
        converter = Converter(color_list, return_type='rgba' if rgba else 'rgb')
        if rgba:
            return [to_rgba(c) for c in converter._color_list]
        else:
            return [to_rgb(c) for c in converter._color_list]
    
    def __create_position(self, colors: list[Any]) -> list[float]:
        length = len(colors)
        return [i / (length - 1) for i in range(length)]
    
    def __check_position(self, positions: list[float]) -> list[float]:
        positions[0] = 0.0
        positions[-1] = 1.0
        positions.sort()
        if all([0 <= p <= 1 for p in positions]):
            return positions
        return self.__create_position(positions)
    
    def color_list_to_linear_cmap(
        self, 
        color_list: UniqueIterable, 
        **kwargs
    ) -> Union['LinearColorMap', LinearSegmentedColormap]:
        positions = kwargs.get('positions', self.__create_position(color_list))
        positions = self.__check_position(positions)
        # カラーマップを作成
        color_list = [(position, color) for position, color in zip(positions, color_list)]
        name = kwargs.get('name', 'custom_cmap')
        cmap = LinearSegmentedColormap.from_list(name, color_list, N=256)
        return LinearColorMap(cmap)
    

class LinearColorMap(object):
    def __init__(self, cmap: LinearSegmentedColormap | ListedColormap):
        if isinstance(cmap, ListedColormap):
            colors = cmap(np.linspace(0, 1, 10))
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        if not isinstance(cmap, LinearSegmentedColormap):
            raise ValueError('cmap must be a LinearSegmentedColormap')
        self.cmap = cmap
    
    def __call__(self, position: int):
        return self.cmap(position)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.cmap, name)
    
    def get(self, position: int, return_type: str = 'rgba') -> Any:
        """
        ## Summary
            This method returns the color at the specified position of the created colormap
            in RGB, RGBA, Hex, int, or inta format.
        Args:
            position (int | Iterable[int]): 
                The position of the color in the color map.An array of integers or whole numbers.
            return_type (str): 
                The type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            Any:
                - rgb: Tuple of 3 float
                - rgba: Tuple of 4 float
                - str: Hexadecimal color code
                - int: Tuple of 3 integers
                - inta: Tuple of 4 integers
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
                    except TypeError:
                        raise TypeError(f'position must be int or Iterable[int], not {type(pos)}')
            return np.array(result)
    
    def _get(self, position: int, return_type: str = 'rgb') -> Any:
        transparency = False
        if position < 0 or 255 < position:
            # Return transparent color if out of range.
            transparency = True
        pattern = ['rgb', 'rgba', 'hex', 'int', 'inta']
        return_type = return_type.lower()
        if return_type not in pattern:
            # return_type must be one of pattern
            raise ValueError(f'return_type must be one of {pattern}')
        # Get the color at the given position
        color = (1., 1., 1., 0) if transparency else self.cmap(position)
        if return_type == 'rgb':
            # If RGB is specified.
            if len(color) == 4:
                return color[:-1]
            else:
                return color[:-1]
        elif return_type == 'rgba':
            # If RGBA is specified.
            if len(color) == 3:
                return color + (1.0,)
            else:
                return color
        elif return_type == 'hex':
            # If Hex is specified.
            return to_hex(color)
        elif return_type == 'int' and len(color) == 4:
            # If int is specified.
            if len(color) == 4:
                return tuple(int(c * 255) for c in color[:-1])
            else:
                return tuple(int(c * 255) for c in color)
        elif return_type == 'inta':
            # If inta is specified.
            if len(color) == 3:
                return tuple(int(c * 255) for c in color) + (255,)
            else:
                return tuple(int(c * 255) for c in color)

    def _get_in_list(self, positions: Iterable[int], return_type: str = 'rgb') -> UniqueIterable:
        """
        ## Summary
            Obtains a list of specified colors from the colormap and stores it in List.
        Args:
            positions (Iterable[int]): 
                List of color positions in the colormap.
            return_type (str): The 
                type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            List:
                - rgb: List of Tuple of 3 float
                - rgba: List of Tuple of 4 float
                - str: List of Hexadecimal color code
                - int: List of Tuple of 3 integers
                - inta: List of Tuple of 4 integers
        """
        return [self._get(pos, return_type) for pos in positions]

    def get_registered_color(self, return_type: str = 'rgb') -> UniqueIterable:
        """
        ## Summary
            Get all colors registered in cmap by List. 256 colors are stored in cmap.
        Args:
            return_type (str): 
                The type of the return value. Can be 'rgb' or 'rgba' or 'hex' or 'int' or 'inta'
        Returns:
            List:
                The length is 256, and each element is stored in one of the following formats
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

    def generate_idx_for_retrieval(self, 
        values: Iterable[float] | Iterable[int],
        in_nodata_value: Any=np.nan,
        out_nodata_index: int=-1
    ) -> Iterable[int]:
        """
        ## Summary
            Generate a colormap index from an array of values.
            This is used to retrieve a color from the colormap.
        Args:
            values (Iterable[float] | Iterable[int]): 
            An array of values to be converted to Index.
            This will be normalized to fall within the range 0-255,
            but 'in_nodata_value', nan and inf values will be replaced by 'out_nodata_index'.
            The array must be a 1D or 2D np.ndarray.
        Returns:
            Iterable[int]: 
                The index for retrieval
        Examples:
            >>> cmap = LinearColorMap(plt.get_cmap('viridis'))
            >>> values = np.random.normal(0, 1, 100).reshape(10, 10)
            >>> indices = cmap.generate_idx_for_retrieval(values)
        """
        if not isinstance(values, np.ndarray) and (2 <= dimensional_count(values) <= 3):
            values = np.array(values)
        if not isinstance(values, np.ndarray):
            raise ValueError('values must be an Iterable')
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

    def values_to_img(self, 
        values: Iterable[Iterable[float]], 
        nodata_value: Any = -1,
        return_type: str='inta'
    ) -> np.ndarray:
        """
        ## Summary
            Create an RGB image from an array of continuous values.
        Args:
            values (Iterable[Iterable[float]]): 
                Array of continuous values.
            return_type (str): 
                The type of the return value. Can be 'rgb' or 'rgba' or 'int' or 'inta'
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
        pattern = ['rgb', 'rgba', 'int', 'inta']
        return_type = return_type.lower()
        if return_type not in pattern:
            # return_type must be one of pattern
            raise ValueError(f'return_type must be one of {pattern}')
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
