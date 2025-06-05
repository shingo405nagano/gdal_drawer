from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np
import pyproj
import shapely

GEOMETRY = Union[shapely.geometry.base.GeometrySequence, str]

CRS = Union[int, str, pyproj.CRS]


class Bounds(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class XY(NamedTuple):
    x: float | list[float]
    y: float | list[float]


class CellSize(NamedTuple):
    x: float
    y: float


@dataclass
class Coordinates:
    """
    各セルの座標を格納するデータクラス
    X(np.ndarray): X座標の2次元配列
    Y(np.ndarray): Y座標の2次元配列
    """

    X: np.ndarray
    Y: np.ndarray


def crs_checker(index: int, kward: str) -> pyproj.CRS:
    """
    ## Summary:
        Check the CRS type and convert it to pyproj.CRS if necessary.
    Args:
        index (int): Index of the argument to check
        kward (str): Keyword argument name to check
    Returns:
        pyproj.CRS: CRS object
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            in_args = True
            crs = None
            if index < len(args):
                crs = args[index]
            elif kward in kwargs:
                crs = kwargs[kward]
                in_args = False

            if isinstance(crs, int):
                crs = pyproj.CRS.from_epsg(crs)
            elif isinstance(crs, str):
                try:
                    crs = pyproj.CRS.from_wkt(crs)
                except:
                    crs = pyproj.CRS.from_string(crs)
            elif not isinstance(crs, pyproj.CRS):
                raise TypeError(f"Invalid type for {kward}: {type(crs)}")

            if in_args:
                args = list(args)
                args[index] = crs
            else:
                kwargs[kward] = crs

            return func(*args, **kwargs)

        return wrapper

    return decorator


def geometry_checker(index: int, kward) -> shapely.geometry.base.BaseGeometry:
    """
    ## Summary:
        Check the geometry type and convert it to shapely.geometry if necessary.
    Args:
        index (int): Index of the argument to check
        kward (str): Keyword argument name to check
    Returns:
        shapely.geometry.base.BaseGeometry: Geometry object
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            in_args = True
            geom = None
            if index < len(args):
                geom = args[index]
            elif kward in kwargs:
                geom = kwargs[kward]
                in_args = False

            if isinstance(geom, str):
                geom = shapely.from_wkt(geom)
            elif not isinstance(geom, shapely.geometry.base.BaseGeometry):
                raise TypeError(f"Invalid type for {kward}: {type(geom)}")

            if in_args:
                args = list(args)
                args[index] = geom
            else:
                kwargs[kward] = geom

            return func(*args, **kwargs)

        return wrapper

    return decorator
