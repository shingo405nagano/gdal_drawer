import json
import subprocess
from typing import Any, Optional

import pyproj
import shapely


class GdalInfo(object):
    """
    ## Summary:
        "gdalinfo"コマンドのJSON出力をパースするクラス。通常`gdal`のpythonバインディングでは
        読み込んだメモリを解放するのが面倒なので、`subprocess`から"gdalinfo"を実行している。
        その為、このクラスはメモリ使用量を抑えたい場合に有効です。
    """

    def __init__(self, file_path):
        cmd = ["gdalinfo", "-json", file_path]
        self._result = subprocess.run(cmd, capture_output=True, text=True)
        if self._result.stderr != "":
            raise ValueError(self._result.stderr)

        self._info = json.loads(self._result.stdout)
        self.__size = self._info.get("size")

    @property
    def file_path(self) -> str:
        """
        ## Summary:
            読み込んだファイルパスを返す。
        """
        return self._info.get("description")

    @property
    def driver(self) -> str:
        """
        ## Summary:
            ドライバー名を返す。
        """
        return self._info.get("driverLongName")

    @property
    def x_size(self) -> Optional[int]:
        """
        ## Summary:
            x方向のピクセル数を返す。
        """
        if self.__size is None:
            return None
        elif len(self.__size) == 2:
            return self.__size[0]
        else:
            raise ValueError("``size``に値が入力されていない")

    @property
    def y_size(self) -> Optional[int]:
        """
        ## Summary:
            y方向のピクセル数を返す。
        """
        if self.__size is None:
            return None
        elif len(self.__size) == 2:
            return self.__size[1]
        else:
            raise ValueError("``size``に値が入力されていない")

    @property
    def geo_transform(self) -> list[float]:
        """
        ## Summary:
            transformパラメータを返す。gdalのバインディングだと"GetGeoTransform()"で取得できる。
        """
        return self._info.get("geoTransform")

    @property
    def x_resolution(self) -> float:
        """
        ## Summary:
            x方向の解像度を返す。
        """
        return self.geo_transform[1]

    @property
    def y_resolution(self) -> float:
        """
        ## Summary:
            y方向の解像度を返す。
        """
        return self.geo_transform[-1]

    @property
    def geo_scope(self) -> dict[str, float]:
        """
        ## Summary:
            画像の範囲を辞書形式で返す。
        """
        transform = self.geo_transform
        x_min = transform[0]
        y_max = transform[3]
        x_max = x_min + self.x_resolution * self.x_size  # type: ignore
        y_min = y_max + self.y_resolution * self.y_size  # type: ignore
        return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    def geo_scope_geometry(self, wgs84: bool = False) -> shapely.Polygon:
        """
        ## Summary:
            画像の範囲をshapelyのPolygon形式で返す。
            `wgs84=True`にするとWGS84座標系で返す。
        """
        if wgs84:
            poly_cds = self._info.get("wgs84Extent").get("coordinates")[0]
        else:
            poly_cds = []
            for key, val in self._info.get("cornerCoordinates").items():
                if "center" not in key:
                    poly_cds.append(val)
        return shapely.Polygon(poly_cds)

    @property
    def upper_left_corner(self) -> tuple[float]:
        """
        ## Summary:
            画像の左上隅の座標を返す。
        """
        return self._info.get("cornerCoordinates").get("upperLeft")

    @property
    def lower_left_corner(self) -> tuple[float]:
        """
        ## Summary:
            画像の左下隅の座標を返す。
        """
        return self._info.get("cornerCoordinates").get("lowerLeft")

    @property
    def lower_right_corner(self) -> tuple[float]:
        """
        ## Summary:
            画像の右下隅の座標を返す。
        """
        return self._info.get("cornerCoordinates").get("lowerRight")

    @property
    def upper_right_corner(self) -> tuple[float]:
        """
        ## Summary:
            画像の右上隅の座標を返す。
        """
        return self._info.get("cornerCoordinates").get("upperRight")

    @property
    def center(self) -> tuple[float]:
        """
        ## Summary:
            画像の中心座標を返す。
        """
        return self._info.get("cornerCoordinates").get("center")

    @property
    def bands(self) -> list[dict[str, Any]]:
        """
        ## Summary:
            バンド情報を辞書形式のリストで返す。
        """
        return self._info.get("bands")

    @property
    def dtypes(self) -> dict[str, str]:
        """
        ## Summary:
            バンドごとのデータ型を辞書形式で返す。
        """
        types = {}
        for band in self.bands:
            types[band["band"]] = band["type"]
        return types

    @property
    def nodata_values(self) -> dict[str, Any]:
        """
        ## Summary:
            バンドごとのNoData値を辞書形式で返す。
        """
        types = {}
        for band in self.bands:
            types[band["band"]] = band.get("noDataValue", "No setting")
        return types

    @property
    def crs(self) -> pyproj.CRS:
        """
        ## Summary:
            座標参照系をpyproj.CRS形式で返す。
        """
        wkt_crs = self._info.get("coordinateSystem").get("wkt")
        return pyproj.CRS(wkt_crs)

    @property
    def epsg(self) -> int:
        """
        ## Summary:
            EPSGコードを返す。
        """
        return self.crs.to_epsg()  # type: ignore
