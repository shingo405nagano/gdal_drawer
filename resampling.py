"""
# Summary
RasterDataのResamplingwを行うモジュール

-------------------------------------------------------------------------------
## 解像度を指定してRasterDataをResamplingする場合
>>> dst = gdal.Open('./raster.tif')
>>> resampled_dst = resampling_with_resol_spec(dst, x_resol=0.01, y_resol=0.01)

-------------------------------------------------------------------------------
## セル数を指定してRasterDataをResamplingする場合
>>> dst = gdal.Open('./raster.tif')
>>> resampled_dst = resampling_with_cells_spec(dst, x_cells=100, y_cells100)

-------------------------------------------------------------------------------
## メートル法で解像度を指定してRasterDataをResamplingする場合
これは、投影法がメートル法でない場合に、メートル法に変換してリサンプリングし、元の投影法に戻す。これは`EPSG:4326`の場合を想定しています。
>>> dst = gdal.Open('./raster.tif')
>>> resampled_dst = resampling_with_metre_resol_spec(
        dst, x_resol=0.01, y_resol=0.01, datum_name='JGD2011')

-------------------------------------------------------------------------------
"""
from osgeo import gdal

from projection import estimate_utm_crs
from projection import RasterProjection



class RasterResampling(RasterProjection):
    def __init__(self):
        super().__init__()

    def option_template_with_resol_spec(self,
        dst: gdal.Dataset,
        x_resolution: float,
        y_resolution: float,
        fmt: str='MEM',
    ) -> gdal.Dataset:
        """
        分解能を指定するgdal.WarpOptionsを生成する
        Args:
            dst(gdal.Dataset): ラスターデータ
            x_resolution(float): x方向の解像度
            y_resolution(float): y方向の解像度
            fmt(str): 出力形式。'MEM' or 'GTiff'
        Returns:
            (gdal.WarpOptions): リサンプリングオプション
        """
        return gdal.WarpOptions(
            format=fmt,
            xRes=x_resolution,
            yRes=y_resolution,
            resampleAlg=gdal.GRA_CubicSpline,
            outputBounds=self.dataset_bounds(dst)
        )

    def option_template_with_cells_spec(self,
        dst: gdal.Dataset,
        x_cells: int,
        y_cells: int,
        fmt: str='MEM',
    ) -> gdal.Dataset:
        """
        セル数を指定するgdal.WarpOptionsを生成する
        Args:
            dst(gdal.Dataset): ラスターデータ
            x_cells(int): x方向のセル数
            y_cells(int): y方向のセル数
            fmt(str): 出力形式。'MEM' or 'GTiff'
        Returns:
            (gdal.WarpOptions): リサンプリングオプション
        """
        return gdal.WarpOptions(
            format=fmt,
            width=x_cells,
            height=y_cells,
            resampleAlg=gdal.GRA_CubicSpline,
            outputBounds=self.dataset_bounds(dst)
        )
    


def resampling_with_resol_spec(
    dst: gdal.Dataset, 
    x_resol: float, 
    y_resol: float
) -> gdal.Dataset:
    """
    分解能を指定してラスターデータをリサンプリングする
    Args:
        dst(gdal.Dataset): ラスターデータ
        x_resol(float): x方向の解像度
        y_resol(float): y方向の解像度
    Returns:
        (gdal.Dataset): リサンプリング後のラスターデータ
    Examples:
        >>> dst = gdal.Open('./raster.tif')
        >>> resampled_dst = resampling_with_resol_spec(dst, x_resol=0.01, y_resol=0.01)
    """
    resampling = RasterResampling()
    ops = resampling.option_template_with_resol_spec(dst, x_resol, y_resol)
    return gdal.Warp('', dst, options=ops)


def resampling_with_cells_spec(
    dst: gdal.Dataset, 
    x_cells: int, 
    y_cells: int
) -> gdal.Dataset:
    """
    セル数を指定してラスターデータをリサンプリングする
    Args:
        dst(gdal.Dataset): ラスターデータ
        x_cells(int): x方向のセル数
        y_cells(int): y方向のセル数
    Returns:
        (gdal.Dataset): リサンプリング後のラスターデータ
    Examples:
        >>> dst = gdal.Open('./raster.tif')
        >>> resampled_dst = resampling_with_cells_spec(dst, x_cells=100, y_cells=100)
    """
    resampling = RasterResampling()
    ops = resampling.option_template_with_cells_spec(dst, x_cells, y_cells)
    return gdal.Warp('', dst, options=ops)


def resampling_with_metre_resol_spec(
    dst: gdal.Dataset,
    x_resol: float,
    y_resol: float,
    datum_name: str='WGS 84'
) -> gdal.Dataset:
    """
    分解能をメートルで指定してラスターデータをリサンプリングする
    Args:
        dst(gdal.Dataset): ラスターデータ
        x_resol(float): x方向の解像度
        y_resol(float): y方向の解像度
        datum_name: 'WGS 84', 'JGD2011' など
    Returns:
        (gdal.Dataset): リサンプリング後のラスターデータ
    Examples:
        >>> dst = gdal.Open('./raster.tif')
        >>> resampled_dst = resampling_with_metre_resol_spec(
                dst, x_resol=0.01, y_resol=0.01, datum_name='JGD2011')
    """
    projection = RasterProjection()
    # 投影法がメートル法かどうかを確認
    is_metre = projection.check_crs_is_metre(dst.GetProjection())
    if is_metre:
        # メートル法の場合はそのままリサンプリング
        return resampling_with_resol_spec(dst, x_resol, y_resol)
    else:
        # メートル法でない場合はメートル法に変換してリサンプリングし、元の投影法に戻す
        in_crs = dst.GetProjection()
        # 中心座標を取得
        in_bounds = projection.dataset_bounds(dst)
        x_mean = (in_bounds.x_min + in_bounds.x_max) / 2
        y_mean = (in_bounds.y_min + in_bounds.y_max) / 2
        # UTMの推定
        utm_crs = estimate_utm_crs(x_mean, y_mean, datum_name)
        # メートル法に変換し、リサンプリング
        metre_dst = projection.reprojection_raster(dst, utm_crs)
        rd_dst = resampling_with_resol_spec(metre_dst, x_resol, y_resol)
        # 元の投影法に戻す
        return projection.reprojection_raster(rd_dst, in_crs)
