from typing import Any

import numpy as np
from osgeo import gdal


class RasterCrop(object):
    def option_template_with_wkt_poly_spec(self,
        dst: gdal.Dataset,
        wkt_poly: str,
        fmt: str='MEM',
        nodata: Any=np.nan,
    ) -> gdal.Dataset:
        return gdal.WarpOptions(
            format=fmt,
            cutlineWKT=wkt_poly,
            cropToCutline=True,
            dstNodata=dst.GetNoDataValue(),
            srcNodata=nodata
        )


def crop_raster(
    dst: gdal.Dataset, 
    wkt_poly: str, 
    nodata: Any=np.nan
) -> gdal.Dataset:
    options = RasterCrop().option_template_with_wkt_poly_spec(
        dst, wkt_poly, nodata=nodata
    )
    return gdal.Warp('', dst, options=options)
