import os
import tempfile

import numpy as np
import pyproj
from osgeo import gdal

driver = gdal.GetDriverByName('GTiff')

_test_file_merc = os.path.join(
    os.path.dirname(__file__), 'data/TEST_DTM__R1_0__EPSG6691__IN_Nodata__SMALL.npy'
)
_test_float_ary_merc = np.load(_test_file_merc).astype(np.float32)
_test_float_ary_merc[_test_float_ary_merc <= -9999] = -9999.0
_transform_merc = (555864.83, 1.0, 0.0, 4405150.83, 0.0, -1.0)
_shape_merc = (1229, 2023)
TEMP_FILE_1D_MERCATOR = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
TEMP_DST_1D_MERCATOR = driver.Create(
    TEMP_FILE_1D_MERCATOR.name, _shape_merc[1], _shape_merc[0], 1, gdal.GDT_Float32
)
TEMP_DST_1D_MERCATOR.SetProjection(pyproj.CRS.from_epsg(6691).to_wkt())
TEMP_DST_1D_MERCATOR.SetGeoTransform(_transform_merc)
TEMP_DST_1D_MERCATOR.GetRasterBand(1).WriteArray(_test_float_ary_merc)
TEMP_DST_1D_MERCATOR.GetRasterBand(1).SetNoDataValue(-9999.0)
TEMP_DST_1D_MERCATOR.FlushCache()


_test_file_deg = os.path.join(
    os.path.dirname(__file__), 'data/TEST_DTM__R1_0__EPSG4326__SMALL.npy'
)
_test_float_ary_deg = np.load(_test_file_deg).astype(np.float32)
_test_float_ary_deg[_test_float_ary_deg <= -9999] = -9999.0
_transform_deg = (141.652397437, 0.00001115, 0.0, 39.794489711, 0.0, -0.00001115)
_shape_deg = (1018, 2157)
TEMP_FILE_1D_DEGREE = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
TEMP_DST_1D_DEGREE = driver.Create(
    TEMP_FILE_1D_DEGREE.name, _shape_deg[1], _shape_deg[0], 1, gdal.GDT_Float32
)
TEMP_DST_1D_DEGREE.SetProjection(pyproj.CRS.from_epsg(4326).to_wkt())
TEMP_DST_1D_DEGREE.SetGeoTransform(_transform_deg)
TEMP_DST_1D_DEGREE.GetRasterBand(1).WriteArray(_test_float_ary_deg)
TEMP_DST_1D_DEGREE.GetRasterBand(1).SetNoDataValue(-9999.0)
TEMP_DST_1D_DEGREE.FlushCache()


TEMP_FILE_3D_MERCATOR = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
TEMP_DST_3D_MERCATOR = driver.Create(
    TEMP_FILE_3D_MERCATOR.name, _shape_merc[1], _shape_merc[0], 3, gdal.GDT_Byte
)
TEMP_DST_3D_MERCATOR.SetProjection(pyproj.CRS.from_epsg(6691).to_wkt())

TEMP_DST_3D_MERCATOR.SetGeoTransform(_transform_merc)
for i in range(3):
    ary = np.random.randint(0, 255, size=_shape_merc).astype(np.uint8)
    band = TEMP_DST_3D_MERCATOR.GetRasterBand(i + 1)
    band.WriteArray(ary)
    band.SetNoDataValue(0)
    
TEMP_DST_3D_MERCATOR.FlushCache()