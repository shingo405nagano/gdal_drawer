class CustomGdalException(object):
    def not_gdal_dataset_err(self) -> None:
        msg = 'The argument must be a gdal.Dataset object.'
        raise ValueError(msg)

    def shape_err(self, org_shape, in_shape) -> None:
        msg = (
            "The shape of the array "
            "must be the same as the raster size.\n"
            f"Original shape: {org_shape}\nIn shape: {in_shape}"
        )
        raise ValueError(msg)

    def not_have_crs_err(self) -> None:
        msg = 'The dataset does not have a CRS.'
        raise ValueError(msg)

    def unknown_crs_err(self) -> None:
        msg = (
            'CRS Error: Invalid projection: wkt: (Internal Proj Error:'
            'proj_create: unrecognized format / unknown name)'
        )
        raise ValueError(msg)
        
    def unknown_datum_err(self) -> None:
        msg = 'CRS Error: Invalid datum_name. WGS84 or JGD2011 ...'
        raise ValueError(msg)
    
    def get_band_err(self) -> None:
        msg = 'The argument must be an integer or an iterable of integers.'
        raise ValueError(msg)

    def get_band_number_err(self) -> None:
        msg = 'The argument must be an iterable of integers.'
        raise ValueError(msg)
    
    def load_wkt_geometry_err(self) -> None:
        msg = 'The argument must be a WKT string or a shapely.geometry object.'
        raise ValueError(msg)
    
    def band_count_err(self) -> None:
        msg = 'The number of bands in the dataset does not match the specified number of bands.'
        raise ValueError(msg)
    
custom_gdal_exception = CustomGdalException()

