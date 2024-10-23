class CustomGdalException(object):
    def shape_err(self, org_shape, in_shape) -> None:
        msg = (
            "The shape of the array "
            "must be the same as the raster size.\n"
            f"Original shape: {org_shape}\nIn shape: {in_shape}"
        )
        raise ValueError(msg)
    
    def unknown_crs_err(self) -> None:
        msg = (
            'CRS Error: Invalid projection: wkt: (Internal Proj Error:'
            'proj_create: unrecognized format / unknown name)'
        )
        raise ValueError(msg)
        
    def unknown_datum_err(self) -> None:
        msg = ('CRS Error: Invalid datum_name. WGS84 or JGD2011 ...')
        raise ValueError(msg)
    
    
custom_gdal_exception = CustomGdalException()

