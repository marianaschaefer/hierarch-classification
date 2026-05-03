import rasterio
import numpy as np

path = 'data_raw/external/mosaic_clip_composite_proj10m_abril.tif'
with rasterio.open(path) as src:
    band1 = src.read(1)
    nodata_val = src.nodata
    total_pixels = band1.size
    nodata_count = np.sum(band1 == nodata_val)
    percent = (nodata_count / total_pixels) * 100
    
    print(f"Arquivo: {path}")
    print(f"Valor de NoData oficial: {nodata_val}")
    print(f"Pixels Totais: {total_pixels}")
    print(f"Pixels NoData: {nodata_count} ({percent:.2f}%)")