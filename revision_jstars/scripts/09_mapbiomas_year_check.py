import sys, rasterio, numpy as np
from rasterio.warp import reproject, Resampling
with rasterio.open("data/mapbiomas/mapbiomas_todasclasses.tif") as b:
    T=b.read(1); tr,cr=b.transform,b.crs
for p in sys.argv[1:]:
    with rasterio.open(p) as s:
        D=np.zeros(T.shape,np.uint8); reproject(rasterio.band(s,1),D,dst_transform=tr,dst_crs=cr,resample=Resampling.nearest,dst_nodata=0)
    m=(T>0)&(D>0); print(p.split('/')[-1], "comparados",m.sum(),"concordância",round((T[m]==D[m]).mean(),4))
