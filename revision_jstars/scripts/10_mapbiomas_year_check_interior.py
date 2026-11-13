import rasterio, numpy as np
from rasterio.warp import reproject, Resampling
U="data/mapbiomas/"
C={"c9 2020":U+"mapbiomas_c9_QF_2020.tif","c9 2021":U+"mapbiomas_c9_QF_2021.tif","c9 2022":U+"mapbiomas_c9_QF_2022.tif",
   "c8 2022":U+"mapbiomas_c8_QF_2022.tif","c9 2023":"data/mapbiomas/mapbiomas-brazil-collection-90-mg-2023.tif",
   "2025 (col. nova)":"data/mapbiomas/mapbiomas_2025.tif"}
with rasterio.open("data/mapbiomas/mapbiomas_todasclasses.tif") as b:
    T=b.read(1); tr,cr=b.transform,b.crs
# interior: 5x5 vizinhança toda da mesma classe em T (>0)
k=2; inter=(T>0)
for dy in range(-k,k+1):
    for dx in range(-k,k+1):
        inter &= (np.roll(np.roll(T,dy,0),dx,1)==T)
inter[:k]=inter[-k:]=False; inter[:,:k]=inter[:,-k:]=False
print("pixels interiores:",inter.sum(),"de",(T>0).sum())
for n,p in C.items():
    with rasterio.open(p) as s:
        D=np.zeros(T.shape,np.uint8); reproject(rasterio.band(s,1),D,dst_transform=tr,dst_crs=cr,resample=Resampling.nearest,dst_nodata=0)
    m=inter&(D>0); 
    print(f"{n:18s} interior {np.mean(T[m]==D[m]):.4f} (n={m.sum()})   total {np.mean(T[(T>0)&(D>0)]==D[(T>0)&(D>0)]):.4f}")
