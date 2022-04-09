import rasterio as rio
import numpy as np
import os.path
from glob import glob


for file in glob(os.path.join("/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/GEE/20city classification/allclass/", '*' +  '.tif')):
    a = rio.open(file).read(1)
    a[a<1]=0
    a[a > 1] = 0
    print file, np.sum(a)



