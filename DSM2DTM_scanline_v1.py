#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 2/15/2017
# Last edit 1/20/2017

# Purpose: To obtain DTM and nDSM from DSM . Follows from algo in advanced DTM generation from very high resolution satellite stereo images
# (1) : Read Gauss smoothened image and generate hole DEM
# (2) : Fill holes using Krigging/TIN based interpolation and genrate proper DEM
# (3) : Genrate nDSM and built height estimation
# (4) : Using classified vector of Landsat built area, find heights
# (5) : reeample it to 750 m and pair with NL and plot NL vs Height


# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used: T -  target resolution to be achieved. usually MODIS, OMI image; S - source of the image to be resampled
'''# output filenames produced

'''

import numpy as np
import math as math
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os.path
from glob import glob
from datetime import timedelta, date
from dateutil import rrule
from PIL import Image
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
import matplotlib.patches as patches
from itertools import product
from adjustText import adjust_text
from scipy.interpolate import griddata

# pvt imports

import infoFinder as info
import shp_rstr_stat as srs
from classRaster import Raster_file
import im_mean_temporal as im_mean
import coord_translate as ct


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

path = r'/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon/Sample2//'
DSMpath = path + 'DSMsample2_rs10.tif'
DSMspath = path + 'DSMsample2_rs10_G15_10.tif'
land_class = r'/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/Classification/Yangon/greyscaleSample2.tif'
# DSMpath = path + 'clipped2.tif'
# DSMspath = path + 'Gauss_20_15.tif'
#


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 1: Holed DTM   * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# constraints/ thresholds
iExt = 15  # extent of filter window; it should be around 90meters depending on the resolution; aster - 5,
dThrHeightDiff = 3  # meter
dThrSlope = 30  # degrees

# 8 directions
scanlines = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
scannum = [0,1,2,3,4,5,6,7] # keyname for scanlines

# function to generate neghbors in the direction of scan line
def neighborhood(arr, dir, c0):

    dict_scannum = {
        0   : np.diag(arr),
        1   : arr[:,c0],
        2   : np.diag(np.fliplr(arr)),
        3   : np.fliplr(arr)[c0],
        4   : np.diag(arr)[::-1],
        5   : arr[:,c0][::-1],
        6   : np.diag(np.fliplr(arr))[::-1],
        7   : arr[c0]
    }

    return dict_scannum[dir]

# actual function for DTM generation
def DSM2DTM_scanline(DSM, DSMs, iExt, dThrHeightDiff, dThrSlope):

    # DSM is the DSM to betreated
    # DSMs smoothened DSM
    [m,n] = np.shape(DSM)

    # will store the output of each pixel for each scanline
    oLabel = np.zeros([8, m,n])

    # running over thewhole imahe
    for x0 in range(0 + (iExt - 1) / 2 ,m - (iExt - 1) / 2):
        for y0 in range(0 + (iExt - 1) / 2, n - (iExt - 1) / 2):

            # temporary subsetting the region around the pix
            c0 =  (iExt-1)/2
            oDSM = DSM[x0 - c0:x0 + c0,y0 - c0:y0 + c0]
            oDSMs = DSMs[x0 - c0:x0 + c0,y0 - c0:y0 + c0]

            # running for each scanline
            for scn in scannum:

                # scanline direction
                [iX, iY] = scanlines[scn]

                # local height difference
                oDSMDiff = oDSM[c0,c0] - DSM[x0 + iX,x0 + iY]

                # local terrain slope
                oDSMsDiff = oDSMs[c0,c0] - DSM[x0 + iX,x0 + iY]

                # get neighborhood(our filter extent)
                oNeigh = neighborhood(oDSM, scn, c0)

                #slope corrected height values
                oNeighCorr = oNeigh - (neighborhood(oDSMs, scn, c0) - oDSMs[c0, c0])

                # slope corrected minimal terrain value
                oMinNeigh = np.nanmin(oNeighCorr)

                # difference to minimum
                dHeightDiff = oDSM[c0, c0] - oMinNeigh

                if (dHeightDiff > dThrHeightDiff):
                    #pixel is non - ground (0)
                    oLabel[scn, x0, y0] = 0

                else :
                    # slope corrected height difference
                    dDelta = oDSMDiff - oDSMsDiff
                    dSignDelta = -np.sign(dDelta)
                    dSlopeLocal = math.atan2(abs(dDelta), 1) * 180 / np.pi

                    #slope corrected angle
                    dSlope = dSlopeLocal * dSignDelta

                    if (dSlope > dThrSlope):
                        # pixel is non - ground (0)
                        oLabel[scn, x0, y0] = 0

                    else:
                        # assign as last label
                        oLabel[scn, x0, y0] = oLabel[scn, x0 - iX, y0 - iY]

                    if (dSlope < 0):
                        #pixel is ground (1)
                        oLabel[scn, x0][y0] = 1

    # with file('oLabel_try.txt', 'w') as outfile:
    #     for slice in oLabel:
    #         np.savetxt(outfile, slice)

    return oLabel
# Function end

# Run function
DSM = srs.raster_as_array(DSMpath)
DSMs = srs.raster_as_array(DSMspath)
DSM[DSM==-9999] = np.nan
DSMs[DSMs==-9999] = np.nan
oLabel = DSM2DTM_scanline(DSM, DSMs, iExt, dThrHeightDiff, dThrSlope)

# Checking which pixels have sum of scanline direction >=5. If yes then ground
ground = DSM*(np.sum(oLabel, axis =0)>=5)

# save ground as a raster
# srs.arr_to_raster(ground, DSMpath, '//Urbanheights/DEMholes_try.tif')

# interpolating and filling holes
# from stack overflow different results for 2d interpolation with scipy.interpolate-gridddata
def fill_holes (f):
    mask = np.isnan(f)
    lx,ly = f.shape
    x, y = np.mgrid[0:lx, 0:ly]
    z = griddata(np.array([x[~mask].ravel() , y[~mask].ravel()]).T, f[~mask].ravel(), (x,y) , method = 'linear')

    return z

ground[ground==0] = np.nan
DEM = fill_holes(ground)

# plt.figure()
# plt.imshow(DEMarr)

# Rasterize the DEM array
srs.arr_to_raster(DEM, DSMpath, 'DEM_S2_rs10.tif')

# Generate nDSM
nDSM = DSM - DEM
nDSM[nDSM<=0] = 0

# visualize nDSM
plt.imshow(nDSM)

# Rasterize the nDSM array
srs.arr_to_raster(nDSM, DSMpath, 'nAstersample2.tif')

# Consider only built area
LC = srs.raster_as_array(land_class)










# a = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [4, 6, 3, 7, 9], [5, 3, 8, 6, 1], [7, 88, 33, 11, 44]]
# b = np.arange(25).reshape((5,5))
