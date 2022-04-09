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

# pvt imports

import infoFinder as info
import shp_rstr_stat as srs
from classRaster import Raster_file
import im_mean_temporal as im_mean
import coord_translate as ct


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

path = '/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon//'
oDSM = path + 'DSMclipsample.tif'
oDSMs = path + 'DSMgauss20_15.tif'



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 1: Holed DTM   * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

iExt = 9
dThrHeightDiff = 3
dThrSlope = 30

# 8 directions
scanlines = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]

def DSM2DTM_scanline(DSM, DSMs, iDir, iExt, dThrHeightDiff, dThrSlope):

    [m,n] = np.shape(oDSM)
    oLabel = np.zeros([m,n])

    # running over thewhole imahe
    for x0 in range(0 - (iExt - 1) / 2 ,m + (iExt - 1) / 2):
        for y0 in range(0 - (iExt - 1) / 2, n + (iExt - 1) / 2):

            c0 =  (iExt-1)/2
            oDSM = DSM[x0 - c0:x0 + c0][y0 - c0:y0 + c0]
            oDSMs = DSMs[x0 - c0:x0 + c0][y0 - c0:y0 + c0]





            # running for each scanline
            for scn in scanlines:

                # scanline direction
                [iX, iY] = scn

                # local height difference
                oDSMDiff = oDSM(x, y) - oDSM(x + iX, y + iY)

                # local terrain slope
                oDSMsDiff = oDSMs(x, y) - oDSM(x + iX, y + iY)

                # get neighborhood(our filter extent)
                oNeigh = oDSM(x + oExtX(iDir), y + oExtY(iDir))

                #slope corrected height values
                oNeighCorr = oNeigh + X * oDSMs(x, y)

                # slope corrected minimal terrain value
                oMinNeigh = min(oNeighCorr)

                # difference to minimum
                dHeightDiff = oDSM(x, y) - oMinNeigh

                if (dHeightDiff > dThrHeightDiff):
                    #pixel is non - ground (0)
                    oLabel[x][y] = 0

                else :
                    # slope corrected height difference
                    dDelta = oDSMDiff - oDSMsDiff
                    dSignDelta = -np.sign(dDelta)
                    dSlopeLocal = math.atan2(abs(dDelta), 1) * 180 / np.pi

                    #slope corrected angle
                    dSlope = dSlopeLocal * dSignDelta

                    if (dSlope > dThrSlope):
                        # pixel is non - ground (0)
                        oLabel[x][y] = 0

                    else:
                        # assign as last label
                        oLabel[x][y] = oLabel[x - iX][y - iY]

                    if (dSlope < 0):
                        #pixel is ground (1)
                        oLabel[x][y] = 1


a = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [4, 6, 3, 7, 9], [5, 3, 8, 6, 1], [7, 88, 33, 11, 44]]
b = np.arange(16).reshape((4,4))