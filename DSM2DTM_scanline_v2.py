#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 2/15/2017
# Last edit 3/2/2017

# Purpose: To obtain DTM and nDSM from DSM . Follows from algo in advanced DTM generation from very high resolution satellite stereo images
# (1) : Read Gauss smoothened image and generate hole DEM
# (2) : Fill holes using Krigging/TIN based interpolation and genrate proper DEM
# (3) : Genrate nDSM and built height estimation
# (4) : Using classified vector of Landsat built area, find heights
# (5) : reeample it to 750 m and pair with NL and plot NL vs Height


# Location of output: E:\Acads\Research\AQM\Data process\CSVOut; mostly in the Urban3D folder and Daraprocessed folder

# terminology used:
'''# output filenames produced

'''

import numpy as np
import math as math
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os.path
from glob import glob
from scipy.interpolate import griddata
import cv2 as cv
from sklearn.metrics import mean_squared_error
from scipy import stats


# pvt imports
import shp_rstr_stat as srs
from classRaster import Raster_file
import coord_translate as ct


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# path = r'/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon/Sample2//'
# DSMpath = path + 'DSMsample2_rs30.tif'
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path
AW3Dpath = gb_path + r'/Data/Data_process/AW3D/India//' + r'AW3D_NewDelhi_DSM_v1.tif'
ASTERpath = gb_path + r'/Data/Data_process/ASTER/India//' + r'ASTER_NewDelhi_DSM_v1.tif'
city = 'NewDelhi'
prod = 'ASTER'
# DSMpath = path + 'clipped2.tif'
# DSMspath = path + 'Gauss_20_15.tif'
#



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 1: Holed DTM   * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# constraints/ thresholds
resolution = 30
Ext = 300    # Extent of neighbors in metres: for 10m _ 200; for 30m - 3000
iExt = np.int(Ext/(2*resolution))*2 + 1  # extent of filter window; it should be around 90meters depending on the resolution; aster - 5, for 10m - 15
dThrHeightDiff = 3  # meter
dThrSlope = 60  # degrees using 60 degress for 30m as difficult to identify ground terrain otherwise

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
    print ' Entered DSM2DTM scanline'
    # DSM is the DSM to betreated
    # DSMs smoothened DSM

    # finding shape
    [m,n] = np.shape(DSM)

    #3 dim array with 8 2D bands will store the output of each pixel for each scanline
    oLabel = np.zeros([8, m,n])

    # running over thewhole imahe
    for x0 in range(0 + (iExt - 1) / 2 ,m - (iExt - 1) / 2):
        for y0 in range(0 + (iExt - 1) / 2, n - (iExt - 1) / 2):

            # temporary subsetting the region around the pix
            c0 =  (iExt-1)/2
            oDSM = DSM[x0 - c0:x0 + c0 + 1,y0 - c0:y0 + c0 + 1]
            oDSMs = DSMs[x0 - c0:x0 + c0 + 1,y0 - c0:y0 + c0 + 1]

            # running for each scanline
            for scn in scannum:

                # scanline direction
                [iX, iY] = scanlines[scn]

                # local height difference
                oDSMDiff = oDSM[c0,c0] - DSM[x0 + iX,y0 + iY]

                # local terrain slope
                oDSMsDiff = oDSMs[c0,c0] - DSMs[x0 + iX,y0 + iY]

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
                    dSlopeLocal = math.atan2(abs(dDelta), resolution) * 180 / np.pi

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

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 2: DEM   * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Dunction to fill holes
def fill_holes(f):
    print 'filling holes'
    # interpolating and filling holes
    # from stack overflow different results for 2d interpolation with scipy.interpolate-gridddata
    # http://stackoverflow.com/questions/40449024/different-results-for-2d-interpolation-with-scipy-interpolate-griddata

    # make mask of all values to be filled
    mask = np.isnan(f)

    # final shape
    lx, ly = f.shape
    x, y = np.mgrid[0:lx, 0:ly]

    # 'Fill it'
    z = griddata(np.array([x[~mask].ravel(), y[~mask].ravel()]).T, f[~mask].ravel(), (x, y), method='linear')

    return z


# Master unction to to run everything and get oytput

def ground(prod, DSMpath, city):

    # Convert raster to array
    DSM = srs.raster_as_array(DSMpath)

    # Gaussian blurre image
    DSMs = cv.GaussianBlur(src = DSM, ksize = (2*int(100/(2*resolution))+1, 2*int(100/(2*resolution))+1), sigmaX = 25, sigmaY = 25)

    # remove all -9999 values as nan
    DSM[DSM==-9999] = np.nan
    DSMs[DSMs==-9999] = np.nan

    # finally run function to find DSM from DEM
    oLabel = DSM2DTM_scanline(DSM, DSMs, iExt, dThrHeightDiff, dThrSlope)

    # Now Checking which pixels have sum of scanline direction >=5. If yes then ground
    ground = DSM*(np.sum(oLabel, axis =0)>=5)

    # save ground as a raster
    # srs.arr_to_raster(ground, DSMpath, '//Urbanheights/DEMholes_try.tif')

    ground[ground==0] = np.nan
    DEM = fill_holes(ground)
    DEM = cv.GaussianBlur(src = DEM, ksize = (5,5), sigmaX = 5, sigmaY = 5)

    # plt.figure()
    # plt.imshow(DEMarr)

    # Rasterize the DEM array
    srs.arr_to_raster(DEM, DSMpath, prod+'_'+city+'_DEM'+'_v1.tif')

    # Generate nDSM
    nDSM = DSM - DEM

    # not sure if this is correct but coverting all <0 pixels to ground
    nDSM[nDSM<=0] = 0

    # visualize nDSM
    plt.imshow(nDSM)

    # Rasterize the nDSM array
    srs.arr_to_raster(nDSM, DSMpath, prod+'_'+city+'_nDSM'+'_v1.tif')

    print 'job done ', prod

# fucntion end

ground('AW3D', AW3Dpath, city)
ground('ASTER', ASTERpath, city )



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 5: Calculate statistics  * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#


# to comapre statistics
def stat_compare(df_DSMcomp):

    # removing all nan values
    df_DSMcomp = df_DSMcomp[df_DSMcomp.ASTER.notnull() & df_DSMcomp.AW3D.notnull()]
    # Statistics
    # print mean, std. min, 25. 50, 75, max
    print ' summary '
    print df_DSMcomp.describe()

    # kurtsis
    print 'Kurtosis'
    print df_DSMcomp.kurtosis()

    # skewness
    print 'skewness'
    print df_DSMcomp.skew()

    # correlation
    print ' Pearson correlation '
    print df_DSMcomp.corr()

    # RMSE
    print 'RMSE'
    print np.sqrt(mean_squared_error(df_DSMcomp.AW3D, df_DSMcomp.ASTER))

    # t-test; assumes variances are equal
    print 't - test'
    print stats.ttest_ind(df_DSMcomp.AW3D,df_DSMcomp.ASTER )

    # F-test
    print 'F-test'



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 3: ASTER and AW3D DEM compare  * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# need to sort issues with identifyoing urban in NCR
def classL8_NCR():

    # open classification
    lulc1 = srs.raster_as_array(gb_path + r'/Data/Data_process/Classification/NewDelhiL8class_v1.tif')
    lulc2 = srs.raster_as_array(gb_path + r'/Data/Data_process/Classification/NewDelhiL8class_v2.tif')
    lulc3 = srs.raster_as_array(gb_path + r'/Data/Data_process/Classification/NewDelhiL8class_v3.tif')

    # create new classification merged layer
    lulc = np.zeros(np.shape(lulc1))

    # merging brick-kilns from first layer lulc1 with urban from lulc2 with disjoint suburban from lulc2 and lulc3
    lulc = (lulc1==8) + (lulc2 ==2) + ((lulc2==3) * (lulc3==4))

    # save file
    srs.arr_to_raster(lulc, gb_path + r'/Data/Data_process/Classification/NewDelhiL8class_v1.tif', gb_path + r'/Data/Data_process/Classification/NewDelhiL8class_v100.tif')
# fucntion end


# function to clip the DSM boundaried data from night light
def clipraster(raster_base, raster_shape, shape):

    # raster_base: whoch needs to clipped; its path address
    # raster_shape: the shape which will be extracted out; its path address
    # shape:shape to be clipped ; get np.shape(raster_shape); has been put in case the cell size of raster_shape and shape is different

    # finding size of raster shape
    # b =  gdal.Open(raster_shape)
    # c = b.GetRasterBand(1)
    # shape = np.shape(c.ReadAsArray().astype(np.float))
    # shape = np.shape(unDSMresmnarr)

    #finding lat long of pixel of raster_shape
    [[lat,long]] = ct.pixelToLatLon(raster_shape, [[0,0]])

    # fidning what pixel of the raster base does the latlong obtained from raster shape correspond
    [pix] = ct.latLonToPixel(raster_base,[[lat,long]] )

    # carve out the raster_shape from base starting coordinates 'pix'
    clippedarr = srs.raster_as_array(raster_base)[pix[1]: (pix[1] + shape[0]), pix[0]: (pix[0] + shape[1])]

    # still need to apply the mask(

    return clippedarr
# fucntion end

# Plotting function
def plot_scat(df_DSMcomp):

    # find plotting limits
    lim_l = int(min(df_DSMcomp['ASTER'].min(), df_DSMcomp['AW3D'].min()))
    lim_u = int(max(df_DSMcomp['ASTER'].max(), df_DSMcomp['AW3D'].max()))

    # all pixels
    plt.figure()
    plt.scatter(df_DSMcomp['ASTER'], df_DSMcomp['AW3D'], alpha=0.5)
    plt.xlabel('ASTER elevation (m)', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize=15)
    plt.ylabel('AW3D elevation (m)', fontsize = 20)
    plt.xlim([lim_l, lim_u])
    plt.ylim([lim_l, lim_u])
    # plt.title('ASTER and AW3D DEM')

    # only urban pixels
    plt.figure()
    plt.scatter(df_DSMcomp[df_DSMcomp.lulc == 1]['ASTER'], df_DSMcomp[df_DSMcomp.lulc == 1]['AW3D'], alpha=0.5)
    plt.xlabel('ASTER elevation(m)', fontsize = 20)
    plt.ylabel('AW3D elevation (m)', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize=15)
    plt.xlim([lim_l, lim_u])
    plt.ylim([lim_l, lim_u])
    # plt.title('ASTER and AW3D DEM')
# function end




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 3: Run functions  * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
city = 'Kanpur'

# DEM
# set paths
AW3Dpath = gb_path + r'/Data/Data_process/AW3D/India//' + r'AW3D_'+city+'_DEM_v1.tif'
ASTERpath = gb_path + r'/Data/Data_process/ASTER/India//' + r'ASTER_'+city+'_DEM_v1.tif'
# lulc = gb_path + r'/Data/Data_process/Classification/lkokprClass.tif'
lulc = gb_path + r'/Data/Data_process/Classification/lkokprClass.tif'

# read all DEMs images
AW3Darr = srs.raster_as_array(AW3Dpath)
ASTERarr = srs.raster_as_array(ASTERpath)
lulcarr = clipraster(lulc, AW3Dpath, np.shape(AW3Darr))

# save the difference DEM between ASTER and AW3D
srs.arr_to_raster((ASTERarr - AW3Darr), AW3Dpath, 'diffDEM_'+ city + '.tif' )

# read images as dataframe
df_DSMcomp = pd.DataFrame(AW3Darr.flatten().tolist(), columns=['AW3D'])
df_DSMcomp['ASTER'] = ASTERarr.flatten().tolist()
df_DSMcomp['lulc'] = lulcarr.flatten().tolist()

#plot scatter
plot_scat(df_DSMcomp)         # NCR `150,350//

# compare stats
stat_compare(df_DSMcomp)
# only urban
stat_compare(df_DSMcomp[df_DSMcomp.lulc==1])




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * #   * *#    Step 4: ASTER and AW3D nDSM compare  * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

AW3Dpath = gb_path + r'/Data/Data_process/AW3D/India//' + r'AW3D_'+city+'_nDSM_v1.tif'
ASTERpath = gb_path + r'/Data/Data_process/ASTER/India//' + r'ASTER_'+city+'_nDSM_v1.tif'
lulc = gb_path + r'/Data/Data_process/Classification/lkokprClass.tif'

AW3Darr = srs.raster_as_array(AW3Dpath)
ASTERarr = srs.raster_as_array(ASTERpath)
lulcarr = clipraster(lulc, AW3Dpath, np.shape(AW3Darr))

df_DSMcomp = pd.DataFrame(AW3Darr.flatten().tolist(), columns=['AW3D'])
df_DSMcomp['ASTER'] = ASTERarr.flatten().tolist()
df_DSMcomp['lulc'] = lulcarr.flatten().tolist()

plot_scat(df_DSMcomp)

# compare stats
stat_compare(df_DSMcomp)
# only urban
stat_compare(df_DSMcomp[df_DSMcomp.lulc==1])







# a = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [4, 6, 3, 7, 9], [5, 3, 8, 6, 1], [7, 88, 33, 11, 44]]
# b = np.arange(25).reshape((5,5))

# Experimenting with Gaussian values

DSMs = (cv.GaussianBlur(src = DSM, ksize = (25,25), sigmaX = 25, sigmaY = 25))
plt.figure()
plt.imshow(DSMs)
srs.arr_to_raster(DSMs, DSMpath, 'DSM10Gauss25_25.tif')



kernel = (cv.getGaussianKernel(ksize=5, sigma=1)*np.transpose(cv.getGaussianKernel(ksize=5, sigma=1)))