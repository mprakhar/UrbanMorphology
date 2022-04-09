#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 2/15/2017
# Last edit 3/2/2017

# Purpose: To obtain use heights of strucuters and pair iwth NL to derive a probabilisitic identification of commercial, residential and industrial
# (1) : Using classified vector of Landsat built area, find heights
# (2) : reeample it to 750 m and pair with NL and plot NL vs Height - check if any relation
# (3) : find contrbutions of different classes within a 750m (1km )NLpixel.
# (4) : put it in a logistic regression mdoel (NL, LULC%ages, meanHeight, Heightstdev, objective variable is manualy identified residential, commerical and industrial pixel )


# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used:
'''# output filenames produced

'''


import numpy as np
import math as math
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import gdal
import os.path
from glob import glob
from datetime import timedelta, date
from dateutil import rrule
from PIL import Image
from dateutil.relativedelta import relativedelta
from matplotlib.path import Path
import matplotlib.patches as patches
from itertools import product
# from adjustText import adjust_text
from scipy.interpolate import griddata
from skimage.measure import block_reduce
from scipy.optimize import curve_fit

# pvt imports

import infoFinder as info
import shp_rstr_stat as srs
from classRaster import Raster_file
import im_mean_temporal as im_mean
import coord_translate as ct


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path
# path = r'/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon/Sample2//'

dnb = Raster_file()
dnb.path = r'/home/prakhar/Research/AQM_research/Data/Data_raw/VIIRS Composite/75N060E//'
dnb.sat = 'VIIRS'
dnb.prod = 'DNB'
dnb.sample = dnb.path + 'SVDNB_npp_20140201-20140228_75N060E_vcmslcfg_v10_c201507201053.avg_rade9.tif'
dnb.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//VIIRS_georef.tif'

# DSMpath = path + 'DSMsample2_rs10.tif'
# DSMspath = path + 'DSMsample2_rs10_G15_10.tif'
# land_class = r'/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/Classification/Yangon/greyscaleSample2.tif'
input_zone_polygon_0 = gb_path+'/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
nlpath = '/home/prakhar/Research/AQM_research/Data/Data_raw/VIIRS Composite/75N060E//'
nlpathout = gb_path + r'/Data/Data_process/VIIRS//'
AW3DDSMout = gb_path + r'/Data/Data_process/AW3D/India//'
ASTERDSMout = gb_path + r'/Data/Data_process/ASTER/India//'
# DSMpath = path + 'clipped2.tif'
# DSMspath = path + 'Gauss_20_15.tif'
#



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: ony urban nDSM     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# For ISRS focus on pairing with NL first
# resample height info, by mean etc, but also preserve the satdard deviation of heights

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


# function to get urban NDSM
def get_undsm(nDSMpath):

    # nDSM array
    nDSMarr = srs.raster_as_array(nDSMpath)

    # shape of nDSMpath
    shapeDSM = np.shape(nDSMarr)

    # Clip landsat as per nDSM
    lulcarr = clipraster(lulc, nDSMpath, shapeDSM)

    # getting the built structure only DSM = unDSM by consideing Class 2
    urbanclass = 1
    lulcarr[lulcarr!=urbanclass]= False
    lulcarr[lulcarr==urbanclass]= True
    unDSMarr =  nDSMarr*lulcarr

    return unDSMarr
# function end


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step2: Median NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# function to derive median NL image for the year ()2014)
def med_img(nlpath , year):

    ls = []
    for file in glob(os.path.join(nlpath, '*' + year + '*'+ '.tif')):

        # first cut big India from grlobal VIIRS
        imgarray, datamask = srs.zone_mask(input_zone_polygon_0, file)

        #append all images
        ls.append(imgarray)

    arr = np.array(ls)

    return np.median(arr, axis =0)
#function end


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: resampled urban nDSM and NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# function to read urban nDSM and return resmapled image of mean and std dev (should try to return other paramters that can represent better)
def resamp(unDSMarr):

    # resampling factor
    in_arr_res = 30
    out_arr_res = 450 #since it is 15 seconds
    xfactor = out_arr_res/in_arr_res
    yfactor = xfactor

    # resampling urban nDSM array mean and std dev
    # suggestion: use official GDAL resampling
    unDSMresmnarr = block_reduce(unDSMarr, block_size=(int(np.ceil(yfactor)), int(np.ceil(xfactor))), func=np.nanmean)
    unDSMressdarr = block_reduce(unDSMarr, block_size=(int(np.ceil(yfactor)), int(np.ceil(xfactor))), func=np.nanstd)
    unDSMressmarr = block_reduce(unDSMarr, block_size=(int(np.ceil(yfactor)), int(np.ceil(xfactor))), func=np.nansum)

    return [unDSMresmnarr, unDSMressdarr, unDSMressmarr]
#function end


# Function to gather info from urbanDSM values and clipped NL image values for plotting
def gather(unDSMpath, nlpath):

    # Reading urban nDSM
    unDSMarr =  srs.raster_as_array(unDSMpath)

    # Getting rid of pixels with elevation zero
    unDSMarr[unDSMarr==0] = np.nan

    # Resampling unDSM to VIIRS
    [unDSMresmnarr, unDSMressdarr, unDSMressmarr] =  resamp(unDSMarr)

    # nlarr = srs.raster_as_array(nlpath)

    # Clipping VIIRS as per resampled unDSM
    nl_arr = clipraster(nlpath, unDSMpath, np.shape(unDSMresmnarr))

    # covnert to pandas dataframe
    df_DSMNL = pd.DataFrame(unDSMresmnarr.flatten().tolist(), columns=['DSM_mn'])
    df_DSMNL['DSM_sd'] = unDSMressdarr.flatten().tolist()
    df_DSMNL['DSM_sm'] = unDSMressmarr.flatten().tolist()
    df_DSMNL['NL'] = nl_arr.flatten().tolist()

    return df_DSMNL
#function end



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step4: plotting urban nDSM and NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Check Hostogram
def plot_hist(df_DSMNL):

    # NL hist
    plt.subplot(121)
    plt.hist(df_DSMNL.NL ,50)    # NL<0.5 has the max pop..and doesnt seem to maek sense
    plt.xlabel('urban nDSM sum', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)

    # sum of heights
    plt.subplot(122)
    plt.hist(df_DSMNL.DSM_sm ,50)      # sumDSM<3 doenst seem to make sense in a 450*450m pixel
    plt.xlabel('urban NL ', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#Plots between NLvsDSM
def plot_NLDSM(df_DSMNL, thresh=True):



    if thresh==False:
        # plot withj mean
        plt.subplot(131)
        plt.scatter(df_DSMNL.DSM_mn, df_DSMNL.NL, alpha=0.5)
        plt.xlabel('urban nDSM mean (m)', fontsize = 20)
        plt.ylabel('Nightlight ($Watt/cm^2/sr$)', fontsize = 20)
        plt.xlim(xmin=0)
        plt.ylim(ymin =0)
        plt.title('NL and DSMmean')

        # plot withj std dev
        plt.subplot(132)
        plt.scatter(df_DSMNL.DSM_sd, df_DSMNL.NL, alpha=0.5)
        plt.xlabel('urban nDSM std dev (m)', fontsize = 20)
        plt.ylabel('Nightlight ($Watt/cm^2/sr$)', fontsize = 20)
        plt.xlim(xmin =0)
        plt.ylim(ymin =0)
        plt.title('NL and DSM stdedev')

        # plot withj sum
        plt.subplot(133)
        plt.scatter(df_DSMNL.DSM_sm, df_DSMNL.NL, alpha=0.5)
        plt.xlabel('urban nDSM sum (m)', fontsize = 20)
        plt.ylabel('Nightlight ($Watt/cm^2/sr$)', fontsize = 20)
        plt.xlim(xmin =0)
        plt.ylim(ymin =0)
        plt.title('NL and DSMsum')

    else:
        #after threshold
        # threshold of NL = 0.3, threshold of height = 3
        df_DSMNLthresh = df_DSMNL[(df_DSMNL.NL>=0.5) & (df_DSMNL.DSM_sm>=3) ]
        # plot withj mean
        plt.subplot(131)
        plt.scatter(df_DSMNLthresh.DSM_mn, df_DSMNLthresh.NL, alpha=0.5)
        plt.xlabel('urban nDSM mean (m)', fontsize = 20)
        plt.ylabel('Nightlight ($Watt/cm^2/sr$)', fontsize = 20)
        plt.title('NL and DSMmean')

        # plot withj std dev
        plt.subplot(132)
        plt.scatter(df_DSMNLthresh.DSM_sd, df_DSMNLthresh.NL, alpha=0.5)
        plt.xlabel('urban nDSM std dev (m)', fontsize = 20)
        plt.ylabel('Nightlight ($Watt/cm^2/sr$)', fontsize = 20)
        plt.title('NL and DSM stdedev')

        # plot withj sum
        plt.subplot(133)
        plt.scatter(df_DSMNLthresh.DSM_sm, df_DSMNLthresh.NL, alpha=0.5)
        plt.xlabel('urban nDSM sum (m)', fontsize = 20)
        plt.ylabel('Nightlight ($Watt/cm^2/sr$)', fontsize = 20)
        plt.title('NL and DSMsum')
# fucntion end


# --------------------  Run and plot and save ----------------------------

# Step1. Initialize
city = 'Kanpur'
AW3DnDSMpath = gb_path + r'/Data/Data_process/AW3D/India/AW3D_'+city+'_nDSM_v1.tif'
ASTERnDSMpath = gb_path + r'/Data/Data_process/ASTER/India/ASTER_'+city+'_nDSM_v1.tif'
lulc = gb_path + r'/Data/Data_process/Classification/NewDelhiL8class_v100.tif'


# Step2. Store the urban nDSM
#   ASWD
srs.arr_to_raster(get_undsm(AW3DnDSMpath), AW3DnDSMpath, AW3DDSMout+'AW3D_'+city+'_unDSM_v1.tif')
#   ASTER
srs.arr_to_raster(get_undsm(ASTERnDSMpath), ASTERnDSMpath, ASTERDSMout+'ASTER_'+city+'_unDSM_v1.tif')

# Step3. Find median nightlight fir the year
year = '2014'
med_nl = med_img(nlpath, year)
srs.arr_to_raster(med_nl, dnb.georef, nlpathout+ 'VIIRS'+year+'median.tif')

# Step4: Plot b/w NLa nd DSM
unDSMpath = ASTERDSMout+ 'ASTER_'+city+'_unDSM_v1.tif'
nlpathO = nlpathout+ '/VIIRS2014median.tif'
#   flatten arrays into DF to prepare for regression plot
df_DSMNL = gather(unDSMpath, nlpathO)
df_DSMNL[df_DSMNL['DSM_mn'].notnull()].to_csv('df_ASTERDSM_NL'+city+'.csv',index=True, header=True)
#    Plot
plot_hist(df_DSMNL)
plot_NLDSM(df_DSMNL, thresh=True)













