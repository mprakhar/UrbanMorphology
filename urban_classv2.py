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
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.measure import block_reduce
from scipy.optimize import curve_fit
from scipy import interpolate
import numpy.ma as ma

# pvt imports

import shp_rstr_stat as srs
from classRaster import Raster_file
import im_mean_temporal as im_mean
import coord_translate as ct
import my_math as mth

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
AW3Dout = gb_path + r'/Data/Data_process/AW3D/India//'
ASTERout = gb_path + r'/Data/Data_process/ASTER/India//'
# DSMpath = path + 'clipped2.tif'
# DSMspath = path + 'Gauss_20_15.tif'
#



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: ony urban nDSM     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# For ISRS focus on pairing with NL first
# resample height info, by mean etc, but also preserve the satdard deviation of heights

# function to clip the DSM boundaried data from night light
def clipraster(raster_base, raster_shape, shape):
    # Assuming both already have same resolution
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
def get_undsm(nDSMpath, lulcpath):

    # nDSM array
    nDSMarr = srs.raster_as_array(nDSMpath)

    # shape of nDSMpath
    shapeDSM = np.shape(nDSMarr)

    # Clip landsat as per nDSM
    lulcarr = clipraster(lulcpath, nDSMpath, shapeDSM)

    # getting the built structure only DSM = unDSM by consideing Class 2
    urbanclass = 1
    lulcarr[lulcarr!=urbanclass]= False
    lulcarr[lulcarr==urbanclass]= True
    unDSMarr =  nDSMarr*lulcarr

    return unDSMarr
# function end


#function end


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: resampled urban nDSM and NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# function to read urban nDSM and return resmapled image of mean and std dev (should try to return other paramters that can represent better)
def downsampDSM(unDSMarr):

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

# function to upsample NL images to the resolution of DSM
def upsampNL(nl_arr):

    # creating resmapling object
    col = np.shape(nl_arr)[1]
    row = np.shape(nl_arr)[0]
    f = interpolate.interp2d(range(col), range(row), nl_arr, kind='linear')

    # resampling factor
    in_arr_res = 450
    out_arr_res = 30
    xfactor = in_arr_res / out_arr_res

    # creating new image size
    colnew = np.linspace(0,col, col*xfactor)
    rownew = np.linspace(0, row, row * xfactor)
    res_nl_arr = f(colnew, rownew)

    return res_nl_arr
# funciton end


# Function to gather info from urbanDSM values and clipped NL image values for plotting. downsamp refers to DSM being downsampled
def gather(unDSMpath, nlpath, downsamp = False):

    # Reading urban nDSM
    unDSMarr =  srs.raster_as_array(unDSMpath)

    # Getting rid of pixels with elevation zero
    unDSMarr[unDSMarr==0] = np.nan

    # downsampling DSM data
    if downsamp == True:
        # Resampling unDSM to VIIRS
        [unDSMresmnarr, unDSMressdarr, unDSMressmarr] = downsampDSM(unDSMarr)

        # Clipping the VIIRS data as per unDSM
        nl_arr = clipraster(nlpath, unDSMpath, np.shape(unDSMresmnarr))

        # covnert to pandas dataframe
        df_DSMNL = pd.DataFrame(unDSMresmnarr.flatten().tolist(), columns=['DSM_mn'])
        df_DSMNL['DSM_sd'] = unDSMressdarr.flatten().tolist()
        df_DSMNL['DSM_sm'] = unDSMressmarr.flatten().tolist()
        df_DSMNL['NL'] = nl_arr.flatten().tolist()

    # non need to downsample DSM. instead NL has bveen upsampled
    if downsamp == False:

        # keeping DSM as it is
        unDSMresmnarr = unDSMarr

        # Clipping the VIIRS data as per unDSM
        nl_arr = clipraster(nlpath, unDSMpath, np.shape(unDSMresmnarr))

        # covnert to pandas dataframe
        df_DSMNL = pd.DataFrame(unDSMresmnarr.flatten().tolist(), columns=['DSM_mn'])
        df_DSMNL['NL'] = nl_arr.flatten().tolist()

    return df_DSMNL
#function end


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step4: plotting urban nDSM and NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Check Hostogram
def plot_hist(df_DSMNL):

    # NL hist
    plt.subplot(121)
    plt.hist(df_DSMNL[df_DSMNL.NL>=10]['NL'] ,20)    # NL<0.5 has the max pop..and doesnt seem to maek sense
    plt.xlabel('urban NL', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)

    # sum of heights
    plt.subplot(122)
    plt.hist(df_DSMNL.DSM_mn ,50)      # sumDSM<3 doenst seem to make sense in a 450*450m pixel
    plt.xlabel('urban DSM ', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)
# function end


# fucntion to plot trend line and scatter
def plotter(subplt, x, y, xlabelstr, functype='log'):

    plt.subplot(subplt)

    # plot points
    plt.plot(x, y, 'ko', label="Data", alpha=0.5)

    # decide which kind of trendline and plot it
    if functype=='log':
        popt, pcov = curve_fit(mth.funclog, x, y)
        plt.plot(x, mth.funclog(x, *popt), 'r-', label=" $NL$ = " + str('%.02f'%popt[0]) + 'ln($h$) + ' + str('%.02f'%popt[1]))
        #logx = np.log(x)
        #res = sm.OLS(y, logx).fit()
        #print('Parameters: ', res.params)
        #print('Standard errors: ', res.bse)
        #prstd, iv_l, iv_u = wls_prediction_std(res)
        #plt.plot(logx, iv_u, 'r--')
        #plt.plot(logx, iv_l, 'r--')


    if functype=='linear':
        popt, pcov = curve_fit(mth.funclin, x, y)
        plt.plot(x, mth.funclin(x, *popt), 'r-', label=" $NL$ = " + str('%.02f'%popt[0]) + '$h$ + ' + str('%.02f'%popt[1]))

    # add labels
    plt.xlabel('$h$ '+xlabelstr+' ($m$)', fontsize=20)
    plt.ylabel('$NL$ ($Watt/cm^2/sr$)', fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()
    plt.title('Nightlight(NL) and DSM(h)'+xlabelstr)

    return popt
# function end

#Plots between NLvsDSM
def plot_NLDSM(df_DSMNL, thresh=True, downsamp=False):

    df_DSMNL = df_DSMNL.sort_values(['DSM_mn'])

    if thresh==False:
        # plot withj mean
        x = df_DSMNL[(df_DSMNL.NL >= 3) & (df_DSMNL.DSM_mn >= 1)]['DSM_mn']
        y = df_DSMNL[(df_DSMNL.NL >= 3) & (df_DSMNL.DSM_mn >= 1)]['NL']

        # plot mean
        plotter(131, x, y, '', functype='log')

        # if height has been downsampled to match NLs resolution
        if downsamp:
            # plot withj std dev
            x = df_DSMNL[(df_DSMNL.NL >= 3) & (df_DSMNL.DSM_sd >= 1)]['DSM_sd']
            plotter(132, x, y, 'stddev', functype='log')

            # plot withj sum
            x = df_DSMNL[(df_DSMNL.NL >= 3) & (df_DSMNL.DSM_sm >= 1)]['DSM_sd']
            plotter(133, x, y, 'sum', functype='log')

    if thresh:
        #after threshold
        # threshold of NL = 0.3, threshold of height = 3
        # plot withj mean
        x = df_DSMNL[(df_DSMNL.NL >= 15) & (df_DSMNL.DSM_mn >= 1)]['DSM_mn']    # for AW3D thresh set as 1 to remove noise heights
        y = df_DSMNL[(df_DSMNL.NL >= 15) & (df_DSMNL.DSM_mn >= 1)]['NL']

        # plot mean
        plotter(131, x, y, '', functype='log')

        if downsamp:

            # plot withj std dev
            x = df_DSMNL[(df_DSMNL.NL >= 15) & (df_DSMNL.DSM_sd >= 1)]['DSM_sd']
            plotter(132, x, y, 'stddev', functype='log')

            # plot withj sum
            x = df_DSMNL[(df_DSMNL.NL >= 15) & (df_DSMNL.DSM_sm >= 1)]['DSM_sd']
            plotter(133, x, y, 'sum', functype='log')
# fucntion end

#function to implelement rule on the pixel classify
def rules(nl_arr, DSMarr, lumparr):

    # first creating boolean array using comditions. 1 means condition fulfilled, 0 otherwise

    # height binary image
    height_thresh = 4
    bin_DSMarrgt = ma.masked_greater(DSMarr,height_thresh)
    DSMarr2 = DSMarr
    DSMarr2[DSMarr2 <= 1] = np.nan
    bin_DSMarrlt = ma.masked(DSMarr2, height_thresh)

    # lump binary image
    # Considering only values >= for lump
    bin_lumparr = ma.masked_greater_equal(lumparr, 2)

    # NL binary image
    bin_nl_arr = ma.masked_greater(nl_arr - (3.16*np.log(DSMarr) + 16.62), -1)

    bin_nl_arr2 = ma.masked_greater(nl_arr, 2)


    # make boolean mask. ensure you areading the mask only.
    # for NL pass values through the log function to check whether greater or less than the expected value

    # residential
    res = np.logical_and(bin_nl_arr2.mask, bin_DSMarrlt.mask)

    # commercial
    com = np.logical_and(bin_lumparr.mask, np.logical_and(bin_nl_arr.mask, bin_DSMarrgt.mask))
    # industrial
    ind = np.logical_or(
                        np.logical_and(bin_DSMarrgt.mask, np.logical_not(bin_nl_arr.mask)),
                        np.logical_and(np.logical_and(bin_DSMarrgt.mask, bin_nl_arr.mask), np.logical_not(bin_lumparr.mask)))

    # save as georefernced image
    urbanclass = np.zeros([com.shape[0], com.shape[1], 3])
    urbanclass[:, :, 0] = res*1
    urbanclass[:, :, 1] = com*1
    urbanclass[:, :, 2] = ind*1

    return urbanclass

# function end

def apply_rules(lumpath, unDSMpath, nlpath ):

    # Reading urban nDSM
    unDSMarr =  srs.raster_as_array(unDSMpath)

    # Clipping the VIIRS data as per unDSM
    nl_arr = clipraster(nlpath, unDSMpath, np.shape(unDSMarr))

    # Clipping the VIIRS data as per unDSM
    lumparr = clipraster(lumpath, unDSMpath, np.shape(unDSMarr))

    # function to apply rules
    urbanclass = rules(nl_arr, unDSMarr, lumparr)

    return urbanclass

# function end


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step5: extract urban morphology 2002     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# USE ASTER, L7 AND AW3D URBAN MORPHOLOGY TO DERIVE 2002 UM

# dientify residential from AW3D's resiudential and urban class of LULC classification fo 2002L7
def res_ASTER(res, lulc_urban):
    return (res*lulc_urban)

# identify commercial from AW3D's commercial and ASTER's nDSM
def com_ASTER(com, ASTER_nDSM):
    return ((ASTER_nDSM*com-4)>=0)

# identify industrial from AW3D's industrial and ASTER's nDSM
def ind_ASTER(ind, ASTER_nDSM):
    return ((ASTER_nDSM*ind-4)>=0)

def apply_ASTER_rules(ASTERnDSMpath, AW3DnDSMpath, AW3DNLDSMpath, lulc):

    # reading the AW3D urban class file
    arrAW3DNLDSM = srs.raster_as_array(AW3DNLDSMpath)

    # first slicing ASTER to same shaoe and location as AW3D
    arrASTER = clipraster(ASTERnDSMpath, AW3DnDSMpath, np.shape(srs.raster_as_array(AW3DnDSMpath)))

    # Classify
    # Residential
    res = res_ASTER(arrAW3DNLDSM[0], (lulc==1))

    # Commercial
    com = com_ASTER(arrAW3DNLDSM[1], arrASTER)

    # Industrial
    ind = ind_ASTER(arrAW3DNLDSM[2], arrASTER)

    # save as georefernced image
    urbanclass = np.zeros([com.shape[0], com.shape[1], 3])
    urbanclass[:,:,0] = res*1
    urbanclass[:, :, 1] = com*1
    urbanclass[:, :, 2] = ind*1

    # georeference
    return urbanclass


# --------------------  Run and plot and save ----------------------------

#Step0:
city = 'Kanpur'

# ------------  AW3D  ------------
# Step1. Initialize
AW3DnDSMpath = gb_path + r'/Data/Data_process/AW3D/India/AW3D_'+city+'_nDSM_v1.tif'
lulc2011 = gb_path + r'/Data/Data_process/Classification/KanpurL8class.tif'
citycorepath = nlpathout + '/Cities/VIIRS2014median_reslump_Kanpur.tif'
DSM = 'AW3D'


# Step2. Store the urban nDSM
#   ASWD
srs.arr_to_raster(get_undsm(AW3DnDSMpath), AW3DnDSMpath, AW3Dout+'AW3D_'+city+'_unDSM_v1.tif')

# Step3. Find median nightlight fir the year
year = '2014'
med_nl = im_mean.med_NLimg(nlpath , year, input_zone_polygon_0)
srs.arr_to_raster(med_nl, dnb.georef, nlpathout+ 'VIIRS'+year+'median.tif')


# Step4: Plot b/w NLa nd DSM
unDSMpath = AW3Dout + 'AW3D_'+city+'_nDSM_v1.tif'
#       use NL image not upsmapled?
nlpathres0 = nlpathout+ '/VIIRS2014median.tif'
#       use NL image upsampled to 30m?
nlpathres1 = nlpathout+ '/Cities/VIIRS2014median_res_Kanpur.tif'
#       flatten arrays into DF to prepare for regression plot
df_DSMNL = gather(unDSMpath, nlpathres1, downsamp=False)
df_DSMNL[df_DSMNL['DSM_mn'].notnull()].to_csv('df_'+DSM+'DSM_NL'+city+'.csv',index=True, header=True)
#       Plot
plot_hist(df_DSMNL)
plot_NLDSM(df_DSMNL, thresh=True, downsamp=False)


# Step5: Classify by rules
#       Apply rules abnd save the image
urbanclass = apply_rules(citycorepath, unDSMpath, nlpathres1 )
#       Save image
srs.ndarr_to_raster(urbanclass, unDSMpath, AW3Dout + 'AW3DUrbanclass' + city + '.tif')

# ------------  ASTER  ------------
#Step1: Initialize
DSM = 'ASTER'
lulc2002 = 0
ASTERnDSMpath = ASTERout + r'ASTER_' + city + '_nDSM_v1.tif'
unDSMpath = ASTERout + r'ASTER_' + city + '_nDSM_v1.tif'
AW3DNLDSMpath = AW3Dout + r'AW3DUrbanclass' + city + '.tif'

#Step2: Apply rules on ASTER
urbanclass = apply_ASTER_rules(ASTERnDSMpath, AW3DnDSMpath, AW3DNLDSMpath, lulc2002) #       cut according to size of AW3D
#       save the urbanclass
srs.ndarr_to_raster(urbanclass, AW3DnDSMpath, ASTERout + 'ASTERUrbanclass' + city + '.tif')














