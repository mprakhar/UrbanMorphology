#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 8/13/2017
# Last edit 8/13/2017

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
from scipy import ndimage
import numpy.ma as ma
import rasterio as rio
import sys

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
from  spatialop import shp_rstr_stat as srs
import spatialop.my_math as mth

gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: ony urban nDSM     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# For ISRS focus on pairing with NL first
# resample height info, by mean etc, but also preserve the satdard deviation of heights



# function to get urban NDSM
def get_undsm(nDSMpath, lulcpath):

    # nDSM array
    nDSMarr = srs.raster_as_array(nDSMpath)

    # shape of nDSMpath
    shapeDSM = np.shape(nDSMarr)

    # Clip landsat as per nDSM
    lulcarr = srs.clipraster(lulcpath, nDSMpath, shapeDSM)

    # getting the built structure only DSM = unDSM by consideing Class 2
    urbanclass = 1
    lulcarr[lulcarr!=urbanclass]= False
    lulcarr[lulcarr==urbanclass]= True
    unDSMarr =  nDSMarr*lulcarr

    return unDSMarr
# function end


# function to get urban NDSM
def rem_ndvi(nDSMpath, ndvipath, threshold):

    # ndvi array
    ndviarr = srs.raster_as_array(ndvipath)[0]

    # shape of nDSMpath
    #shapeDSM = np.shape(nDSMarr)

    # Clip nDSM as per landsat
    nDSMarr = srs.clipraster( nDSMpath, ndvipath)

    # since resolution of Landsat is not exactly 30m ()it is 29.6 which manifest clearly in large images) we need to resample and match size
    # creating resmapling object
    #col = np.shape(ndviarr)[1]
    #row = np.shape(ndviarr)[0]
    #f = interpolate.interp2d(range(col), range(row), ndviarr, kind='cubic')

    ## resampling factor
    ##in_arr_res = 450
    ##out_arr_res = 30
    #yfactor = float(np.shape(nDSMarr)[0])/ float(np.shape(ndviarr)[0])
    #xfactor = float(np.shape(nDSMarr)[1]) / float(np.shape(ndviarr)[1])

    ## creating new image size
    #colnew = np.linspace(0,col, col*xfactor)
    #rownew = np.linspace(0, row, row * yfactor)
    #res_ndvi_arr = f(colnew, rownew)

    # getting the built structure only by removing ndvi desner than 0.75
    unDSMarr =  nDSMarr*(ndviarr<=threshold)

    return unDSMarr
# function end

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: resampled urban nDSM and NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# function to read urban nDSM and return resmapled image of mean and std dev (should try to return other paramters that can represent better)
def downsampDSM(unDSMarr, in_arr_res, out_arr_res):

    # resampling factor
    #in_arr_res = 30
    #out_arr_res = 450 #since it is 15 seconds
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
def upsampNL(nl_arr, in_arr_res, out_arr_res):

    # creating resmapling object
    col = np.shape(nl_arr)[1]
    row = np.shape(nl_arr)[0]
    f = interpolate.interp2d(range(col), range(row), nl_arr, kind='linear')

    # resampling factor
    #in_arr_res = 450
    #out_arr_res = 30
    xfactor = in_arr_res / out_arr_res

    # creating new image size
    colnew = np.linspace(0,col, col*xfactor)
    rownew = np.linspace(0, row, row * xfactor)
    res_nl_arr = f(colnew, rownew)

    return res_nl_arr
# funciton end


# Function to gather info from urbanDSM values and clipped NL image values for plotting. downsamp refers to DSM being downsampled
def gather(unDSMarr, nl_arr, core_arr, downsamp = False):

    # Reading urban nDSM
    #unDSMarr =  srs.raster_as_array(unDSMpath)

    # Getting rid of pixels with elevation zero
    #unDSMarr[unDSMarr==0] = np.nan

    # downsampling DSM data
    if downsamp == True:
        # Resampling unDSM to VIIRS
        [unDSMresmnarr, unDSMressdarr, unDSMressmarr] = downsampDSM(unDSMarr)

        # Clipping the VIIRS data as per unDSM
        #nl_arr = srs.clipraster(nlpath, unDSMpath, np.shape(unDSMresmnarr))

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
        #nl_arr = srs.clipraster(nlpath, unDSMpath, np.shape(unDSMresmnarr))

        # covnert to pandas dataframe
        df_DSMNL = pd.DataFrame(unDSMresmnarr.flatten().tolist(), columns=['DSM_mn'])
        df_DSMNL['NL'] = nl_arr.flatten().tolist()
    df_DSMNL['core'] = core_arr.flatten().tolist()
    return df_DSMNL
#function end


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step4: plotting urban nDSM and NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Check Hostogram
def plot_hist(df_DSMNL):

    plt.figure()
    # NL hist
    plt.subplot(121)
    plt.hist(df_DSMNL[df_DSMNL.NL>=1]['NL'] ,10)    # NL<0.5 has the max pop..and doesnt seem to maek sense
    plt.xlabel('urban NL', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)

    # sum of heights
    plt.subplot(122)
    plt.hist(df_DSMNL.DSM_mn ,10)      # sumDSM<3 doenst seem to make sense in a 450*450m pixel
    plt.xlabel('urban DSM ', fontsize = 20)
    plt.ylabel('Frequency', fontsize = 20)
# function end


# fucntion to plot trend line and scatter
def plotter(subplt, x, y, xlabelstr = '', city = 'city', functype='log',save='False'):

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
    plt.xlim(0,80)
    plt.ylim(0, 120)
    if save == True:
        plt.savefig(plt_save_path + r'/Urbanheights/20city/NLDSM' + '_' + city + '.png')

    return popt
# function end

# function to find the core path of t eh city
def fun_citycore(nlarr, bin, erode, dilate):
    # bin is the binary threshold
    # erode is the sze of erosion matrix

    # make binary image
    nlbin = nlarr>bin

    # mathematical morphology
    erodearr = ndimage.binary_erosion(nlbin, structure=np.ones((erode, erode))).astype(nlbin.dtype)

    dilarr = ndimage.binary_dilation(erodearr, structure=np.ones((dilate, dilate))).astype(nlbin.dtype)

    plt.figure()
    plt.title('citycore')
    plt.imshow(dilarr)

    return dilarr

#Plots between NLvsDSM
def plot_NLDSM(df_DSMNL, city,  nlth=15, dsmth=1, thresh=True, downsamp=False, core = True, save = False):

    # thresholds for NL and DSM are nlth, dsmth () in past basic nlth to remove useless feature =3 (thresh = False) and to get HQ feature = 15 (thresh = True), dsmth = 1

    df_DSMNL = df_DSMNL.sort_values(['DSM_mn'])

    if core == True:
        df_DSMNL = df_DSMNL[df_DSMNL.core>=1]

    if thresh==False:
        # plot withj mean
        nlth = 1
        x = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_mn >= dsmth)]['DSM_mn']
        y = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_mn >= 1)]['NL']

        # plot mean
        plotter(131, x, y, xlabelstr =' '+city, city=city, functype='log')

        # if height has been downsampled to match NLs resolution
        if downsamp:
            # plot withj std dev
            x = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_sd >= dsmth)]['DSM_sd']
            plotter(132, x, y, 'stddev', functype='log')

            # plot withj sum
            x = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_sm >= dsmth)]['DSM_sd']
            plotter(133, x, y, 'sum', functype='log')

    if thresh:
        #after threshold
        # threshold of NL = 0.3, threshold of height = 3
        # plot withj mean
        plt.figure()
        x = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_mn >= dsmth)]['DSM_mn']    # for AW3D thresh set as 1 to remove noise heights
        y = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_mn >= dsmth)]['NL']

        # plot mean
        plotter(111, x, y, xlabelstr =' '+city,city=city, functype='log', save=save)

        if downsamp:
            plt.figure()
            # plot withj std dev
            x = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_sd >= dsmth)]['DSM_sd']
            plotter(131, x, y, 'stddev', functype='log')

            # plot withj sum
            x = df_DSMNL[(df_DSMNL.NL >= nlth) & (df_DSMNL.DSM_sm >= dsmth)]['DSM_sd']
            plotter(132, x, y, 'sum', functype='log')

# fucntion end

#function to implelement rule on the pixel classify
def rules(nl_arr, DSMarr, lumparr):

    # first creating boolean array using comditions. 1 means condition fulfilled, 0 otherwise

    # height binary image mask
    height_thresh = 4
    # greater than mask
    bin_DSMarrgt = ma.masked_greater(DSMarr,height_thresh)
    DSMarr2 = DSMarr

    # removing height values less than equal 1
    DSMarr2[DSMarr2 <= 1] = np.nan
    # less than mask
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
    res = np.logical_and(bin_nl_arr2.mask,
                         bin_DSMarrlt.mask)

    # commercial
    com = np.logical_and(bin_lumparr.mask,
                         np.logical_and(bin_nl_arr.mask,
                                        bin_DSMarrgt.mask))
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

def apply_rules(lumparrpath, unDSMpath, nl_arrpath ):

    # Reading urban nDSM
    unDSMarr =  srs.raster_as_array(unDSMpath)
    lumparr = srs.raster_as_array(lumparrpath)
    nl_arr = srs.raster_as_array(nl_arrpath)

    # Clipping the VIIRS data as per unDSM
    #nl_arr = srs.clipraster(nlpath, unDSMpath, np.shape(unDSMarr))

    # Clipping the VIIRS data as per unDSM
    #lumparr = srs.clipraster(lumpath, unDSMpath, np.shape(unDSMarr))

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
    arrASTER = srs.clipraster(ASTERnDSMpath, AW3DnDSMpath, np.shape(srs.raster_as_array(AW3DnDSMpath)))

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

