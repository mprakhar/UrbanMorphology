#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 9/11/2017
# Last edit 9/11/2017

# Purpose:
# 1) to get the original AW3D DSM corresponding to Urban Morphological points
# 2) to compare heights of corresponding points in the AW3D.
# 3) to assign UM based height difference found

from matplotlib import pyplot as plt
import numpy as np
from AW3D_urbanmorphology import urbanmorphology
from skimage.exposure import rescale_intensity
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
import rasterio as rio
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
from  spatialop import shp_rstr_stat as srs


# Function to remove /estimate direction of noise in ASTER
def Houghlinear(image):

    gray = rescale_intensity(image, out_range=(0, 255))
    image = gray
    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=20,
                                     line_gap=3, theta = np.array([15, 16, 17, 18, 19, 20,21, 22, 23, 24, 25])/180.0 * np.pi )

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()
#function end

# eprform FFT to find the magnitude imagto help remove the nosie in ASTER image
def fftanalyse(image):
    # fourier tranform for noise reduction
    f = np.fft.fft2(image)  # 600,900;1600,1900..
    fshift = np.fft.fftshift(f)
    # magnitude spectrum
    magspec = 20 * np.log(np.abs(fshift))
    display_mer(image, magspec, im3=False, label1='ASTER image', label2='FFT image')
#function end

def histeq(im,nbr_bins=20):
    """ function to display array as images by equalizing them from 0-255


    Parameters
    ----------
    im : array name
    nbr_bins :

   #get image histogram
   """
    # histogram
    n, bin, patches = plt.hist(im.flatten(), nbr_bins, facecolor='green', alpha=0.8)
    plt.xlabel('height difference (m)')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.show
#function end

# FUNCTION to diplay plot of AW3D, ASTER DEM, difference
def display_mer(image1, image2, image3 = 0, im3 = False, label1 = 'AW3D', label2 = 'ASTER', label3 = 'difference'):

    if im3:
        # display results
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(image1, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title(label1)

        ax[1].imshow(image2, cmap=plt.cm.gray, interpolation='nearest')
        ax[1].set_title(label2)

        ax[2].imshow(image3, cmap=plt.cm.gray, interpolation='nearest')
        ax[2].set_title(label3)

    else:
        # display results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(image1, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title(label1)

        ax[1].imshow(image2, cmap=plt.cm.gray, interpolation='nearest')
        ax[1].set_title(label2)



    # ax[2].imshow(self.labels, cmap=rcmap, interpolation='nearest')
    # ax[2].set_title("Merged labels")
    #
    # ax[3].imshow(self.baseimage[1], cmap=plt.cm.gray, interpolation='nearest')
    # ax[3].imshow(self.labels, cmap=rcmap, interpolation='nearest', alpha=.5)
    # ax[3].set_title("Segmented")

    for a in ax:
        a.axis('off')


class ASTER_urbanmorphology(urbanmorphology):

    #ASTER iel all merged images
    ASTERpath = urbanmorphology.gb_path + r'//Data/Data_process/ASTER/India/20city/ASTER_20city.tif'

    # path for compute urban morpholgical maps from AW3D
    AW3DUMallpath = urbanmorphology.gb_path + r'/Data/Data_process/AW3D/India/20city/UMpostprocess/'

    def __init__(self, i, prod='ASTER'):

        #inherited stuff
        urbanmorphology.__init__(self,i)

        # each city is accessed Count code
        self.Count = 'C'+str('%02d'%i)

        # which product for DSM - AW3D or ASTER
        self.prod2 = prod

        # name of the city
        self.city = self.dict_city[self.Count]

        #AW3D final UM path
        self.AW3Dpath = self.AW3DUMallpath + self.city + 'UM.tif'

        #outsave path for ASTER UM

        self.ASTERUMpath = self.gb_path + r'//Data/Data_process/ASTER/India/20city/UMmaps/ASTER_' + self.city + 'UM.tif'

        #NDVI taken in 2001
        # ndvi image found uisng GEE
        self.ndvipath = self.gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/' + self.city + '_ndvi7.tif'  # ndvi image

        #   out ndvi path for ndvi which has been resampled to match DSM
        self.ndvipathout = self.gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/' + self.city + '_ndvi7res.tif'

    def get_ASTERDSM(self):

        #print the city
        print self.city

        # get the big DSM path for AW3D
        AW3DDSMbigpath = self.input_DSM_path

        #UM from AW3D
        self.UMarr = rio.open(self.AW3Dpath).read(1)

        #crop the big DSM to the size of the final UM obtained
        self.AW3Darr = srs.clipraster(AW3DDSMbigpath, self.AW3Dpath, shape=0)

        #crop the big DSM to the size of the final UM obtained
        self.ASTERarr = srs.clipraster(self.ASTERpath, self.AW3Dpath, shape=0)

        #diffarr is the differencein heights between AW3D and ASTER DSM. if both heights are eqal then UM element has been preserved.
        diffarr = self.AW3Darr - self.ASTERarr




        #display_mer(self.AW3Darr, self.ASTERarr, diffarr<-20, im3=True, label3 = 'difference < -20')
        #display_mer(self.AW3Darr, self.ASTERarr, (diffarr<0)&(diffarr>-10), im3=True, label3 = 'difference le0 ge-10')

        #display all the stuff
        #display_mer(self.AW3Darr, self.ASTERarr, diffarr, im3=True)

        #get the difference on only where UM exists
        diffUM = diffarr*(self.UMarr>0)

        #checking heright difference point distribution under different UM
        histeq((diffarr*((diffarr<40)&(diffarr>-20)))[np.where(self.UMarr == 1)], 40)

        # since RMSE of ASTER and AW3D DEM is 5m, we disregard diffarray values within +-5m. Also we found <-10 mostly angular noise
        diffarrn = ((diffarr<5)&(diffarr>-5))  # <<< <<< <<< not very sure of this thresholding. CHECK

        # keep only UM freatures
        self.ASTERUMarr = diffarrn*((self.UMarr*(self.UMarr>0)))

        #plot
        #display_mer( self.UMarr, diffarr,self.ASTERUMarr, im3=True, label1='AW3D UM', label2='difference', label3 = 'ASTER UM')

        srs.rio_arr_to_raster(self.ASTERUMarr, self.AW3Dpath, self.ASTERUMpath)

        #FOR MORE ACCURACY
        #removing ndvi stuff
        #   resampling ndvi function. this is needed because ndvi images differ slightly in resolution with DSM image
        #self.resndvi()
        #resndvi = rio.open(self.ndvipathout).read(1)

        #plot the difference of the images
        #display_mer( diffarrn*((self.UMarr*(self.UMarr>0))), resndvi>0.69, im3=False, label1='AW3D UM', label2='difference', label3 = 'ASTER UM')
