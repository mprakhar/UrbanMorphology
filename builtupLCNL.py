#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 2/07/2018
# Last edit 2/07/2018

# Purpose: To estimate built up area expansion of 20 cities in time series using MODIS LULC product
#           Plot the expansion




import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from mpl_toolkits.mplot3d import Axes3D
import os.path
from glob import glob
import rasterio as rio

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from  spatialop import shp_rstr_stat as srs
from classDSM import DSMtrans
from spatialop.classRaster import Raster_file
import spatialop.coord_translate as ct
import fun_urban_class as fuc

# initialize
gb_path = r'/home/prakhar/Research/AQM_research//'  # global path tp be appended to each path
LULCdir = gb_path + r'/Data/Data_raw/MCD12Q1/all/'
NLdir = gb_path + r'/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
cityshpdir = gb_path + r'//Data/Data_process/Shapefiles/India_20citydistrict/'
csv_save_path = gb_path + r'Codes/CSVOut/NL/'  # cas output path

#for year in range(2001,2014):




def LULCtrned():
    # list to store the desreid info
    ls = []

    # open the LULC file
    for filepath in glob(os.path.join(LULCdir, '*'  +  '*' + '.Global')):

        #extract the year
        head, tail = os.path.split(filepath)
        year = int(tail[9:13])
        print year

        #open the shapefile list
        for shppath in glob(os.path.join(cityshpdir, '*' + '.shp')):

            # extract the city
            head, tail = os.path.split(shppath)
            city = tail[19:-4]


            #subset the desire polygon into temp file
            srs.rio_zone_mask(shppath, filepath, 'temp.tif')

            #read the temo file
            cityLULCsubset = rio.open('temp.tif').read(1)

            #find the sum of urban pixels
            area = np.sum(cityLULCsubset==13)*0.5*0.5 #500m resoltuoin. output in km2
            #store in the list
            ls.append([year, city, area])


def NLtrned(thresh):
    # list to store the desreid info
    ls = []

    # open the LULC file
    for filepath in glob(os.path.join(NLdir, '*' + '*' + '.tif')):

        # extract the year
        head, tail = os.path.split(filepath)
        year = int(tail[3:7])
        print year

        # open the shapefile list
        for shppath in glob(os.path.join(cityshpdir, '*' + '.shp')):
            # extract the city
            head, tail = os.path.split(shppath)
            city = tail[19:-4]

            # subset the desire polygon into temp file
            srs.rio_zone_mask(shppath, filepath, 'temp.tif')

            # read the temo file
            cityLULCsubset = rio.open('temp.tif').read(1)

            # find the sum of urban pixels
            area = np.sum(cityLULCsubset >= thresh) * .9 * .9  # 1km resoltuoin. output in km2
            #print year, city, area
            # store in the list
            ls.append([year, city, area])

    dflulc = pd.DataFrame(data=ls, columns = ['year', 'city', 'area'])
    return dflulc


def plotfrp_annual(df):
    #function to plot the FRP trend for each city in the city

    #Countid = 'C02'
    #convert to datettime
    df['date'] = pd.to_datetime(df['year'], format = '%Y')
    df = df.sort_values(['area'], ascending=[0])
    df = df.sort_values(['year'], ascending=[1])
    # new figure
    plt.figure()
    ax = plt.subplot(111)
    ax.set_color_cycle(sns.color_palette('husl', 26))
    #plt.style.use('seaborn-dark-palette')
    #plotting for each year


    for yy in df['city'].unique():

        if yy in ["Mumbai Suburban", "New Delhi", "Mumbai_full", "MumbaiU",
                  ]:#"Kolkata", "Chennai"]:
            continue


        #extract df for yy
        df_yy = df[df.city==yy]
        x = df_yy.year
        y = df_yy['area']
        #y.fillna(0, inplace = True)

        #plot for this year
        ax.plot(x,y, label = np.str(yy))

    plt.legend()
    plt.xlabel('DOY')
    plt.ylabel('Area (km2)')
    plt.show()
#functio ned






# RUN this
thresh = 30
dflulc = NLtrned(thresh = thresh) #succesful = 30
dflulc.to_csv(csv_save_path+"/NLareathresh"+str(thresh)+".csv")

plotfrp_annual(dflulc)


