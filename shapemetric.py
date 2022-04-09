#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 7/02/2017
# Last edit 7/29/2017

# Most functions from https://github.com/Martin-Jung/LecoS/blob/master/lecos_functions.py
# https://github.com/Martin-Jung/LecoS/blob/master/landscape_statistics.py

# Purpose: To find shapes metric of classified urban area in terms of
# (1) : Area
# (2) : Built-up density
# (3) : Landscape shape index
# (4) : Largest patch index
# (5) ; number of pathces
# (6) : patch density
# (7) : total edge
# (8) : edge density

# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used:
'''# output filenames produced

'''



import os.path
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio

import coord_translate as ct
from classLeEcos import LandCoverAnalysis

# pvt imports
# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * * *# # * * * *  * * # * * * *  * * # * *


#Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path


# location of all binary urbna class files
urban_path = gb_path + 'Data/Data_process/GEE/20city classification/all/'

# city list
city_list = gb_path +'/Codes/Urbanheights/CityList.csv'


#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: Create class with all functions     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step2: Calculate metric      * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *

ls = []

# opening dataframe contiangn all city anmesand their locations
df_city = pd.read_csv(city_list, header=0)

for file in glob(os.path.join(urban_path, '*'+'.tif')):

    print file
    cityname = os.path.basename(file)[0:-15]
    # open the file with rasterio and read it as array
    fopen = rio.open(file)
    img_arr = fopen.read(1)

    # replcae all Nan values
    img_arr[np.isnan(img_arr)]=0

    # remove all dummy values
    img_arr[img_arr<0]=0

    # finding central pixel (x,y) from its lat long
    y, x = ct.latLonToPixel(file, [[ float(df_city[df_city.City == cityname ]['Lat']) , float(df_city[df_city.City == cityname ]['Lon']) ]])[0]
    if cityname == 'NewDelhi':
        x = 1369
    if cityname == 'Firozabad':
        x = 200
        y = 310

    # Create instances object of the class - city object
    cobj = LandCoverAnalysis(array=img_arr, cellsize=30, classes={1})

    # generate labeled array of CCL
    cobj.f_ccl()

    # cityname, Landsatver, area, builtupdensity, LSI, LPI, Numpatch, total edge, edge density
    ls.append((cityname, os.path.basename(file)[-14:-12], cobj.f_returnArea(), cobj.f_builtupdensity(10, x, y) , cobj.f_returnLandscapeindex(),
               cobj.f_returnLargestPatchIndex(), cobj.numpatches, cobj.f_patchDensity(), cobj.f_returnEdgeLength(), cobj.f_returnEdgeDensity() ))

df_landscapemetric = pd.DataFrame(ls, columns=['City','Ltype', 'Area', 'BD', 'LSI', 'LPI', 'NP', 'PD', 'TE', 'ED' ])
df_landscapemetric.to_csv(csv_save_path+'df_landscapemetrics.csv', index = False, header = True)
