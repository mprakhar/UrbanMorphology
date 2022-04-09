#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 8/25/2017
# Last edit 9/01/2017

# Purpose: To count pixels in diffferent classes and in AW3D andASTER at 10km and 20km radius..



from AW3D_urbanmorphology import urbanmorphology
from ASTER_urbanmorphology import ASTER_urbanmorphology
import pandas as  pd
import numpy as np
import rasterio as rio
import sys

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop.coord_translate as ct
#prefix
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path from my folders



## ---------------- ------------ ----- ---- ---- --- -- -- -- --
##                           Analyse
## ---------------- ------------ ----- ---- ---- --- -- -- -- --

# count num of pixel in each UM class
def get_pix(imagepath, Counti):
    # open the csv
    df_citylist = pd.read_csv(urbanmorphology(1).citylist_path, header = 0)
    Count = 'C' + str('%02d' % Counti)
    # read the lat lon
    lat = df_citylist[df_citylist.Count==Count]['Lat']
    lon = df_citylist[df_citylist.Count==Count]['Lon']

    #get the pixel coordinate
    [pix1] = ct.latLonToPixel(imagepath,[[float(lat),float(lon)]] )

    return pix1

def UMcount(arr):
    # function to count the different UM values in a given array
    c1 = np.sum(arr== 1) # residential
    c2 = np.sum(arr == 2) # commercial
    c3 = np.sum(arr == 3) # industrial
    c4 = np.sum(arr == 4) # tall residential
    return [c1, c2, c3, c4]


def get_UM(imagepath, Counti):
    # get UM counts at 10km, 20km, full arr lvel
    # read the image as array
    fullarr = rio.open(imagepath).read(1)

    # get the pixel value of central pixel
    pixll = get_pix(imagepath, Counti)

    # ge thte array 10km on either side
    arr10 = fullarr[pixll[1]-10000/30:pixll[1]+10000/30, pixll[0]-10000/30:pixll[0]+10000/30 ]
    UM10 = UMcount(arr10)
    # boolean to check oif array can have 20 km radius
    fullsize = True
    try:
        arr20 = fullarr[pixll[1] - 20000 / 30:pixll[1] + 20000 / 30, pixll[0] - 20000 / 30:pixll[0] + 20000 / 30]
    except IndexError:
        fullsize = False
        UM20 = [0,0,0,0]

    if fullsize:
        UM20 = UMcount(arr20)

    # counting toal UM
    UMF = UMcount(fullarr)

def get_pix(imagepath, Counti):
    # open the csv
    df_citylist = pd.read_csv(urbanmorphology(1).citylist_path, header = 0)
    Count = 'C' + str('%02d' % Counti)
    # read the lat lon
    lat = df_citylist[df_citylist.Count==Count]['Lat']
    lon = df_citylist[df_citylist.Count==Count]['Lon']

    #e xception for coastal cities. because their 20km from city centre extends in to the sea causing the image to overflow
    # exception for mumbai
    #lat = 19.062
    #lon = 72.9999
    # exceptio for chennai
    #lat = 13.0762
    #lon = 80.1480

    #get the pixel coordinate
    [pix1] = ct.latLonToPixel(imagepath,[[float(lat),float(lon)]] )

    return pix1

def UMcount(arr):
    # function to count the different UM values in a given array
    c1 = np.sum(arr==1) # residential
    c2 = np.sum(arr == 2) # commercial
    c3 = np.sum(arr == 3) # industrial
    c4 = np.sum(arr == 4) # tall residential
    return [c1, c2, c3, c4]


def get_UM(imagepath, Counti):
    # get UM counts at 10km, 20km, full arr lvel
    # read the image as array
    fullarr = rio.open(imagepath).read(1)

    # get the pixel value of central pixel
    pixll = get_pix(imagepath, Counti)

    # ge thte array 10km on either side
    arr10 = fullarr[pixll[1]-10000/30:pixll[1]+10000/30, pixll[0]-10000/30:pixll[0]+10000/30 ]
    UM10 = UMcount(arr10)
    # boolean to check oif array can have 20 km radius
    full20size = True
    try:
        arr20 = fullarr[pixll[1] - 20000 / 30:pixll[1] + 20000 / 30, pixll[0] - 20000 / 30:pixll[0] + 20000 / 30]
    except IndexError:
        full20size = False
        UM20 = [0,0,0,0]

    if full20size:
        UM20 = UMcount(arr20)

    UMF = UMcount(fullarr)
    UMF.append(np.size(fullarr))


    return [UM10, UM20, UMF]

def get_df():

    ls = []
    for Counti in range(1, 21):

        # city name
        Count = ASTER_urbanmorphology(Counti).dict_city['C' + str('%02d' % Counti)]

        #AW3DUMpath = ASTER_urbanmorphology(Counti).AW3Dpath
        #ASTERUMpath = ASTER_urbanmorphology(Counti).ASTERUMpath

        AW3DUMpath =  gb_path + r'/Data/Data_process/AW3D/India/20city/UMpostprocessgeorefshp/' + Count + 'UM.tif'
        ASTERUMpath = gb_path + r'/Data/Data_process/ASTER/India/20city/UMpostprocessgeorefshp/' + Count + 'UM.tif'

        lst =  get_UM(AW3DUMpath, Counti)
        flat_list = [item for sublist in lst for item in sublist]
        # appending the year name
        flat_list.extend([Counti, Count, 2015])
        ls.append(tuple(flat_list))

        lst = get_UM(ASTERUMpath, Counti)
        flat_list = [item for sublist in lst for item in sublist]
        # appending the year name
        flat_list.extend([Counti, Count, 2001])
        ls.append(tuple(flat_list))
    #convert to df
    labels = [ 'R10', 'C10', 'I10','B10', 'R20', 'C20', 'I20','B20', 'R00', 'C00', 'I00','B00', 'area', 'id', 'City','year' ]
    df_UM = pd.DataFrame.from_records(ls,columns = labels)
    #dataframe
    df_UM.to_csv(urbanmorphology(1).csv_save_path+'UMshpcountv2.csv', header = True, index=False )


# to run:
# simply run content on get_df()


