#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 8/08/2017
# Last edit 8/13/2017

#

# Purpose: To extract AW3D of each each city using the shape file and then generate the corresponding DTM suing the DSM2DTM
# (1) : generate DSM
# (2) : generate nDSM
# (3) : classify with NL

# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used:
'''# output filenames produced

'''

import rasterio as rio
import pandas as pd
import fun_urban_class as fuc
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
# pvt imports
import shp_rstr_stat as srs
from classDSM import DSMtrans
from classRaster import Raster_file
import coord_translate as ct

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# path = r'/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon/Sample2//'
# DSMpath = path + 'DSMsample2_rs30.tif'
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path
AW3Dpath = gb_path + r'/Data/Data_raw/AW3D/India/all/merged_v1.tif'
output_value_raster_path = gb_path + r'/Data/Data_process/AW3D/India/20city/'
# shapefiles of the cities to be processed
city20shppath = gb_path + r'/Data/Data_process/Shapefiles/20city_big_shapefiles/'
#product name
prod = 'AW3D'
#nightlight image path
nlpath = gb_path + '/Data/Data_process/VIIRS/VIIRS2014median.tif'
nlpathout = gb_path + r'/Data/Data_process/VIIRS/20city/'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/DSM/20city/'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'
csv_in_path = gb_path + '/Codes/CSVIn/'


dnb = Raster_file()
dnb.path = gb_path + r'/Data/Data_raw/VIIRS Composite/75N060E//'
dnb.sat = 'VIIRS'
dnb.prod = 'DNB'
dnb.sample = dnb.path + 'SVDNB_npp_20140201-20140228_75N060E_vcmslcfg_v10_c201507201053.avg_rade9.tif'
dnb.georef = gb_path + r'Data/Data_process/Georef_img//VIIRS_georef.tif'

dict_city = {
'C01':	'Agra',
'C02':	'Ahmedabad',
'C03':	'Allahabad',
'C04':	'Amritsar',
'C17':	'Bangalore',
'C05':	'Chennai',
'C20':	'Dehradun',
'C06':	'Firozabad',
'C07':	'Gwalior',
'C18':	'Hyderabad',
'C19':	'Jaipur',
'C08':	'Jodhpur',
'C09':	'Kanpur',
'C10':	'Kolkata',
'C11':	'Lucknow',
'C12':	'Ludhiana',
'C13':	'Mumbai',
'C14':	'NewDelhi',
'C15':	'Patna',
'C16':	'Raipur'
}

# refine resolution of VIIRS and DSM in metres
dnb_res = 450.0
DSM_res = 30.0


# using threshold rad as 3.5.30 is the threshold DN to identify urban vs non-urban in various papers for OLS> using our regression func we found this
# corresponds to rad 3.12 (ACRS2016). using here 3.5 a little strciter version.
NL_threshold = 3.5

# num of locations
num_city = 20


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: Generate clipped DSM and NL    * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# function to clop DSM for eACH CITY
def clipDSM(AW3Dpath):
    for i in range(1,num_city+1 ):
        prod = 'AW3D'
        print 'doing ',i

        # declaring shapefile for each COde
        input_zone_polygon_path = city20shppath + 'Big20city_Count_C'+str('%02d'%i) + '.shp'

        # and the corresponding output raster path
        output_raster_path = output_value_raster_path + prod+'_C'+str('%02d'%i)+'_DSMv2.tif'

        # run the making algo
        srs.rio_zone_mask(input_zone_polygon_path, AW3Dpath, output_raster_path)

# fucntion to clip NL for each city
def clipNL(NLpath):
    for i in range(1,num_city+1 ):
        prod = 'DNB'
        print 'doing ',i

        # declaring shapefile for each COde
        input_zone_polygon_path = city20shppath + 'Big20city_Count_C'+str('%02d'%i) + '.shp'

        # and the corresponding output raster path
        output_raster_path0 = nlpathout + prod+'_C'+str('%02d'%i)+'.tif'

        # run the masking algo
        srs.rio_zone_mask(input_zone_polygon_path, NLpath, output_raster_path0)

        # L1 - upsampled image; and the corresponding output raster path
        output_raster_path1 = nlpathout + prod + '_C' + str('%02d' % i) + 'L1.tif'

        #s ave the resampled array as raster
        srs.rio_resample(output_raster_path0, output_raster_path1, res_factor=15)

        # now binarizing the image for city core generator.
        # L1b - binary of upsampled image; and the corresponding output raster path
        output_raster_path2 = nlpathout + prod + '_C' + str('%02d' % i) + 'L1b.tif'

        # reopen the raster
        nlopen = rio.open(output_raster_path1)
        nl_arr = nlopen.read(1)

        # binarising at rad = 3.5
        nl_arr = nl_arr>=NL_threshold

        # savign the binary array
        srs.rio_arr_to_raster(nl_arr, output_raster_path1, output_raster_path2)
# function edn

# resample ndvi tp 30m
def resndvi(ndvipath, ndvipathout):
    # resampling step
    out_res = 0.0002777777777777778

    in_res = rio.open(ndvipath).profile['transform'][1]
    res_factor = in_res / out_res
    srs.rio_resample(ndvipath, ndvipathout, res_factor)
# function end




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step2: DEM from DSM     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

for i in range(1,num_city+1 ):

    # city id
    Count = 'C'+str('%02d'%i)
    print Count
    # input DSM name
    input_raster_path = output_value_raster_path + prod+'_'+Count+'_DSM.tif'

    # open the DSM arr from the path
    DSMarr = srs.raster_as_array(input_raster_path)[0]

    # set the arr as DSMtrans object
    cobj = DSMtrans(DSMarr)

    # find the DEM and nDSM from the array object
    (cDEM, cnDSM) = cobj.ground()

    # save the array as tif files
    srs.rio_arr_to_raster(cDEM, input_raster_path, output_value_raster_path + prod+'_'+Count+'_DEML0.tif')
    srs.rio_arr_to_raster(cnDSM, input_raster_path, output_value_raster_path + prod+'_'+Count+'_nDSML0.tif')

print 'done all'




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: Pair with NL in a  df    * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#



# store urban ndsm less rock, tree, river
def urbannDSM(prod, Count, nDSMpath, rrnDSMpath, ndvipathout, ndvi_threshold ):

    # remove river rock using masks created
    maskDSMpath = output_value_raster_path + prod + '_' + Count + '_masknDSM.tif'

    # Save only the masked areas
    #srs.rio_zone_mask(rrmask, nDSMpath, maskDSMpath)

    # reopen and save converse of masked areas
    unmaskarr = rio.open(nDSMpath).read(1) - rio.open(rrnDSMpath).read(1)

    # Save only the unmasked areas
    srs.rio_arr_to_raster(unmaskarr, nDSMpath, maskDSMpath)

    # saving path for ndsm array; cropped till ndvi extent
    unDSMpath = output_value_raster_path + prod + '_' + Count + '_unDSM.tif'

    # crop the nDSM y ndvi path
    #resnl_arr = srs.clipraster(unDSMpath, ndvipath)

    # Crop according to ndvi size and use ndvi threshold to rmoev trees
    unDSMarr = fuc.rem_ndvi(maskDSMpath, ndvipathout, ndvi_threshold)

    # save the ndvi less unDSM
    srs.rio_arr_to_raster(unDSMarr, ndvipathout, unDSMpath)




#for i in range(16,num_city+1 ):

def pairup(i):
    # Step1. Initialize
    Count = 'C'+str('%02d'%i) # city id
    city = dict_city[Count]
    print 'doing', Count, city
    nDSMpath = output_value_raster_path + prod+'_'+Count+'_nDSML0.tif'  # normalized DSM
    ndvipath = gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/'+city+'_ndvi8.tif' #ndvi image
    nl_arr_path = nlpathout + 'DNB' + '_' + Count + 'L1.tif' #nighltigh imsge
    bnl_arr_path = nlpathout + 'DNB' + '_' + Count + 'L1b.tif' # binary nl iamge
    rrmask = city20shppath + 'rockrivermask/'+ Count + '.shp'   # rock river mask
    rrnDSMpath = output_value_raster_path + '/MasknDSM/'+prod+'_'+Count+'_nDSML0rr'
    ndvi_threshold = 0.69 #  nDSM without ndvi above threshold

    # new files created
    #   out ndvi
    ndvipathout = gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/' + city + '_ndvi8res.tif'
    #   clipped and masked nDSM
    unDSMpath = output_value_raster_path + prod + '_' + Count + '_unDSM.tif'
    #   clipped nl as per ndvi
    resnl_arr_path = nlpathout + 'DNB' + '_'+ Count + 'L1clip.tif'
    # city core path
    corenl_arr_path = nlpathout + 'DNB' + '_' + Count+ 'L1core.tif'  # nighltigh imsge
    # csv tro save all info
    df_DSMNLpath = csv_save_path+'df_'+prod+'DSM_NL'+city+'.csv'


    # Step 2: resample ndvi to 30m (it is not correct, dont know hwy; maybe GEE)

    #   resampling ndvi function
    #resndvi(ndvipath, ndvipathout)


    # Step3. Store the urban nDSM after clipping river, rock and tree
    #urbannDSM(prod, Count, nDSMpath, rrnDSMpath, ndvipathout, ndvi_threshold)



    # Step3.5 Clip NL to ndvi and find urban core

    #   crop the upsampled NL image as per ndvi image
    #resnl_arr = srs.clipraster(nl_arr_path, ndvipathout)
    #srs.rio_arr_to_raster(resnl_arr, ndvipathout, resnl_arr_path)

    #   croppped binary NL image
    #bnl_arr = srs.clipraster(bnl_arr_path, ndvipath)

    #   returns core arry niary
    #citycore = fuc.fun_citycore(resnl_arr, NL_threshold+2 , erode=80, dilate = 20)

    #   store citycore
    #srs.rio_arr_to_raster(citycore, ndvipathout, corenl_arr_path)


    # Step4: Plot b/w NLa nd DSM
    unDSMarr = rio.open(unDSMpath).read(1)
    citycore = rio.open(corenl_arr_path).read(1)
    resnl_arr = rio.open(resnl_arr_path).read(1)
    #       flatten arrays into DF to prepare for regression plot
    df_DSMNL = fuc.gather(unDSMarr, resnl_arr, citycore, downsamp=False)
    df_DSMNL = df_DSMNL[df_DSMNL.DSM_mn != 0]
    df_DSMNL[df_DSMNL['DSM_mn'].notnull()].to_csv(df_DSMNLpath,index=True, header=True)

    #       Plot
    df_DSMNL = pd.read_csv(df_DSMNLpath, header =0)
    df_DSMNL = df_DSMNL[pd.notnull(df_DSMNL['DSM_mn'])]
    fuc.plot_hist(df_DSMNL)
    fuc.plot_NLDSM(df_DSMNL, city, nlth=5, dsmth=1, thresh=True, downsamp=False, core = True, save = True)

# function end


# Step5: Classify by rules
#       Apply rules abnd save the image
urbanclass = fuc.apply_rules(corenl_arr_path, unDSMpath, resnl_arr_path)

#       Save image urban morphology (UM)
umoutpath = output_value_raster_path + prod + '_' + Count + '_UM.tif'
srs.ndarr_to_raster(urbanclass, unDSMpath, umoutpath)





# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: Pair with NL in a  df    * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# to first preapre data frame from training images in a form that can be ingested by the logit reg on scikit

trainingimgpath = gb_path + '/Data/Data_process/AW3D/India/20city/TrainingUM/'
nlarrpath = nlpathout + 'DNB' + '_' + Count + 'L1clip.tif'
Count = 'C'+str('%02d'%i) # city id
city = dict_city[Count]
nlarr = rio.open(nlarrpath).read(1)
citylist_path = csv_in_path+'/CityList.csv'

# create distancefrom centre array
def distarray(citylist_path, Count):

    # open the csv
    df_citylist = pd.read_csv(citylist_path, header = 0)

    # read the lat lon
    lat = df_citylist[df_citylist.Count==Count]['Lat']
    lon = df_citylist[df_citylist.Count==Count]['Lon']

    #get the pixel coordinate
    [pix1] = ct.latLonToPixel(nlarrpath,[[float(lat),float(lon)]] )

    # create empty arr
    distarr = np.empty(rio.open(nlarrpath).read(1).shape)

    # fill array with distacen
    distarr = np.fromfunction(lambda i,j: np.sqrt((i-pix1[1])**2 + (j-pix1[0])**2) , rio.open(nlarrpath).read(1).shape, dtype = np.float32)

    return distarr
# funcion end

#read residential image and create its df
# find the distance array

distarr = distarray(citylist_path, Count)

def classif_df(suff, umclass):
    # open dsm and nl as arr
    rdsmarr = rio.open(trainingimgpath+city+suff+'.tif').read(1)
    rnlarr = nlarr*[rdsmarr>0][0]

    # convert to pandas
    df_rDSMNL = pd.DataFrame(rdsmarr.flatten().tolist(), columns=['DSM_mn'])
    df_rDSMNL['NL'] = rnlarr.flatten().tolist()
    df_rDSMNL['dist'] = distarr.flatten().tolist()
    df_rDSMNL = df_rDSMNL[(df_rDSMNL.NL!=0) | (df_rDSMNL.DSM_mn!=0) ]

    #assign class label
    df_rDSMNL['umclass'] = umclass

    return df_rDSMNL


# fucntion end

# generatte training df
def training_df():

    # gathering df for different classes
    df_r = classif_df('_r', 1)
    df_c = classif_df('_c', 2)
    df_i = classif_df('_i', 3)

    # further cleaning traing data
    # removing non residential strucutre fom r
    df_r = df_r[df_r.DSM_mn<=8]

    # removing low lying strucutres from commercial and industrial
    df_c = df_c[df_c.DSM_mn > 8]
    df_i = df_i[df_i.DSM_mn > 8]

    # consolidate into single
    df_train = df_r.append([df_c, df_i])

    # generate dummies for the 'class column
    dummy_class = pd.get_dummies(df_train.umclass, prefix = 'dumclass' )

    # join it back
    df_train = df_train.join(dummy_class)

    return df_train

# function end


def logit():

    # get training data
    df_train = training_df()

    features = df_train[['DSM_mn', 'NL', 'dist']]

    # define logit object
    logistic = LogisticRegression()

    Y = df_train['dumclass_3'].values
    X = df_train.ix[:, ['DSM_mn', 'NL']].as_matrix()

    # this also gives the same result
    #Y = df_train['dumclass_2']
    #x = df_train[['DSM_mn', 'NL', 'dist']]

    f_tr, f_ts, t_tr, t_ts = train_test_split(X,Y)

    logistic.fit(X, Y)

    pred = logistic.predict(f_ts)

    # trying to plot the result
    h = .5  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('DSM_mn')
    plt.ylabel('NL')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

# ------------  ASTER  ------------
#Step1: Initialize
DSM = 'ASTER'
lulc2002 = 0
ASTERnDSMpath = ASTERout + r'ASTER_' + city + '_nDSM_v1.tif'
unDSMpath = ASTERout + r'ASTER_' + city + '_nDSM_v1.tif'
AW3DNLDSMpath = AW3Dout + r'AW3DUrbanclass' + city + '.tif'

#Step2: Apply rules on ASTER
urbanclass = fuc.apply_ASTER_rules(ASTERnDSMpath, AW3DnDSMpath, AW3DNLDSMpath, lulc2002) #       cut according to size of AW3D
#       save the urbanclass
srs.ndarr_to_raster(urbanclass, AW3DnDSMpath, ASTERout + 'ASTERUrbanclass' + city + '.tif')

