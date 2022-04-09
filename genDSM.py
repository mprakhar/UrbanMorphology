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
# pvt imports
import shp_rstr_stat as srs
from classDSM import DSMtrans
from classRaster import Raster_file

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# path = r'/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon/Sample2//'
# DSMpath = path + 'DSMsample2_rs30.tif'
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path
AW3Dpath = gb_path + r'/Data/Data_raw/AW3D/India/all/merged_v1.tif'
output_value_raster_path = gb_path + r'/Data/Data_process/AW3D/India/20city/'
city20shppath = gb_path + r'/Data/Data_process/Shapefiles/20city_big_shapefiles/'
prod = 'AW3D'
nlpath = gb_path + '/Data/Data_process/VIIRS/VIIRS2014median.tif'
nlpathout = gb_path + r'/Data/Data_process/VIIRS/20city/'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/DSM/20city/'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'



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

#  nDSM without ndvi above threshold
ndvi_threshold = 0.75

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




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: Par with NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# resample NL tp 30m
def resNL(nlpathout, ndvipath, ndvipathout, city):
    # resampling step
    out_res = 0.0002777777777777778

    in_res = rio.open(ndvipath).profile['transform'][1]
    res_factor = in_res / out_res
    srs.rio_resample(ndvipath, ndvipathout, res_factor)

# store urban ndsm less rock, tree, river
def urbannDSM(prod, Count, nDSMpath, rrnDSMpath, ndvipathout ):

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


for i in range(17,num_city+1 ):

    # city id
    Count = 'C'+str('%02d'%i)
    city = dict_city[Count]

    print 'doing', Count, city

    # ------------  AW3D  ------------
    # Step1. Initialize
    nDSMpath = output_value_raster_path + prod+'_'+Count+'_nDSML0.tif'  # normalized DSM
    ndvipath = gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/'+city+'_ndvi8.tif' #ndvi image
    nl_arr_path = nlpathout + 'DNB' + '_C' + str('%02d' % i) + 'L1.tif' #nighltigh imsge
    bnl_arr_path = nlpathout + 'DNB' + '_C' + str('%02d' % i) + 'L1b.tif' # binary nl iamge
    rrmask = city20shppath + 'rockrivermask/'+ Count + '.shp'   # rock river mask
    rrnDSMpath = output_value_raster_path + '/MasknDSM/'+prod+'_'+Count+'_nDSML0rr'

    # Step 2.0: resample ndvi to 30m (it is not correct, dont know hwy)
    # out ndvi
    ndvipathout = gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/' + city + '_ndvi8res.tif'

    #resampling step
    out_res = 0.0002777777777777778

    in_res = rio.open(ndvipath).profile['transform'][1]
    res_factor = in_res/out_res
    srs.rio_resample(ndvipath, ndvipathout, res_factor )

    # Step2. Store the urban nDSM

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


    # Step3. Uplsample NL to desired resolution and clip

    # crop the upsampled NL image as per ndvi image
    resnl_arr = srs.clipraster(nl_arr_path, ndvipathout)

    # croppped binary NL image
    bnl_arr = srs.clipraster(bnl_arr_path, ndvipath)

    # returns core arry niary
    citycore = fuc.fun_citycore(resnl_arr, NL_threshold , erode=40)
    # show citycore
    plt.figure
    plt.title(city+'citycore')
    plt.imshow(citycore)

    # Step4: Plot b/w NLa nd DSM

    #       flatten arrays into DF to prepare for regression plot
    df_DSMNL = fuc.gather(unDSMarr, resnl_arr, downsamp=False)
    df_DSMNL[df_DSMNL['DSM_mn'].notnull()].to_csv(csv_save_path+'df_'+prod+'DSM_NL'+city+'.csv',index=True, header=True)

    #       Plot
    df_DSMNL = df_DSMNL[pd.notnull(df_DSMNL['DSM_mn'])]
    fuc.plot_hist(df_DSMNL)
    fuc.plot_NLDSM(df_DSMNL, thresh=False, downsamp=False)


    # Step5: Classify by rules
    #       Apply rules abnd save the image
    urbanclass = fuc.apply_rules(citycore, unDSMpath, resnl_arr)

    #       Save image urban morphology (UM)
    umoutpath = output_value_raster_path + prod + '_' + Count + '_UM.tif'
    srs.ndarr_to_raster(urbanclass, unDSMpath, umoutpath)




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

