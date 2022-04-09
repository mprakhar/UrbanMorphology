#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 2/09/2018
# Last edit 2/09/2018

#

# Purpose: To classify AW3D unDSM and NL into UM. Based on Function classif_data from AW3D_urbanmorphology.py


import rasterio as rio
import numpy as np



#provate import
from AW3D_urbanmorphology import urbanmorphology
from ASTER_urbanmorphology import ASTER_urbanmorphology
# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from  spatialop import shp_rstr_stat as srs
from classDSM import DSMtrans
from spatialop.classRaster import Raster_file
import spatialop.coord_translate as ct
import fun_urban_class as fuc




def custom_classif(unDSMpath, corenl_arr_path, resnl_arr_path, savefilepathUM, clfmodel):

    height_thresh = 4


    # Step4: Plot between NL and DSM
    unDSMarr = rio.open(unDSMpath).read(1)
    citycore = rio.open(corenl_arr_path).read(1)
    resnl_arr = rio.open(resnl_arr_path).read(1)

    #       flatten arrays into DF to prepare for regression plot
    df_DSMNL = fuc.gather(unDSMarr, resnl_arr, citycore, downsamp=False)

    # cleaning some points
    # df_DSMNL = df_DSMNL[df_DSMNL.DSM_mn != 0]

    # predicted class labels are stored in Z
    df_DSMNL['Z'] = clfmodel.predict(df_DSMNL.ix[:, ['DSM_mn', 'NL']].as_matrix())

    # convert Z to 2d array
    umarr = np.reshape(df_DSMNL['Z'].tolist(), resnl_arr.shape)

    # cleaning some points

    # only considering thoise points which have heights >=1
    unDSMarr = np.where(unDSMarr > 0, unDSMarr, np.nan)
    umarr = umarr * (unDSMarr >= 0)

    # considering all tall structures (>=5m) outside the city core as industry
    bin_DSMarrlt = np.ma.masked_greater_equal(unDSMarr, height_thresh).mask
    bin_nl_arr = np.ma.masked_greater_equal(resnl_arr, 1).mask  # the light should be at least 1 units
    ind = np.logical_and(np.logical_and(bin_DSMarrlt, bin_nl_arr), np.logical_not(citycore))
    umarr[ind == 1] = 3

    # considering all small structures (<5m) outside the city core as residential
    bin_DSMarrlt = np.ma.masked_less(unDSMarr, height_thresh).mask
    ind = np.logical_and(bin_DSMarrlt, np.logical_not(citycore))
    umarr[ind == 1] = 1

    # remove regions with low light
    umarr[bin_nl_arr == 0] = 0

    # save the urban morphology classfied array
    srs.rio_arr_to_raster(umarr, unDSMpath, savefilepathAW3DUM )


def custom_classifASTER( AW3DDSMpath, savefilepathAW3DUM, ASTERDSMpath, savefilepathASTERUM):

    # get the big DSM path for AW3D
    AW3DDSMbigpath = AW3DDSMpath

    # UM from AW3D
    UMarr = rio.open(savefilepathAW3DUM).read(1)

    # crop the big DSM to the size of the final UM obtained
    AW3Darr = rio.open(AW3DDSMbigpath).read(1)

    # crop the big DSM to the size of the final UM obtained
    ASTERarr = rio.open(ASTERDSMpath).read(1)


    #NEVER use different methods to subset. need to be consistent with either qgis  or pythonsusbet. else following mismatch will occur
    ASTERarr= ASTERarr[1:,:]

    # diffarr is the differencein heights between AW3D and ASTER DSM. if both heights are eqal then UM element has been preserved.
    diffarr = AW3Darr - ASTERarr

    # display_mer(self.AW3Darr, self.ASTERarr, diffarr<-20, im3=True, label3 = 'difference < -20')
    # display_mer(self.AW3Darr, self.ASTERarr, (diffarr<0)&(diffarr>-10), im3=True, label3 = 'difference le0 ge-10')

    # display all the stuff
    # display_mer(self.AW3Darr, self.ASTERarr, diffarr, im3=True)

    # get the difference on only where UM exists
    #UM1
    diffarr = diffarr[:-1, :-1]
    #UM2
    #diffarr = diffarr[1:, 1:]

    # since RMSE of ASTER and AW3D DEM is 5m, we disregard diffarray values within +-5m. Also we found <-10 mostly angular noise
    diffarrn = ((diffarr < 5) & (diffarr > -5))  # <<< <<< <<< not very sure of this thresholding. CHECK
    diffarrn2 = ((diffarr < 7) & (diffarr > -5))
    diffarrn[:,5000:] = diffarrn2[:,5000:]
    # keep only UM freatures
    ASTERUMarr = diffarrn * ((UMarr * (UMarr > 0)))

    # plot
    # display_mer( self.UMarr, diffarr,self.ASTERUMarr, im3=True, label1='AW3D UM', label2='difference', label3 = 'ASTER UM')

    srs.rio_arr_to_raster(ASTERUMarr, savefilepathAW3DUM, savefilepathASTERUM1)




# train on jaipur and apply to group B
ojpr = urbanmorphology(19)
ojpr.classif_model('svm')

#classify custom fuiles
mergedAW3D = r"/home/prakhar/Research/AQM_research/Data/Data_raw/AW3D/India/all/merged_v1.tif"
input_zone_polygon_path = r"/home/prakhar/Research/AQM_research/Data/Data_process/Shapefiles/20city_big_shapefiles/KanpurLucknow.shp"
AW3DDSMpath = r"/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/India/KanpurLucknow/KanpurLucknow_DSM.tif"
AW3DunDSMpath = r"/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/India/KanpurLucknow/KanpurLucknow_unDSM.tif"
corenl_arr_path = r"/home/prakhar/Research/AQM_research/Data/Data_process/VIIRS/KanpurLucknow/KanpurLucknow_core.tif"
resnl_arr_path = r"/home/prakhar/Research/AQM_research/Data/Data_process/VIIRS/KanpurLucknow/DNB_KanpurLucknowL1.tif"
savefilepathAW3DUM = r"/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/India/KanpurLucknow/KanpurLucknow_UMth4.tif"
ASTERDSMpath = r"/home/prakhar/Research/AQM_research/Data/Data_process/ASTER/India/KanpurLucknow/KanpurLucknowDSM.tif"
savefilepathASTERUM1 = r"/home/prakhar/Research/AQM_research/Data/Data_process/ASTER/India/KanpurLucknow/KanpurLucknowDSM_UM1.tif"
savefilepathASTERUM2 = r"/home/prakhar/Research/AQM_research/Data/Data_process/ASTER/India/KanpurLucknow/KanpurLucknowDSM_UM2.tif"
savefilepathASTERUM = r"/home/prakhar/Research/AQM_research/Data/Data_process/ASTER/India/KanpurLucknow/KanpurLucknowDSM_UM.tif"


#prepare required AW3D DSMa s well
#srs.rio_zone_mask(input_zone_polygon_path, mergedAW3D, AW3DDSMpath)


#prepare required ASTER DSMa s well
#srs.rio_zone_mask(input_zone_polygon_path, r"/home/prakhar/Research/AQM_research/Data/Data_raw/ASTER/India/KanpurLucknow/20180209110931_250281685.tif", ASTERDSMpath)


#classify AW3D UM
custom_classif(AW3DunDSMpath, corenl_arr_path, resnl_arr_path, savefilepathAW3DUM, ojpr.clfmodel)

#classify ASTER
custom_classifASTER( AW3DDSMpath, savefilepathAW3DUM, ASTERDSMpath, savefilepathASTERUM)

#merge ASTERUM1 and ASTER UM2
ASTERUM1 = rio.open(savefilepathASTERUM1).read(1)
ASTERUM2 = rio.open(savefilepathASTERUM2).read(1)
ASTERUM= ASTERUM1
ASTERUM = ((ASTERUM1 == 2) | (ASTERUM2 == 2))*2 | ((ASTERUM1 == 3) | (ASTERUM2 == 3))*3 |  (ASTERUM1 == 1)*1
srs.rio_arr_to_raster(ASTERUM, savefilepathASTERUM1, savefilepathASTERUM)

