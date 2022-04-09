#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 8/08/2017
# Last edit 11/09/2017

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

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import sys

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from  spatialop import shp_rstr_stat as srs
from classDSM import DSMtrans
from spatialop.classRaster import Raster_file
import spatialop.coord_translate as ct
import fun_urban_class as fuc

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# path = r'/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/Yangon/Sample2//'
# DSMpath = path + 'DSMsample2_rs30.tif'


# num of locations
num_city = 20


class urbanmorphology():


    # initialize
    gb_path = r'/home/prakhar/Research/AQM_research//'  # global path tp be appended to each path

    #

    AW3Dpath = gb_path + r'/Data/Data_raw/AW3D/India/all/merged_v1.tif'
    #AW3Dpath = gb_path + r'/Data/Data_raw/AW3D/India/all/merged_patna v1.tif'

    city20shppath = gb_path + r'/Data/Data_process/Shapefiles/20city_big_shapefiles/'
    nlpath = gb_path + '/Data/Data_process/VIIRS/VIIRS2014median.tif'
    nlpathout = gb_path + r'/Data/Data_process/VIIRS/20city/'

    # Output location
    output_value_raster_path = gb_path + r'/Data/Data_process/AW3D/India/20city/'
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

    # funxtion to generate distance image from city centre
    citylist_path = csv_in_path + '/CityList.csv'

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
    ndvi_threshold = 0.69  # nDSM without ndvi above threshold
    height_thresh = 5 # 5m(+5m due to error) is the residential structyure height in the bylaws

    def __init__(self, i, prod='AW3D'):

        # each city is accessed Count code
        self.Count = 'C'+str('%02d'%i)

        # which product for DSM - AW3D or ASTER
        self.prod = prod

        # name of the city
        self.city = self.dict_city[self.Count]

        # declaring shapefile for each city from which DSM, NL will be cropped.
        self.input_zone_polygon_path = self.city20shppath + 'Big20city_Count_' +self.Count + '.shp'

        # clipped DSM path
        self.input_DSM_path = self.output_value_raster_path + self.prod + '_' + self.Count + '_DSM.tif'

        # and the corresponding output raster path
        self.output_DSM_path = self.output_value_raster_path + self.prod + '_' + self.Count + '_DSMv2.tif'

        #nDSM - thgis is L0 nDSM. it will be prcocessed further to remove useless artifacts # normalized DSM
        self.nDSM0path = self.output_value_raster_path + self.prod + '_' + self.Count + '_nDSML0.tif'

        # path of the nDSM file which consists of rocks and river; selected by using shapefiles  using QGIS
        self.rrnDSMpath = self.output_value_raster_path + '/MasknDSM/' + self.prod + '_' + self.Count + '_nDSML0rr'

        # saving path for nDSM file from which ndvi has also been removed. this file is generated after/using sself.rrnDSMpath
        self.unDSMpath = self.output_value_raster_path + self.prod + '_' + self.Count + '_unDSM.tif'

        # nightlight image that is resampled
        self.nl_arr_path = self.nlpathout + 'DNB' + '_' + self.Count + 'L1.tif' #nighltigh imsge

        # binary nighltight resampled imaeg
        self.bnl_arr_path = self.nlpathout + 'DNB' + '_' + self.Count + 'L1b.tif' # binary nl iamge

        #   clipped nl as per ndvi
        self.resnl_arr_path = self.nlpathout + 'DNB' + '_'+ self.Count + 'L1clip.tif'

        # city core path
        self.corenl_arr_path = self.nlpathout + 'DNB' + '_' + self.Count+ 'L1core.tif'  # nighltigh imsge

        # ndvi image found uisng GEE
        self.ndvipath = self.gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/' + self.city + '_ndvi8.tif'  # ndvi image

        #   out ndvi path for ndvi which has been resampled to match DSM
        self.ndvipathout = self.gb_path + r'/Data/Data_process/GEE/20city classification/allndvi/' + self.city + '_ndvi8res.tif'



    # * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: Generate clipped DSM and NL    * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

    # function to clop DSM for eACH CITY
    def clipDSM(self):

        print 'clipping city DSM from mother DSM ', self.Count

        # run the making algo
        srs.rio_zone_mask(self.input_zone_polygon_path, self.AW3Dpath, self.output_DSM_path)


    # fucntion to clip NL for each city
    def clipNL(self):
        """function to resample nightlight images and clip the maccording to extentof input zone polygon

        Parameters
        ----------
        nlpath : monther NL image

        ndvipathout: path of resampled ndvi image

        Returns
        -------
        [X]data : 2 image
        nl_arr_path : cropped NL image
        bnl_arr_path : binarized cropped NL image
        """
        prod = 'DNB'
        print 'clipping city night light from mother file ',self.Count

        # and the corresponding output raster path
        output_NLraster_path0 = self.nlpathout + prod+'_'+self.Count+'L0.tif'

        # run the masking algo
        srs.rio_zone_mask(self.input_zone_polygon_path, self.nlpath, output_NLraster_path0)

        #s ave the resampled array as raster
        srs.rio_resample(output_NLraster_path0, self.nl_arr_path, res_factor=15)

        # now binarizing the image for city core generator.

        # reopen the clipped raster as array
        nl_arr = rio.open(self.nl_arr_path).read(1)

        # binarising at rad = 3.5
        nl_arr = nl_arr>=self.NL_threshold

        # savign the binary array
        srs.rio_arr_to_raster(nl_arr, self.nl_arr_path, self.bnl_arr_path)
    # function edn

    # resample ndvi tp 30m
    def resndvi(self):

        # resampling step
        out_res = 0.0002777777777777778

        #desired resolutino
        in_res = rio.open(self.ndvipath).profile['transform'][1]
        res_factor = in_res / out_res

        srs.rio_resample(self.ndvipath, self.ndvipathout, res_factor)
    # function end




    # * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step2: DEM from DSM     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

    def DEM2DSM(self):
        """function to find DEM from the DSM

        Parameters
        ----------
        input_DSM_path : DSM image path. original image.
        ndvi_threshold : ndvi threshold above which everything considered as trees 
        rrnDSMpath : rock, river only image, created in QGIS
        ndvipathout: path of resampled ndvi image

        Returns
        -------
        [X]data : image
        nDSM0path - basic nDSM image. hence L0
        """

        print 'Calculating DEM from DSM ', self.Count

        # open the DSM arr from the path
        DSMarr = srs.raster_as_array(self.input_DSM_path)[0]

        # set the arr as DSMtrans object
        cobj = DSMtrans(DSMarr)

        # find the DEM and nDSM from the array object
        (cDEM, cnDSM) = cobj.ground()

        # save the array as tif files
        # DEM
        srs.rio_arr_to_raster(cDEM, self.input_DSM_path, self.output_value_raster_path + self.prod + '_' + self.Count + '_DEML0.tif')

        srs.rio_arr_to_raster(cnDSM, self.input_DSM_path, self.nDSM0path)
    # finction end




    # * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: Pair with NL in a  df    * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

    #
    def urbannDSM(self ):
        """function to find to store urban nDSM without rock, tree, river by masking them using shapefiles.  
        Using ndvi image with a tyhreshold to remove trees
        saves an unDSMarr array produced into unDSMpath 

        Parameters
        ----------
        nDSMpath : nDSM image path
        ndvi_threshold : ndvi threshold above which everything considered as trees 
        rrnDSMpath : rock, river only image, created in QGIS
        ndvipathout: path of resampled ndvi image
 
        Returns
        -------
        [X]data : image
        unDSMarr - urban DSM iamge. saved in the unDSMpath
        """



        # save path for DSM from which remove river rock using masks created.
        maskDSMpath = self.output_value_raster_path + self.prod + '_' + self.Count + '_masknDSM.tif'

        # Save only the masked areas. Not needed since we already performed this step in QGIS
        #rrmask = self.city20shppath + 'rockrivermask/' + self.Count + '.shp'  # rock river mask
        #srs.rio_zone_mask(rrmask, nDSMpath, maskDSMpath)

        # reopen and save converse of masked areas
        unmaskarr = rio.open(self.nDSM0path).read(1) - rio.open(self.rrnDSMpath).read(1)

        # Save only the unmasked areas
        srs.rio_arr_to_raster(unmaskarr, self.nDSM0path, maskDSMpath)

        # crop the nDSM y ndvi path
        #resnl_arr = srs.clipraster(unDSMpath, ndvipath)

        # Crop according to ndvi size and use ndvi threshold to rmoev trees
        unDSMarr = fuc.rem_ndvi(maskDSMpath, self.ndvipathout, self.ndvi_threshold)

        # save the ndvi less unDSM
        srs.rio_arr_to_raster(unDSMarr, self.ndvipathout, self.unDSMpath)
    #function end


    def dataprep(self):
        #funciton to run all previous functions in logical order and preapre the data
        # you can choose to not run this function and run indiviual functions instead


        # Step 2: resample ndvi to 30m (it is not correct, dont know hwy; maybe GEE)

        #   resampling ndvi function. this is needed because ndvi images differ slightly in resolution with DSM image
        self.resndvi()


        # Step3. Store the urban nDSM after clipping river, rock and tree
        self.urbannDSM()


        # Step3.5 Clip NL to ndvi and find urban core

        #   crop the upsampled NL image as per ndvi image
        resnl_arr = srs.clipraster(self.nl_arr_path, self.ndvipathout)
        srs.rio_arr_to_raster(resnl_arr, self.ndvipathout, self.resnl_arr_path)

        #   croppped binary NL image
        bnl_arr = srs.clipraster(self.bnl_arr_path, self.ndvipath)

        #   returns core arry niary
        citycore = fuc.fun_citycore(resnl_arr, self.NL_threshold+2 , erode=80, dilate = 20)

        #   store citycore
        srs.rio_arr_to_raster(citycore, self.ndvipathout, self.corenl_arr_path)


    def pairup(self):
        # Function to put final data rpeapred into a dataframe and also to display the plot of NL and height

        print 'pairing unDSM and NL into a df', self.Count, self.city

        # csv tro save all info
        df_DSMNLpath = self.csv_save_path+'df_'+self.prod+'DSM_NL'+self.city+'.csv'

        # Step4: Plot between NL and DSM
        unDSMarr = rio.open(self.unDSMpath).read(1)
        citycore = rio.open(self.corenl_arr_path).read(1)
        resnl_arr = rio.open(self.resnl_arr_path).read(1)

        #       flatten arrays into DF to prepare for regression plot
        df_DSMNL = fuc.gather(unDSMarr, resnl_arr, citycore, downsamp=False)
        df_DSMNL = df_DSMNL[df_DSMNL.DSM_mn != 0]
        df_DSMNL[df_DSMNL['DSM_mn'].notnull()].to_csv(df_DSMNLpath,index=True, header=True)

        #       Plot
        df_DSMNL = pd.read_csv(df_DSMNLpath, header =0)
        df_DSMNL = df_DSMNL[pd.notnull(df_DSMNL['DSM_mn'])]
        fuc.plot_hist(df_DSMNL)
        fuc.plot_NLDSM(df_DSMNL, self.city, nlth=5, dsmth=1, thresh=True, downsamp=False, core = True, save = True)

    # function end





    # * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: Logistic regression   * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
    # to first preapre data frame from training images in a form that can be ingested by the logit reg on scikit


    # create array distance from city centre array
    def distarray(self):

        # open the csv
        df_citylist = pd.read_csv(self.citylist_path, header = 0)

        # read the lat lon
        lat = df_citylist[df_citylist.Count==self.Count]['Lat']
        lon = df_citylist[df_citylist.Count==self.Count]['Lon']

        #get the pixel coordinate
        [pix1] = ct.latLonToPixel(self.resnl_arr_path,[[float(lat),float(lon)]] )

        # create empty arr
        distarr = np.empty(rio.open(self.resnl_arr_path).read(1).shape)

        # fill array with distacen
        distarr = np.fromfunction(lambda i,j: np.sqrt((i-pix1[1])**2 + (j-pix1[0])**2) , rio.open(self.resnl_arr_path).read(1).shape, dtype = np.float32)

        return distarr
    # funcion end

    #read residential image and create its df
    # find the distance array


    def classif_df(self, umclass):
        #opens the training images that have been prepared using QGIS; and collects data in a dataframe
        # for training. the rows are also labelled by their class. return dataframe

        # dictionary to specify siffix of the training image. -> residential, commerical, insutrial
        dict_um = {1: '_r',
                   2 : '_c',
                   3 : '_i',
                   4 : '_rb'}


        # path for the training image
        trainingimgpath = self.gb_path + '/Data/Data_process/AW3D/India/20city/TrainingUM/'

        # open dsm and nl as arr
        rdsmarr = rio.open(trainingimgpath+self.city+dict_um[umclass]+'.tif').read(1)
        rdsmarr = np.nan_to_num(rdsmarr)
        nlarr = rio.open(self.resnl_arr_path).read(1)
        rnlarr = nlarr*[rdsmarr>0][0]

        # convert to pandas
        df_rDSMNL = pd.DataFrame(rdsmarr.flatten().tolist(), columns=['DSM_mn'])
        df_rDSMNL['NL'] = rnlarr.flatten().tolist()

        # funxtion to generate distance image from city centre
        citylist_path = self.csv_in_path + '/CityList.csv'
        distarr = self.distarray()

        # set the image as list in the df
        df_rDSMNL['dist'] = distarr.flatten().tolist()
        df_rDSMNL = df_rDSMNL[(df_rDSMNL.NL!=0) | (df_rDSMNL.DSM_mn!=0) ]

        #assign class label
        df_rDSMNL['umclass'] = umclass

        return df_rDSMNL
    # fucntion end


    # generatte training df
    def training_df(self, res_bldg= False):
        # calls to generate the dataframe geenrated from images classif_df(self, umclass) and cleans it by applying height conditions
        # returns a df_train which will be used for training

        # gathering df for different classes - 1) res; 2) commerc; 3) indus
        df_r = self.classif_df(1)
        df_c = self.classif_df(2)
        df_i = self.classif_df(3)
        if res_bldg==True:
            df_rb = self.classif_df(4)

        # further cleaning traing data
        # removing non residential strucutre fom r
        df_r = df_r[df_r.DSM_mn<=8]

        # removing low lying strucutres from commercial and industrial
        df_c = df_c[df_c.DSM_mn > 5]
        df_i = df_i[df_i.DSM_mn > 5]

        # consolidate into single
        df_train = df_r.append([df_c, df_i])

        # generate dummies for the 'class column
        dummy_class = pd.get_dummies(df_train.umclass, prefix = 'dumclass' )

        # join it back
        df_train = df_train.join(dummy_class)

        return df_train
    # function end

    def plot_model(self, model, mmodel, X, Y):
        # function to plot the results of trained model called model. also plots given training inputs X, Y

        # trying to plot the result
        h = .5  # step size in the mesh
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 6))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('DSM_mn (meter)')
        plt.ylabel('NL (Watts/cm$^2$/sr)')
        plt.title(mmodel +'  regression -' + self.city )

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #plt.xticks(())
        #plt.yticks(())

        plt.show()

    # function to get confusion matrix by dividing ground truth into training and test
    def get_accuracy(self, mmodelparam, X, Y ):

        #this split is for reporting accuracy. for actual classification we use all dta
        x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y)

        # trian the model
        mmodelparam = mmodelparam.fit(x_tr, y_tr)

        #get the predictions
        y_pred = mmodelparam.predict(x_ts)

        print(classification_report(y_ts, y_pred, target_names=['residential', 'commercial', 'industrial']))
        print(confusion_matrix(y_ts, y_pred, labels=[1,2,3]))
        print 'kappa', cohen_kappa_score(y_ts, y_pred, labels=[1,2,3])
    #funciton end




    def classif_model(self, mmodel,  res_bldg= False):
        # function to train the mmodel on given image set. Need to specify the type of classifier mmodel to be sued
        # mmodel = 'multinomial', 'svm'

        # get training data
        df_train = self.training_df( res_bldg= res_bldg)

        # features to be cosnidered as independetn
        features = df_train[['DSM_mn', 'NL', 'dist']]

        # dependent variable
        Y = df_train['umclass'].values

        # prepare the independent data in requried format. use 'features variable
        X = df_train.ix[:, ['DSM_mn', 'NL']].as_matrix()
        # this also gives the same result
        # Y = df_train['dumclass_2']
        # x = df_train[['DSM_mn', 'NL', 'dist']]


        # choose which classifier to be used
        if mmodel == 'multinomial':

            # define logit object
            logistic = LogisticRegression(solver = 'sag', max_iter=100, random_state=42,multi_class= mmodel)

            #also get the accuracy
            self.get_accuracy(logistic, X, Y)

            logistic.fit(X, Y)

            self.plot_model(logistic, 'Logit multinomial', X, Y)

            self.clfmodel = logistic


            return logistic

        if mmodel == 'svm':

            #mmodel = 'svm'
            #features = df_train[['DSM_mn', 'NL', 'dist']]

            # define logit object
            clfsvm = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=100.)

            #also get the accuracy
            self.get_accuracy(clfsvm, X, Y)

            clfsvm.fit(X, Y)

            self.plot_model(clfsvm, 'SVM rbf', X, Y)

            self.clfmodel = clfsvm

            return clfsvm

    def classif_data(self, clfmodel):

        # Step4: Plot between NL and DSM
        unDSMarr = rio.open(self.unDSMpath).read(1)
        citycore = rio.open(self.corenl_arr_path).read(1)
        resnl_arr = rio.open(self.resnl_arr_path).read(1)

        #       flatten arrays into DF to prepare for regression plot
        df_DSMNL = fuc.gather(unDSMarr, resnl_arr, citycore, downsamp=False)

        # adding additional column of distance array
        distarr = self.distarray()

        # set the image as list in the df
        df_DSMNL['dist'] = distarr.flatten().tolist()

        # cleaning some points
        #df_DSMNL = df_DSMNL[df_DSMNL.DSM_mn != 0]

        # predicted class labels are stored in Z
        df_DSMNL['Z'] = clfmodel.predict(df_DSMNL.ix[:, ['DSM_mn', 'NL']].as_matrix())

        # convert Z to 2d array
        umarr = np.reshape( df_DSMNL['Z'].tolist(), resnl_arr.shape)

        # cleaning some points

        #only considering thoise points which have heights >=1
        unDSMarr = np.where(unDSMarr>0,unDSMarr, np.nan )
        umarr = umarr*(unDSMarr >= 0)

        # considering all tall structures (>=5m) outside the city core as industry
        bin_DSMarrlt = np.ma.masked_greater_equal(unDSMarr, self.height_thresh).mask
        bin_nl_arr = np.ma.masked_greater_equal(resnl_arr, 1).mask # the light should be at least 1 units
        ind = np.logical_and(np.logical_and(bin_DSMarrlt, bin_nl_arr), np.logical_not(citycore))
        umarr[ind == 1] = 3

        # considering all small structures (<5m) outside the city core as residential
        bin_DSMarrlt = np.ma.masked_less(unDSMarr, self.height_thresh).mask
        ind = np.logical_and(bin_DSMarrlt, np.logical_not(citycore))
        umarr[ind == 1] = 1

        #remove regions with low light
        umarr[bin_nl_arr == 0] = 0

        #save the urban morphology classfied array
        srs.rio_arr_to_raster(umarr, self.unDSMpath, self.city +'UM.tif')




