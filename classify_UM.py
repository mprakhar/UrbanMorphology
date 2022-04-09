#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 8/25/2017
# Last edit 9/01/2017

# Purpose: Main function to run all functions to extract urban morphology from AW3D AND ASTER


from AW3D_urbanmorphology import urbanmorphology
from ASTER_urbanmorphology import ASTER_urbanmorphology


# RUN ALL COMMANDS HERE

for i in range(16,21):
    urbanmorphology(i).pairup()



# train on ludhana and apply to group A
oldh = urbanmorphology(12)
#build training model
oldh.classif_model('svm')
#apply
#Agra
urbanmorphology(1).classif_data(oldh.clfmodel)
#Gwalior
urbanmorphology(7).classif_data(oldh.clfmodel)
#Allahabad
urbanmorphology(3).classif_data(oldh.clfmodel)
#Amritsar
urbanmorphology(4).classif_data(oldh.clfmodel)
#Jodhpur
urbanmorphology(8).classif_data(oldh.clfmodel)
#Dehradun
urbanmorphology(20).classif_data(oldh.clfmodel)
#Firozabad
urbanmorphology(6).classif_data(oldh.clfmodel)
#Patna
urbanmorphology(15).classif_data(oldh.clfmodel)



# train on jaipur and apply to group B
ojpr = urbanmorphology(19)
ojpr.classif_model('svm')

# apply
#Lucknow
urbanmorphology(11).classif_data(ojpr.clfmodel)
#Kanpur
urbanmorphology(9).classif_data(ojpr.clfmodel)
#Raipur
urbanmorphology(16).classif_data(ojpr.clfmodel)

#Ahmedabad
oamd = urbanmorphology(2)
oamd.classif_model('svm')
urbanmorphology(2).classif_data(oamd.clfmodel)


# train on new delhi and apply to group C
ondl = urbanmorphology(14)
ondl.classif_model('svm')
ondl.classif_data(ondl.clfmodel)

#Hyderabad
ohyd = urbanmorphology(18)
ohyd.classif_model('svm')
urbanmorphology(18).classif_data(ohyd.clfmodel)

#Kolkata
urbanmorphology(10).classif_data(ondl.clfmodel)

#Chennai
ochn = urbanmorphology(5)
ochn.classif_model('svm')
urbanmorphology(5).classif_data(ochn.clfmodel)

#Bangalore
oblr = urbanmorphology(17)
oblr.classif_model('svm')
urbanmorphology(17).classif_data(oblr.clfmodel)

#Mumbai
omum = urbanmorphology(13)
omum.classif_model('svm')
urbanmorphology(13).classif_data(omum.clfmodel)



#---------------------------------------------
##                  ASTER
#---------------------------------------------
#earlier
AW3DUMpath = '/home/prakhar/Research/AQM_research///Data/Data_process/AW3D/India/20city/UMpostprocess/'
#now 19 novemeber
AW3DUMpath = '/home/prakhar/Research/AQM_research///Data/Data_process/AW3D/India/20city/UMpostprocessgeoref/'
for i in range(5,21):
    obj = ASTER_urbanmorphology(i)
    obj.AW3DUMallpath = AW3DUMpath
    obj.get_ASTERDSM()






