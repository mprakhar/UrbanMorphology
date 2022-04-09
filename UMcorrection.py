
#helper functions for final corrections

# Step 1. Code to individually correct for rockriver or convert one UM class to another by brute force.
# Step 2. Code to individually correct georeference by brute force.

import rasterio as rio
import numpy as np
import sys

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
from  spatialop import shp_rstr_stat as srs

#prefix
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path from my folders

dict_city = {
    'C01': 'Agra',
    'C02': 'Ahmedabad',
    'C03': 'Allahabad',
    'C04': 'Amritsar',
    'C17': 'Bangalore',
    'C05': 'Chennai',
    'C20': 'Dehradun',
    'C06': 'Firozabad',
    'C07': 'Gwalior',
    'C18': 'Hyderabad',
    'C19': 'Jaipur',
    'C08': 'Jodhpur',
    'C09': 'Kanpur',
    'C10': 'Kolkata',
    'C11': 'Lucknow',
    'C12': 'Ludhiana',
    'C13': 'Mumbai',
    'C14': 'NewDelhi',
    'C15': 'Patna',
    'C16': 'Raipur'
}
# PLEASE UPDATE THE CORRECTED IMAGES FROM HERE TO 'UMpostprocess' folder


# DEFINE template functins for paths to be followed is as:

# correction image path

#rockriver
def rrpath(city):
    cpath = gb_path + r'/Data/Data_process/AW3D/India/20city/rockrivermaskUM/'+city+'UM_rr.tif'
    return cpath

#UMcorrect to cconvert between residentialm inustrial and commercial by force
def rcipath(city, type = 'i'):
    cpath = gb_path + r'/Data/Data_process/AW3D/India/20city/UMcorrection/'+city+'UM_'+type+'.tif'
    return cpath

#inout UM path (UM to be corrected)
def inpath(city):
    UMpath = gb_path + r'/Data/Data_process/AW3D/India/20city/UMmaps_v1_20170902/'+city+'UM.tif'
    return UMpath

#outputsavepath
def opath(city):
    UMout = gb_path + r'/Data/Data_process/AW3D/India/20city/UML2/' + city + 'UML2.tif'
    return UMout

# -------------------------- ----------------------
# C01 AGRA
city = 'Agra'
rrarr = rio.open(rrpath(city)).read(1)
UMarr = rio.open(inpath(city)).read(1)
#clean rr
UMout = UMarr*(np.where(rrarr>0,0,1 ))
#save
srs.rio_arr_to_raster(UMout, inpath(city), opath(city))

# C16 Raipur
city = 'Raipur'
rrarr = rio.open(rrpath(city)).read(1)
UMarr = rio.open(inpath(city)).read(1)
rciarr = rio.open(rcipath(city, 'i')).read(1)
#clean rr
UMout = UMarr*(np.where(rrarr>0,0,1 ))
#clean rci
UMout[rciarr==2]=3
#save
srs.rio_arr_to_raster(UMout, inpath(city), opath(city))


# C07 Gwalior
city = 'Gwalior'
rrarr = rio.open(rrpath(city)).read(1)
UMarr = rio.open(inpath(city)).read(1)
#clean rr
UMout = UMarr*(np.where(rrarr>0,0,1 ))
#save
srs.rio_arr_to_raster(UMout, inpath(city), opath(city))


# C14 New Delhi
city = 'NewDelhi'
rrarr = rio.open(rrpath(city)).read(1)
UMarr = rio.open(inpath(city)).read(1)
rciarr = rio.open(rcipath(city, 'r')).read(1)
#clean rr
UMout = UMarr*(np.where(rrarr>0,0,1 ))
#clean rci
UMout[rciarr==3]=1
#save
srs.rio_arr_to_raster(UMout, inpath(city), opath(city))

# C02 Ahmedabad
city = 'Ahmedabad'
UMarr = rio.open(inpath(city)).read(1)
rciarr = rio.open(rcipath(city, 'rb')).read(1)
#clean rci conver tall tall building training area to residential
UMarr[rciarr>=1]=1
UMout = UMarr
#save
srs.rio_arr_to_raster(UMout, inpath(city), opath(city))



# ----------------  Correction of georeference by shifting pixels  -------------------------------

# function that corrects DSM array by rolling
def get_correctnDSM(DSMpath, offsetx, offsety):
    DSMarr = rio.open(DSMpath).read(1)

    #lets rolllll. and shift DSMarr
    #xaxis
    DSMarrx = np.roll(DSMarr, offsetx, axis = 1)
    #yaxis
    DSMarry = np.roll(DSMarrx, offsety, axis = 0)

    #set re-introduced values as zero
    DSMarry[-1,:] = 0
    DSMarry[:, -1] = 0
    DSMarry[:, offsetx:-1] = 0
    DSMarry[offsety,:-1 :] = 0

    return DSMarry

# calling all cities and updating to the folder UMpostprocessgeoref
# inUMpath  has rock rover corrected from step1
inUMpath = gb_path + r'/Data/Data_process/ASTER/India/20city/UMpostprocess/'
outUMpath = gb_path + r'/Data/Data_process/ASTER/India/20city/UMpostprocessgeoref/'
offsetx = -1 # for those with L2 rrockriver correction -2
offsety = -2 # for those with L2 rrockriver correction -2

for i  in range(1,21):
    city = dict_city['C'+str('%02d'%i)]
    print city
    #open the city
    DSMpath = inUMpath + 'ASTER_'+ city + 'UM.tif'

    #outpath
    outDSMpath = outUMpath + city + 'UM.tif'

    #get its corrected array
    DSMarry = get_correctnDSM(DSMpath, offsetx, offsety)

    #save the DSMarray
    srs.rio_arr_to_raster(DSMarry, DSMpath,outDSMpath)



# ----------------  Correction of  Um y clipping with their respective shapefile   -------------------------------
# this step is necessary to maintain consistency with the gdp and populaiton statistic data used
#for ASTER
inUMpath = gb_path + r'/Data/Data_process/ASTER/India/20city/UMpostprocessgeoref/'
outUMpath = gb_path + r'/Data/Data_process/ASTER/India/20city/UMpostprocessgeorefshp/'
#for AW3D
inUMpath = gb_path + r'/Data/Data_process/AW3D/India/20city/UMpostprocessgeoref/'
outUMpath = gb_path + r'/Data/Data_process/AW3D/India/20city/UMpostprocessgeorefshp/'
shppath = gb_path + r'/Data/Data_process/Shapefiles/India_20citydistrict/'

for i  in range(1,21):
    city = dict_city['C'+str('%02d'%i)]
    print city

    #open the city
    DSMpath = inUMpath + city + 'UM.tif'

    #name of shapefile
    shpfilepath = shppath + '2011_Dist_DISTRICT_'+city+'.shp'

    #outpath
    outDSMpath = outUMpath + city + 'UM.tif'

    #get the subset array
    srs.rio_zone_mask(shpfilepath, DSMpath, outDSMpath)


