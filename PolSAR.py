import numpy as np
from numpy import *
import csv
import os
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import zipfile
import os.path
import coord_translate as ct
import gdal
from PIL import Image
import pandas as pd
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from glob import glob
import seaborn as sns
from osgeo import gdal, gdalnumeric, ogr, osr

#Pvt imports
import shp_rstr_stat as srs
import my_math as mth
import my_plot as mpt
import classRaster
from classRaster import Raster_file
from classRaster import Image_arr
import infoFinder as info


# Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path


# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'



# Filter eval plots
filter_eval = pd.read_csv(csv_save_path + 'filter_eval.csv')

fig,ax = plt.subplots()
ax.plot(filter_eval['Noise variance']*100, 'o-', label= 'Noise variance x100', alpha=.8)
ax.plot(filter_eval['MSE']*10, '^-', label= 'MSE x10', alpha=.8)
ax.plot(filter_eval['ENL']/100, 'o--', label= 'ENL/100', alpha=.8)
ax.plot(filter_eval['SSI'], '^--', label= 'SSI', alpha=.8)
ax.plot(filter_eval['SMPI'], '.-', label= 'SMPI', alpha=.8)
ax.plot(filter_eval['SISA mean']/10, '.--', label= 'SISA mean/10', alpha=.8)
ax.plot(filter_eval['SISA variance']/10, '>-', label= 'SISA var/10', alpha=.8)

ax.set(xlim=(min(filter_eval.index) - 1, max(filter_eval.index) + 1), xticks=filter_eval.index,xticklabels=filter_eval['Filters'])
labels = ax.get_xticklabels()
ax.legend(ncol=2, loc=1, fontsize=20)
plt.setp(labels, rotation=90, fontsize=20)
plt.tight_layout()




