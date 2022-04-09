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



import numpy as np
import scipy as sp
from scipy import ndimage # import ndimage module seperately for easy access
from scipy.spatial.distance import cdist
import math as math


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * * *# # * * * *  * * # * * * *  * * # * *




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: Create class with all functions     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

class LandCoverAnalysis():

    def __init__(self, array, cellsize, classes):
        self.array = array
        self.cellsize = cellsize
        self.cellsize_2 = math.pow(cellsize, 2)
        self.classes = classes


    # non zero count function
    def count_nonzero(self, array):
        if hasattr(np, 'count_nonzero'):
            return np.count_nonzero(array)
        elif hasattr(sp, 'count_nonzero'):
            return sp.count_nonzero(array)
        else:
            return (array != 0).sum()


    # Connected component labeling function
    def f_ccl(self, s=2):
        #cl_array = clone array
        cl_array = np.copy(self.array)  # create working array
        # Binary structure
        self.cl_array = cl_array
        struct = sp.ndimage.generate_binary_structure(s, s)
        self.labeled_array, self.numpatches = ndimage.label(cl_array, struct)



    # area
    def f_returnArea(self):
        #sizes = scipy.ndimage.sum(array, labeled_array, range(numpatches + 1)).astype(labeled_array.dtype)
        area = self.count_nonzero(self.labeled_array) * self.cellsize_2
        self.Larea = area
        return area

    # builtup density
    def density(self, array, cpix, rad):
        # cpix is the xentral pixel aorund which 2*rad m square  density will be cosnidered ()

        #convert radius distance as oer image reoslution
        radres = rad/self.cellsize

        # keep only the required roi from arr
        arrroi = array[cpix[0]-radres:cpix[0]+radres, cpix[1]-radres:cpix[1]+radres ]

        # find the density
        return (np.sum(arrroi)/np.size(arrroi))

    # Aggregates all class area, equals the sum of total area for each class
    def f_LandscapeArea(self):
        res = []
        i = self.classes
        arr = np.copy(self.array)
        arr[self.array!=i] = 0
        res.append(self.f_returnArea(arr))
        self.Larea = sum(res)


    # Return Patchdensity
    def f_patchDensity(self):
        self.f_returnArea() # Calculate LArea
        try:
            val = (float(self.numpatches) / float(self.Larea))
        except ZeroDivisionError:
            val = None
        return val


    # Return greatest, smallest or mean patch area
    def f_returnPatchArea(self,what):
        sizes = ndimage.sum(self.cl_array,self.labeled_array,range(1,self.numpatches+1))
        sizes = sizes[sizes!=0] # remove zeros
        self.cl = 1 # since  this code only will focus on class urban = 1

        if len(sizes) != 0:
            if what=="max":
                return (np.max(sizes)*self.cellsize_2) / int(self.cl)
            elif what=="min":
                return (np.min(sizes)*self.cellsize_2) / int(self.cl)
            elif what=="mean":
                return (np.mean(sizes)*self.cellsize_2) / int(self.cl)
            elif what=="median":
                return (np.median(sizes)*self.cellsize_2) / int(self.cl)
        else:
            return None

    # The largest patch index
    def f_returnLargestPatchIndex(self):
        ma = self.f_returnPatchArea("max")
        self.f_returnArea()
        return ( ma / self.Larea ) * 100


    # Returns the given matrix with a zero border coloumn and row around
    def f_setBorderZero(self,matrix):
        heightFP,widthFP = matrix.shape #define hight and width of input matrix
        withBorders = np.ones((heightFP+(2*1),widthFP+(2*1)))*0 # set the border to borderValue
        withBorders[1:heightFP+1,1:widthFP+1]=matrix # set the interior region to the input matrix
        return withBorders


    # Returns sum of patches perimeter
    def f_returnPatchPerimeter(self):
        labeled_array = self.f_setBorderZero(self.labeled_array) # make a border with zeroes
        TotalPerimeter = np.sum(labeled_array[:,1:] != labeled_array[:,:-1]) + np.sum(labeled_array[1:,:] != labeled_array[:-1,:])
        return TotalPerimeter


    # Returns total Edge length
    def f_returnEdgeLength(self):
        TotalEdgeLength = self.f_returnPatchPerimeter()
        #Todo: Mask out the boundary cells
        return TotalEdgeLength * self.cellsize


    # Return Edge Density
    def f_returnEdgeDensity(self):
        self.f_returnArea() # Calculate LArea
        try:
            val = float(self.f_returnEdgeLength()) / float(self.Larea)
        except ZeroDivisionError:
            val = None
        return val

    # Return Landscape shape index
    def f_returnLandscapeindex(self):
        total_perimeter = self.f_returnPatchPerimeter()
        simplaeshape = 2*3.14*np.sqrt(self.f_returnArea()) # cosnidering simple shape as a circle

        return (total_perimeter/simplaeshape)

    # Get builtup area in a r radius from center
    def f_circularmask(self,rad, a,b):

        arr = np.copy(self.array)

        m, n = np.shape(arr)
        r = int(rad*1000/self.cellsize)     # rad is in kilometer

        y, x = np.ogrid[-a:m - a, -b:n - b]
        mask = x * x + y * y <= r * r

        arr[~mask] = 0

        return arr, np.sum(mask)

    # Return built upo density
    def f_builtupdensity(self,rad, a, b):
        # rad - radius in kilonter
        # a,b center pixel
        area, tot_area  = self.f_circularmask(rad, a,b)
        area = np.sum(area)
        density = area/tot_area

        return density


    # Get average distance between landscape patches
    def f_returnAvgPatchDist(self, metric="euclidean"):
        if self.numpatches == 0:
            return np.nan
        elif self.numpatches < 2:
            return 0
        else:
            """
            Takes a labeled array as returned by scipy.ndimage.label and 
            returns an intra-feature distance matrix.
            Solution by @morningsun at StackOverflow
            """
            I, J = np.nonzero(self.labeled_array)
            labels = self.labeled_array[I, J]
            coords = np.column_stack((I, J))

            sorter = np.argsort(labels)
            labels = labels[sorter]
            coords = coords[sorter]

            sq_dists = cdist(coords, coords, 'sqeuclidean')

            start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
            nonzero_vs_feat = np.minimum.reduceat(sq_dists, start_idx, axis=1)
            feat_vs_feat = np.minimum.reduceat(nonzero_vs_feat, start_idx, axis=0)

            # Get lower triangle and zero distances to nan
            b = np.tril(np.sqrt(feat_vs_feat))
            b[b == 0] = np.nan
            res = np.nanmean(b) * self.cellsize  # Calculate mean and multiply with cellsize

            return res


