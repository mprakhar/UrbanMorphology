#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 9/01/2017
# Last edit 11/01/2017

# Purpose: To estimate area expansion of type 1,2,3 cities between 2001 and 2013. using population, GDP. we use the results generated using the code analysis.py.
#           Further, we add information about populaition and GDP and city type. Since Populaiton and GDP are correlated themseleves, we explore ridge regression
#           and show the significance f the parqameters.Made changes about I, C, counts in Ferozabad

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy.optimize import curve_fit
import statsmodels.api as sm
import matplotlib
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from mpl_toolkits.mplot3d import Axes3D

# path of result genertaed by analysis.py
#earlier
filepath = r'//lib/UM_GDP_pop.csv'
# new, after subset each city (20171120)
filepath = r'//lib/UM_GDP_pop_shp_20180204.csv'

#get present dirctory
pwd = os.getcwd()

# Open the dataframe. GDP, pop, UM = pgu
df_gpu = pd.read_csv(pwd+ filepath, header = 0)


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Explore     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#


#  Exploratory plotting
def plot4(df_gpu, res = True, type=1):

    '''function to 
        plot industry an dcommercial count with population and GDP. if option res = Tru, then plots of residential also shown. type specifies city type - 1,2,3 
        this is just for exploration . more thngs happen later.
    
    # to run use
        plot4(df_gpu[df_gpu.Type == 3], res = True, type = 1)
    '''

    # basic plottig
    plt.figure(1)
    plt.suptitle(" Urban areas - Tier " + str(type))
    # plot GDP stuff

    # plot at 10
    y1 = df_gpu.R10
    y2 = df_gpu.C10
    y3 = df_gpu.I10
    x = df_gpu.GDP / 1000

    ax1 = plt.subplot(221)
    if res:
        plt.scatter(x, y1, c='green', marker='o', label = 'Residential')
    plt.scatter(x, y2, c='blue', marker='d', label = 'Commercial')
    plt.scatter(x, y3, c='red', marker='s', label = 'Industrial')
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('UM Area in 20km*20km')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.legend()
    # sns.pointplot("GDP", "R10", data=df_gpu)
    # sns.lmplot("GDP", "C10", data=df_gpu, hue = 'Type',  fit_reg=False)

    # plot at 20
    y4 = df_gpu.R20
    y5 = df_gpu.C20
    y6 = df_gpu.I20
    ax3 = plt.subplot(223, sharex=ax1)
    ax3.set_ylabel('UM Area in 40km*40km')
    ax3.set_xlabel('GDP in 10billion Rs')
    if res:
        plt.scatter(x, y4, c='green', marker='o', label = 'Residential')
    plt.scatter(x, y5, c='blue', marker='d', label = 'Commercial')
    plt.scatter(x, y6, c='red', marker='s', label = 'Industrial')
    ax3.set_xlim(left = 0)
    ax3.set_ylim(bottom=0)
    plt.legend()

    # populaiton stuff

    # plot at 10
    y1 = df_gpu.R10
    y2 = df_gpu.C10
    y3 = df_gpu.I10
    x = df_gpu.Population / 1000

    ax2 = plt.subplot(222, sharey=ax1)
    if res:
        plt.scatter(x, y1, c='green', marker='o', label = 'Residential')
    plt.scatter(x, y2, c='blue', marker='d', label = 'Commercial')
    plt.scatter(x, y3, c='red', marker='s', label = 'Industrial')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylim(bottom=0)
    plt.legend()

    # plot at 20
    y4 = df_gpu.R20
    y5 = df_gpu.C20
    y6 = df_gpu.I20
    x = df_gpu.Population / 1000

    ax4 = plt.subplot(224, sharex=ax2, sharey=ax3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)
    ax4.set_xlabel('Population in million')
    if res:
        plt.scatter(x, y4, c='green', marker='o', label = 'Residential')
    plt.scatter(x, y5, c='blue', marker='d', label = 'Commercial')
    plt.scatter(x, y6, c='red', marker='s', label = 'Industrial')
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.legend()

# function end
plot4(df_gpu[df_gpu.Type == 3], res = True, type = 1)





# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: Linear/Nonlinear fitting     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Functions fitting

# BUnch of fitting functions. Linear. Polynomia. Log defining fitting curves for linearizing GDP and pop
def fit_linear(x, m, c):
    return m*x + c

def fit_exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_logarithm(x, m, c ):
    return m * np.log(x) + c

def fit_power(x, a, b ):
    return a * np.power(x, b)

def fit_quadratic(x, a, b, c ):
    return a * x**2 + b*x + c


def get_fit_params(df_gpu, UMc, type, ind, fitfunc):
    # function to help explore the best fitting curve type and get its parameters
    # specify:
    # df_gpu : df containg GDP , pop and area values
    # UMc : UM class type. e.g. R20 means residentiaal fo rediauss 20km
    # type : type of the city that needs to be explored
    # ind : indiciator type: Populaiton or GDP
    # fitfunc : the type of the functgion that needs to be explored
    # returns - parameters for th4e functions
    # plots the data to visualize the give n dataset and the fitted curve

    #sort the datafrme
    df_gpu.sort_values([ind], inplace= True)
    # dependent and independent variable
    x = df_gpu[df_gpu.Type==type][ind]
    y = df_gpu[df_gpu.Type==type][UMc]

    # define dictionary of all functon abbrev
    functype = {
        'lin': fit_linear,
        'exp': fit_exponential,
        'log': fit_logarithm,
        'pow': fit_power,
        'qad': fit_quadratic
        }

    # fit the curve
    popt, pcov = curve_fit(functype[fitfunc], x, y)

    # make plot
    plt.figure()
    plt.plot(x, y, 'ko', label="observed")
    plt.plot(x, functype[fitfunc](x, *popt), 'ro-', label="fit")

    #the usual plot info
    plt.xlabel(ind)
    plt.ylabel(UMc)
    plt.legend()
    plt.show()

    return popt, pcov

#end function
get_fit_params(df_gpu, 'R10', 1, 'Population', 'pow')

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step2: Linear models only     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# beacuse I am not sure how non-linear models perform.  i will use statsmodel OLS to generate linear models.. NAd check model summary.

# Functions fitting
def smfit_linear(x):
    X = x
    return X

def smfit_exponential(x, b, c):
    X = np.exp(b - c* x)
    return X

def smfit_logarithm(x):
    X = np.log(x)
    return X

def smfit_power(x, b ):
    X = x**b
    return X

def smfit_quadratic(x ):
    X = np.column_stack((x, x**2))
    return X



def smget_fit_params(df_gpu, UMc, type, ind, fitfunc, **kwargs):
    # function to help explore the best fitting curve type and get its parameters using statmodels
    # specify:
    # df_gpu : df containg GDP , pop and area values
    # UMc : UM class type. e.g. R20 means residentiaal fo rediauss 20km
    # type : type of the city that needs to be explored
    # ind : indiciator type: Populaiton or GDP
    # fitfunc : the type of the functgion that needs to be explored
    # returns - parameters for th4e functions
    # plots the data to visualize the give n dataset and the fitted curve

    #sort the datafrme
    df_gpu.sort_values([ind], inplace= True)
    # dependent and independent variable
    x = df_gpu[df_gpu.Type==type][ind]
    y = df_gpu[df_gpu.Type==type][UMc]

    # define dictionary of all functon abbrev
    functype = {
        'lin': smfit_linear,
        'exp': smfit_exponential,
        'log': smfit_logarithm,
        'pow': smfit_power,
        'qad': smfit_quadratic
        }
    X = sm.add_constant(functype[fitfunc](x))
    # fit the curve
    res = sm.OLS(y,X).fit()

    prstd, iv_l, iv_u = wls_prediction_std(res)

    # make plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'ko', label="observed")
    ax.plot(x, res.fittedvalues, 'r^-', label=fitfunc+" fit, $R^2$ " + str('%0.2f'%res.rsquared) + '\n'+str(res.params))
    ax.plot(x, iv_u, 'r--', alpha = 0.2)
    ax.plot(x, iv_l, 'r--', alpha = 0.2)


    #the usual plot info
    plt.xlabel(ind)
    plt.ylabel(UMc)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(loc='best')
    plt.show()

    return res

#end function
res = smget_fit_params(df_gpu, 'R10', 1, 'Population', 'lin').summary()


print(res.summary())
print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('Predicted values: ', res.predict())

prstd, iv_l, iv_u = wls_prediction_std(res)


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step3: Area = f(GDP, pop)     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# 3d plot
# ridge regression


def smget_GAF(df_gpu, UMc, tier, fitfuncgdp, fitfuncpop, rotate = False, GDPpc = True):
    # function to help explore the best fitting curve type and get its parameters using statmodels
    # specify:
    # df_gpu : df containg GDP , pop and area values
    # UMc : UM class type. e.g. R20 means residentiaal fo rediauss 20km
    # type : type of the city that needs to be explored
    # ind : indiciator type: Populaiton or GDP
    # fitfunc : the type of the functgion that needs to be fit for gdp and pop based on analysis done in step1
    # returns - parameters for th4e functions
    # plots the data to visualize the give n dataset and the fitted curve

    # define dictionary of all functon abbrev
    functype = {
        'lin': smfit_linear,
        'exp': smfit_exponential,
        'log': smfit_logarithm,
        'pow': smfit_power,
        'qad': smfit_quadratic
        }

    #sort the datafrme
    df_gpu.sort_values(['Population'], inplace= True)

    # dependent and independent variable
    xgdp = df_gpu[df_gpu.Tier==tier]['GDP']
    xgdpc = df_gpu[df_gpu.Tier == tier]['GDPpc']
    xpop = df_gpu[df_gpu.Tier == tier]['Population']
    y = df_gpu[df_gpu.Tier==tier][UMc]

    #Please uncomment this if GDP is required in the model
    #x1 = functype[fitfuncgdp](xgdp)
    x1 = 0
    if GDPpc:
        x1 = functype[fitfuncgdp](xgdpc)
    x2 = functype[fitfuncpop](xpop)

    if fitfuncgdp=='lin':
        x1 = xgdp
        if GDPpc:
            x1 = xgdpc
    if fitfuncpop=='lin':
        x2 = xpop

    X = sm.add_constant(np.column_stack((x1, x2)))
    # fit the curve
    res = sm.OLS(y,X).fit()

    prstd, iv_l, iv_u = wls_prediction_std(res)

    ## Create the 3d plot -- skip reading this
    # TV/Radio grid for 3d plot
    xx1, xx2 = np.meshgrid(np.linspace(x1.min(), x1.max(), 100),
                           np.linspace(x2.min(), x2.max(), 100))
    # plot the hyperplane by evaluating the parameters on the grid
    Z = res.params[0] + res.params[1] * xx1 + res.params[2] * xx2

    # create matplotlib 3d axes
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig, azim=-115, elev=15)
    #ax = fig.gca(projection = '3d')

    # plot hyperplane
    ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0,
                           label=fitfuncgdp+','+fitfuncpop+" fit, $R^2$ " + str('%0.2f'%res.rsquared) + '\n'+str(res.params))

    # plot data points - points over the HP are white, points below are black
    resid = y - res.predict(X)
    ax.scatter(x1[resid >= 0], x2[resid >= 0], y[resid >= 0], color='black', alpha=1.0, facecolor='white', label = 'fit')
    ax.scatter(x1[resid < 0], x2[resid < 0], y[resid < 0], color='black', alpha=1.0, label = 'fit')

    # set axis labels
    if GDPpc:
        ax.set_xlabel(fitfuncgdp + ' GDP per capita (INR)')
    else:
        ax.set_xlabel(fitfuncgdp+ ' GDP (crore)')
    ax.set_ylabel(fitfuncpop +' Population (thousand)')
    ax.set_zlabel(UMc)

    # dummy polots because plt doesnt supprot legend in 3d scatter
    scatter0_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='red', marker='s')
    scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='black', marker='o', markerfacecolor='white')
    scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='black', marker='o')
    ax.legend([scatter0_proxy, scatter1_proxy, scatter2_proxy], ["Fit model, $R^2$ " + str('%0.2f'%res.rsquared) + '\n'+str(res.params),
                                                                 'fit + residue',
                                                                 'fit - residue'], numpoints=1)

    if rotate:
        for angle in range(0,360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.001)


    plt.show()

    return res

#end function

#earleier using type
res = smget_GAF(df_gpu, 'R10', 1, 'lin', 'log')
res = smget_GAF(df_gpu, 'R10', 1, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = False).summary()
res = smget_GAF(df_gpu, 'C10', 1, fitfuncgdp='lin', fitfuncpop='log', GDPpc = False).summary()
res = smget_GAF(df_gpu, 'I20', 1, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = False).summary()

res = smget_GAF(df_gpu, 'R10', 2, fitfuncgdp='log', fitfuncpop='log', GDPpc = False).summary()
res = smget_GAF(df_gpu, 'C10', 2, fitfuncgdp='log', fitfuncpop='lin', GDPpc = False).summary()
res = smget_GAF(df_gpu, 'I20', 2, fitfuncgdp='log', fitfuncpop='lin', GDPpc = False).summary()

res = smget_GAF(df_gpu, 'R20', 3, fitfuncgdp='log', fitfuncpop='log', GDPpc = False).summary()
res = smget_GAF(df_gpu, 'C20', 3, fitfuncgdp='log', fitfuncpop='lin', GDPpc = False).summary()
res = smget_GAF(df_gpu, 'I20', 3, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = False).summary()

#now after subset shape and using TIER nomenclauture (20171120)
res = smget_GAF(df_gpu, 'R00', 1, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'C00', 1, fitfuncgdp='log', fitfuncpop='lin', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'I00', 1, fitfuncgdp='log', fitfuncpop='lin', GDPpc = True).summary()

res = smget_GAF(df_gpu, 'R10', 2, fitfuncgdp='log', fitfuncpop='log', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'C10', 2, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'I10', 2, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()


#now after subset shape and using TIER nomenclauture and assigning GDP/pop from correct dates (2080204)
res = smget_GAF(df_gpu, 'R00', 1, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'C00', 1, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'I00', 1, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()

res = smget_GAF(df_gpu, 'R10', 2, fitfuncgdp='log', fitfuncpop='log', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'C10', 2, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()
res = smget_GAF(df_gpu, 'I00', 2, fitfuncgdp='lin', fitfuncpop='lin', GDPpc = True).summary()



def smfit_logarithm2(x):
    X = np.log(x)
    return X
