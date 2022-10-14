#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 22 August 2022

Create figure S8: prediction sensitivity to tuning parameters.

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
import numpy as np

# %% Parameters
methods = ['pls', 'ridge', 'nn']
methodLabels = ['PLS Regression', 'Ridge Regression', 'Neural Network']
parameterLabels = ['N$_{components}$', '$\\alpha$', '$\\alpha$']
parameterSymbols = ['N', '$\\alpha$', '$\\alpha$']
metricLabels = ['Forced', 'Unforced', 'Total']
metricIndices = [1, 2, 0]
metricColors = ['red', 'blue', 'black', 'gray']
logXAxis = [False, True, False]
labelLocs = [[5, 0.3e-3], [10**3, 0.2e-3], [20, 0.25e-3]]

# %% Figure block
plt.figure(figsize=(8.5, 2.75), dpi=300)

for IM, method in enumerate(methods):
    # create subplot
    plt.subplot(1, 3, IM+1)
    # load data
    tuningDict = pickle.load(open('../data/tuning/' + method + '.pickle', 'rb'))
    # get tuning parameter values
    tuningParameter = list(tuningDict['MSE'].keys())
    # initialize output MSE and prediction vectors
    MSE = []
    PRED = []
    # loop over parameters to get prediction values and mean MSE value
    for t in tuningParameter:
        PRED.append(tuningDict['PREDS'][t])
        row = list(np.mean(tuningDict['MSE'][t], axis=0))
        row = np.mean(row)
        MSE.append(row)
    # cast to array
    MSE = np.array(MSE)
    PRED = np.array(PRED)
    # get minimum mean MSE value
    inds = np.where(np.min(MSE) == MSE)[0][0]
    for i in range(3):
        # get index for trend type
        MI = metricIndices[i]
        # plot trend values for trend type
        plt.plot(tuningParameter, PRED[:, MI], '.-', color=metricColors[i])
        # denote value for minimum in mean MSE
        plt.plot(tuningParameter[inds], PRED[inds, MI], 'x', color=metricColors[i])
        # use log axis if specified
        if logXAxis[IM]:
            plt.xscale('log')
    # print prediction corresponding to minimum MSE value
    print(methodLabels[IM], PRED[inds, :])
    # labels
    plt.title(methodLabels[IM])
    plt.xlabel(parameterLabels[IM])
    if IM == 0:
        plt.ylabel('Tropical TMT Trend [K decade$^{-1}$]')
    # ensure same y-range in each plot
    plt.ylim(-0.15, 0.3)
    # reduce y-axis scaling font size
    ax = plt.gca()
    ax.yaxis.offsetText.set_fontsize(8)
# square up and save results
plt.tight_layout()
plt.savefig('../figures/s10_prediction_sensitivity.pdf', bbox_inches='tight')
plt.savefig('../figures/s10_prediction_sensitivity.png', bbox_inches='tight')
plt.show()

