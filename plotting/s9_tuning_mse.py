#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 22 August 2022

Create figure S7: Plot MSE from tuning.

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
parameterLabels = ['N$_{\mathrm{components}}$', '$\\alpha$', '$\\alpha$']
parameterSymbols = ['N', '$\\alpha$', '$\\alpha$']
metricLabels = ['Forced', 'Unforced', 'Total', 'Mean']
metricIndices = [1, 2, 0, 3]
metricColors = ['red', 'blue', 'black', 'gray']
logXAxis = [False, True, False]
labelLocs = [[5, 0.3e-3], [10**3, 0.2e-3], [100, 0.25e-3]]

# %% Figure block
plt.figure(figsize=(8.5, 2.75), dpi=300)
# loop over ML methods
for IM, method in enumerate(methods):
    # create subplot
    plt.subplot(1, 3, IM+1)
    # load data
    tuningDict = pickle.load(open('../data/tuning/' + method + '.pickle', 'rb'))
    # get tuning parameter values
    tuningParameter = list(tuningDict['MSE'].keys())
    # initialize output MSE vector
    MSE = []
    # loop over parameters to get MSE values
    for t in tuningParameter:
        row = list(np.mean(tuningDict['MSE'][t], axis=0))
        row = row + [np.mean(row)]
        MSE.append(row)
    # cast to array
    MSE = np.array(MSE)
    for i in range(4):
        mse = MSE[:, metricIndices[i]]
        # get minimum MSE value
        inds = np.where(np.min(mse) == mse)[0][0]
        # retain the minimum MSE value for the mean (index 3)
        if i == 3:
            MI = inds
        # plot mse values for trend type
        plt.plot(tuningParameter, mse, '.-', color=metricColors[i])
        # denote value for minimum in mean MSE
        plt.plot(tuningParameter[inds], mse[inds], 'x', color=metricColors[i])
        # use log axis if specified
        if logXAxis[IM]:
            plt.xscale('log')
    # label parameter value that corresponds to minimum in MSE
    plt.text(labelLocs[IM][0], labelLocs[IM][1], parameterLabels[IM] + ' = ' + str(tuningParameter[MI]), color='darkgray')
    # labels
    plt.title(methodLabels[IM])
    plt.xlabel(parameterLabels[IM])
    if IM == 0:
        plt.ylabel('Mean Squared Error')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ensure same y-range in each plot
    plt.ylim(0, 1.75e-3)
    # reduce y-axis scaling font size
    ax = plt.gca()
    ax.yaxis.offsetText.set_fontsize(8)
# custom legend
for i in range(4):
    plt.text(5, 1.6e-3-i*1.5e-4, metricLabels[i], color=metricColors[i])
# square up and save results
plt.tight_layout()
plt.savefig('../figures/s9_tuning_mse.pdf', bbox_inches='tight')
plt.savefig('../figures/s9_tuning_mse.png', bbox_inches='tight')
plt.show()

