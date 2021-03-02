#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:55:31 2020

@author: emmareid
"""
from math import log10
from math import sqrt
import numpy as np

#This code computes the Peak Signal to Noise Ratio.
#PSNR assumes that gt and recon are 2D image arrays in the range [0,1].

def mse(groundtrutharr, imarr):
    imvec = imarr.reshape(-1)
    groundtruthvec = groundtrutharr.reshape(-1)
    mse = np.mean(np.power(imvec-groundtruthvec,2))
    rmse = sqrt(mse)
    nrmse = sqrt(mse)/(np.mean(np.power(groundtruthvec,2)))
    return [mse, rmse, nrmse]
#    return('MSE = ', mse, 'RMSE = ', rmse, 'NRMSE = ', nrmse)
    
def error(arr1, arr2):
    #Error (x,v)
    vec1 = arr1.reshape(-1)
    vec2 = arr2.reshape(-1)
    err = sqrt(np.sum(np.power(vec2-vec1,2)))
    errnorm = err/sqrt(np.sum(np.power(vec2,2)))
    return errnorm

def psnr(gt,recon):
    MSE = mse(gt,recon)[0]
    out = 10*log10(1/MSE)
    return out

