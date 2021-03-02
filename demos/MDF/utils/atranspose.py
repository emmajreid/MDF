#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:44:04 2018

@author: emmareid
"""
import numpy as np
import torch
import cv2
from skimage.io import imread
import scipy

#This code performs upsampling by block replication in L x L grids.
#This maps an N x N image to an NL x NL image.

def ATgen(imagearr, L):
    
    outarr = np.zeros((imagearr.shape[0]*L, imagearr.shape[1]*L))
    for i in range(0,imagearr.shape[0]):
        for j in range(0, imagearr.shape[1]):
            outarr[L*i:L*(i+1), L*j:L*(j+1)] = imagearr[i,j]

    return outarr




