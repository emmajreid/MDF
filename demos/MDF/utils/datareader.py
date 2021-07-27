#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:37:40 2019

@author: emmajreid
"""

#Simple code to extract patches of a given size (patchsize) and number (patchmax) 
#from a given image hrimg. All images are saved in the current directory.

import cv2
import numpy as np
import sklearn.feature_extraction as feature_extraction

def make_patches(img, patchsize, patchmax):
    arr = feature_extraction.image.extract_patches_2d(img, (patchsize,patchsize),patchmax)
    return arr

hrimg = cv2.imread('bm3decolival.png',2)
patchsize = 180
patchmax = 100
outarr = make_patches(hrimg, patchsize, patchmax)
np.save('newpatches.npy', outarr)

loadver = np.load('newpatches.npy')
num, h, w= loadver.shape

for i in range(0,num):
        tempimg =loadver[i,:,:]
        if (i==0):
            cv2.imwrite('test_'+str(num).rjust(3,'0')+'.png',tempimg)
        else:
            cv2.imwrite('test_'+str(i).rjust(3,'0')+'.png', tempimg)

