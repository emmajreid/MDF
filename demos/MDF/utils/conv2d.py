#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:08:38 2018

@author: emmareid
"""
import torch
import cv2
import numpy as np

#This code performs image decimation by taking the sum of the pixel values in a L x L grid.
#from a N x N image, it returns an N/L x N/L image.

class Conv2D:

    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.i = in_channel
        self.out = o_channel
        self.ker = kernel_size
        self.stride = stride
        self.mode = mode
        
    def forward(self, input_image, L):
        #Here using the // operator under the assumption that we'll only be applying
        #kernels that result in no empty columns.
        m = self.ker
        ##Just K1
        if self.out == 1 and self.mode == 'known':
            kernel = torch.FloatTensor(torch.ones((L,L)))
            kernel_list = [kernel]
            kernel = torch.stack(kernel_list)
        [M,N] = input_image.size()
        stride = self.stride
        [n,p,q] = kernel.size()
        temp = torch.tensor([0])
        out_image = torch.zeros((M-m)//stride+1,(N-m)//stride+1, n)

        for i in range(0,M-m+1,stride):
            for j in range(0,N-m+1,stride):
                for l in range(0,n):
        #Multiply the input image with the kernel
                        sub_img = input_image[i:i+m,j:j+m]
                        temp = torch.FloatTensor([torch.sum(sub_img * kernel[l,:])])
                        out_image[i//stride,j//stride] = out_image[i//stride,j//stride]+(temp)
                        temp=torch.tensor([0])
        return out_image
    
def Agen(imagearr, L):
    img = torch.from_numpy(imagearr)
    img = img.float()
    params = Conv2D(1,1,L,L,'known')
    out_image= params.forward(img,L)
    out_image = out_image[:,:,0]
    return out_image.numpy()
