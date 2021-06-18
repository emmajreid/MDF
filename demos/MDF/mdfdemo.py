#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:01:55 2019

@author: emmareid
"""
# General imports
import sys
import os
import numpy as np
import argparse
import cv2

from utils import conv2d
from utils import atranspose
from utils import psnr
from utils import maceutils

import torch
from models.network_dncnn import DnCNN as net

if __name__ == '__main__':
    
    '''
    Variable Definitions:
        beta: User-defined parameter for calculating \sigma^2_\lambda, our Lagrangian parameter, and c.
              This parameter plays a role when sigy is assumed to be nonzero.
        iter: User-defined parameter for the number of Mann iterations in the MACE framework.
              Complete convergence is usually achieved by 200 iterations.
        sign: Noise level that the denoising prior is trained to remove. For all provided prior models,
              sign = 0.1.
        sigy: Assumed level of noise in the ground truth image
        mu:   This is the weight of the forward model in the MACE framework, in [0,1]
              mu = 0 -----> Only considering the output of the prior model. 
              mu = 0.5 ---> Equal consideration of the outputs of the forward and prior models.
              mu = 1 -----> Only considering the output of the forward model.
        rho: This is the step size that we take throughout the MACE framework, generally in (0,1).
             Larger rho tends to lead to faster convergence.
    '''
    
    parser = argparse.ArgumentParser(description="Gather P&P input parameters.")
    parser.add_argument('--SRval', type=int, default=4,help='Super-Resolution factor')
    parser.add_argument('--beta', type=float, default= 0.5, help='Regularization factor')
    parser.add_argument('--iterstop', type=int, default=1, help='Number of iterations to run')
    parser.add_argument('--sign', type=float,default=0.1, help='Noise level trained to remove')
    parser.add_argument('--sigy', type=float, default=0, help='Noise level in image')
    parser.add_argument('--mu', type=float, default=0.5, help='Weighting factor')
    parser.add_argument('--rho', type=float, default=0.5, help='Convergence factor')
    
    parser.add_argument('--model_dir', default=os.path.join('priors'), help='directory of the model')
    
    #Options for model names are dncnn_25.pth and MDF.pth
    parser.add_argument('--model_name', default='nano.pth', type=str, help='the model name')
    parser.add_argument('--hrname', default='nanotest.png', type=str, help='the HR image name')
    
    # Choices for forward and prior models should be entered as arrays separated by commas
    # Currently there is only one option for a prior model, but this will be updated in the future.
    parser.add_argument('forwards', nargs = '*', default = [1]) # 0 is for AT, 1 for bicubic
    parser.add_argument('denoisers', nargs = '*', default = [1])
    parser.add_argument('--SRmode', type=bool, default=False, help="Whether running in SR mode")
    
    # Read in the user-defined arguments.
    args = parser.parse_args()

    # Name of high-resolution ground truth.
    sys.stdout.flush()
    base_path = os.path.dirname(os.path.relpath(__file__))
    testimg_file = os.path.join(base_path, 'images/')
    lrimg_file = os.path.join(base_path, 'images/LR images')
    resultsimg_file = os.path.join(base_path, 'images/results')
    
    lrname = 'lrL='+str(args.SRval)+str(args.hrname)+'noise'+str(args.sigy)+'.png'

    # Read in the high-resolution ground truth.
    gt = cv2.imread(os.path.join(testimg_file,args.hrname),0)/255
    
    # Initialize synthetic low-resolution image.
    
    if os.path.exists('LR Images/lrL='+str(args.SRval)+str(args.hrname)+'noise'+str(args.sigy)+'.png'):
        lr = cv2.imread('LR Images/lrL='+str(args.SRval)+str(args.hrname)+'noise'+str(args.sigy)+'.png',0)/255
    else:
        lr = conv2d.Agen(gt,args.SRval)/(args.SRval*args.SRval)+np.random.normal(0,args.sigy,(gt.shape[0]//args.SRval,gt.shape[1]//args.SRval))
    lr = np.float64(lr)
    cv2.imwrite(os.path.join(lrimg_file, lrname), lr*255)
    LR = cv2.imread(os.path.join(lrimg_file, lrname), 0)/255

    # Initialize other parameters.
    varn = args.sign**2
    vary = args.sigy**2
    varlam = varn/args.beta
    c = varlam/(vary/args.SRval + varlam)
    numagents = len(args.denoisers) + len(args.forwards)
    
    args.GT = gt
    # Run the MACE algorithm.
    outW, PSNR, maceerr= maceutils.mace(LR,numagents, c, args)
    cv2.imwrite(os.path.join(resultsimg_file,str(args.iterstop)+'iters-DnCNNmaceout'+str(args.mu)+'.noise'+str(args.sign)+'.png'),outW.reshape(LR.shape[0]*args.SRval,LR.shape[0]*args.SRval)*255)

    # Load images for metric purposes
    gt = cv2.imread(os.path.join(testimg_file,args.hrname),0)
    srx = cv2.imread(os.path.join(resultsimg_file,str(args.iterstop)+'iters-DnCNNmaceout'+str(args.mu)+'.noise'+str(args.sign)+'.png'),0)
    print("Path to the reconstruction is: ", resultsimg_file)


    # Calculate PSNR for the final reconsruction and return our convergence metric.
    ps = psnr.psnr(gt/255,srx/255)
    print("PSNR for Our Reconstruction: ", ps)
    print("MACE Error for Our Reconstruction: ", maceerr[-1])
    
    #Display our reconstruction
    print('Press 1 to close pop-up image')
    cv2.imshow('Reconstruction', srx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)





    
    