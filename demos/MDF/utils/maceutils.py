#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:01:55 2019

@author: emmareid
"""
# General imports
import os
import numpy as np
import cv2

from utils import conv2d
from utils import atranspose

import torch
from models.network_dncnn import DnCNN as net

def mace(LR,numagents,c,args):
    '''
    This is a Python implementation of the MACE framework written about in 4D X-Ray CT Reconstruction using Multi-Slice Fusion by
    S. Majee et. al (2019). Fundamentally it consists of stacked application of the prior and forward models to our image vectors (L)
    followed by a weighted averaging (G) and an update step.
    
    Inputs:
        LR: Low resolution image scaled to be in [0,1]
        numagents: number of agents in F
        c: c = varlam/(vary/args.SRval + varlam)
        args: Dictionary containing command line arguments
            SRval: Super resolution factor 
            beta: User-defined parameter for calculating \sigma^2_\lambda, our Lagrangian parameter, and c.
              This parameter plays a role when sigy is assumed to be nonzero.
            iterstop: User-defined parameter for the stopping number of Mann iterations in the MACE framework.
            sign: Noise level that the denoising prior is trained to remove. For all provided prior models,
              sign = 0.1.
            sigy: Assumed level of noise in the ground truth image
            mu:   This is the weight of the forward model in the MACE framework, in [0,1]
               mu = 0 -----> Only considering the output of the prior model. 
               mu = 0.5 ---> Equal consideration of the outputs of the forward and prior models.
               mu = 1 -----> Only considering the output of the forward model.
            rho: This is the step size that we take throughout the MACE framework, generally in (0,1).
             Larger rho tends to lead to faster convergence.
            model_dir: Path to the directory containing the saved priors.
            model_name: Name of the denoising prior model.
            hrname: Name of the HR ground truth image.
            forwards: Choice of forward model, either using the standard or RAP.
            denoisers: Choice of denoising prior model.
    Outputs: 
        OutW: Super resolved image
        PSNR: Array containing the PSNR values for each step of the algorithm.
        MACE: Array containing the MACE errors for each step of the algorithm.
    '''   
    
    # Generate initial guess as starting point for algorithm, write to an image,
    # and use it to initialize X and W.

    init = cv2.resize(LR,None, fx = args.SRval, fy = args.SRval, interpolation = cv2.INTER_CUBIC)

    mdim,ndim = init.shape
    cv2.imwrite('images/results/init.png', init*255)
    init = init.reshape(-1)
    init = init.reshape(init.shape[0],1)
    X = np.tile(init, (1,numagents))
    W = np.copy(X)
    
    # Initialize vectors for metric analysis.
    PSNR = np.zeros(args.iterstop)
    maceerr = np.zeros(args.iterstop)

    # MACE Framework
    for i in range(0, args.iterstop):
        print("Currently on Iteration", i)
        X = L(W,X,numagents, mdim,ndim,c, LR, args)
        Z = G(2*X-W,numagents,args.mu)
        W = W+2*args.rho*(Z-X)

     # Save metrics for this iteration to the vector. 
        Y=np.copy(W)
        maceerr[i]=(1/args.sign)*np.linalg.norm(G(Y,numagents,args.mu)-L(Y,Y,numagents, mdim, ndim, c, LR, args))/np.linalg.norm(G(Y,numagents,args.mu))


    return G(W,numagents, args.mu)[:,0], maceerr

#L takes in all of the state vectors and applies denoisers to the first k state
#vectors and the forward model to the last state vector.

def L(W,X,numagents, mdim,ndim,c, LR, args):
    # Prior Model Application
    Lout = np.copy(X)
    for i in args.denoisers:
        iternum = 0
        xi =np.copy(W[:,iternum])
        xi = xi.reshape(mdim,ndim)
            
        if (args.model_name == 'dncnn_25.pth'):
            denoiser = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
        
        else:
            denoiser = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR')
        denoiser.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)), strict=True)
        denoiser.eval()
        
        xi = torch.FloatTensor(xi)
        if torch.cuda.is_available()==True:
            denoiser.cuda()
        if torch.cuda.is_available() == True:
            imagei = (xi.cuda()).reshape(1,1,mdim,ndim)
        else:
            imagei = xi.reshape(1,1,mdim,ndim)
        x = denoiser(imagei)
        if torch.cuda.is_available() == True:
            x = x.cpu().detach().numpy()
        else:
            x = x.detach().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        x = x.reshape(mdim*ndim)
        Lout[:,iternum] = np.copy(x)
            
        iternum += 1
    wi =np.copy(W[:,iternum])
    wi = wi.reshape(mdim,ndim)
    
    #Forward Model Application
    x = forward(wi, LR, c, args)
    x = x.reshape(mdim*ndim)
    Lout[:,iternum] = np.copy(x)
    return Lout
    
# G outputs the weighted average of the state vectors, weighting the Forward Model's vector with mu and all others with (1-mu)/(N-1)
# where N is the number of agents.
def G(X,numagents,mu):
    if numagents ==2:
        x = mu*X[:,numagents-1,np.newaxis]+((1-mu)/(numagents-1))*np.sum(X[:,0:numagents-1,np.newaxis],1)
    else:
        x = mu*X[:,numagents-1]+(1-mu)/((numagents-1))*np.sum(X[:,0:numagents-1],1)
    x = x.reshape(x.shape[0],1)
    Xnew = np.tile(x, (1,numagents))
    return Xnew

def forward(wi, LR,c,args):
    
    new = LR-(1/args.SRval**2)*conv2d.Agen(wi,args.SRval)
    
    # 0 is for AT, 1 for bicubic
    for i in args.forwards:
        if i==0:
            filt = atranspose.ATgen(new,args.SRval)    
        elif i==1:
            filt = cv2.resize(new,None,fx=args.SRval,fy=args.SRval,interpolation=cv2.INTER_CUBIC)
        else:
            print("Invalid filter choice")
            break
    out = wi+c*filt
    if args.sigy==0:
        out = np.clip(out, a_min=0, a_max=None)
    return out