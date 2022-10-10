# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np
import torch
from math import ceil

from .util import chkM
from .fft import fftp, ifftp


def nsgtf_sl(f_slices, g, wins, nn, M=None, matrixform=False, real=False, reducedform=0, measurefft=False, multithreading=False, device="cpu"):
    M = chkM(M,g)
    dtype = g[0].dtype
    
    fft = fftp(measure=measurefft, dtype=dtype)
    ifft = ifftp(measure=measurefft, dtype=dtype)
    
    if real:
        assert 0 <= reducedform <= 2
        sl = slice(reducedform,len(g)//2+1-reducedform)
    else:
        sl = slice(0,None)
    
    maxLg = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(M[sl],g[sl]))
    temp0 = None
    
    mmap = map

    loopparams = []
    for mii,gii,win_range in zip(M[sl],g[sl],wins[sl]):
        Lg = len(gii)
        col = int(ceil(float(Lg)/mii))
        assert col*mii >= Lg
        assert col == 1

        p = (mii,win_range,Lg,col)
        loopparams.append(p)

    jagged_indices = [None]*len(loopparams)

    ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, maxLg-gii.shape[0])) for gii in g[sl]]
    giis = torch.conj(torch.cat(ragged_giis))

    ft = fft(f_slices)

    Ls = f_slices.shape[-1]
    #print("yo",nn, Ls)
    assert nn == Ls

    if matrixform:
        c = torch.zeros(*f_slices.shape[:2], len(loopparams), maxLg, dtype=ft.dtype, device=torch.device(device))

        for j, (mii,win_range,Lg,col) in enumerate(loopparams):
            t = ft[:, :, win_range]*torch.fft.fftshift(giis[j, :Lg])

            sl1 = slice(None,(Lg+1)//2)
            sl2 = slice(-(Lg//2),None)

            c[:, :, j, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
            c[:, :, j, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2

        return ifft(c)
    else:
        block_ptr = -1
        bucketed_tensors = []
        ret = []

        for j, (mii,win_range,Lg,col) in enumerate(loopparams):
            
            c = torch.zeros(*f_slices.shape[:2], 1, Lg, dtype=ft.dtype, device=torch.device(device))

            t = ft[:, :, win_range]*torch.fft.fftshift(giis[j, :Lg])

            sl1 = slice(None,(Lg+1)//2)
            sl2 = slice(-(Lg//2),None)

            c[:, :, 0, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
            c[:, :, 0, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2

            # start a new block
            if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != Lg:
                bucketed_tensors.append(c)
                block_ptr += 1
            else:
                # concat block to previous contiguous frequency block with same time resolution
                bucketed_tensors[block_ptr] = torch.cat([bucketed_tensors[block_ptr], c], dim=2)

        # bucket-wise ifft
        for bucketed_tensor in bucketed_tensors:
            ret.append(ifft(bucketed_tensor))

        return ret
        

# non-sliced version
def nsgtf(f, g, wins, nn, M=None, real=False, reducedform=0, measurefft=False, multithreading=False, matrixform=False, device="cpu"):
    ret = nsgtf_sl(torch.unsqueeze(f[0], dim=0), g, wins, nn, M=M, real=real, reducedform=reducedform, measurefft=measurefft, multithreading=multithreading, device=device, matrixform=matrixform)
    #return torch.squeeze(ret, dim=0)
    return ret
