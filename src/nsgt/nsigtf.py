# -*- coding: utf-8

"""
Thomas Grill, 2011-2015
--
Original matlab code comments follow:

NSIGTF.N - Gino Velasco 24.02.11

fr = nsigtf(c,gd,shift,Ls)

This is a modified version of nsigt.m for the case where the resolution 
evolves over frequency.

Given the cell array 'c' of non-stationary Gabor coefficients, and a set 
of windows and frequency shifts, this function computes the corresponding 
inverse non-stationary Gabor transform.

Input: 
          c           : Cell array of non-stationary Gabor coefficients
          gd          : Cell array of Fourier transforms of the synthesis 
                        windows
          shift       : Vector of frequency shifts
          Ls          : Length of the analyzed signal

Output:
          fr          : Synthesized signal

If a non-stationary Gabor frame was used to produce the coefficients 
and 'gd' is a corresponding dual frame, this function should give perfect 
reconstruction of the analyzed signal (up to numerical errors).

The inverse transform is computed by simple 
overlap-add. For each entry of the cell array c,
the coefficients of frequencies around a certain 
position in time, the Fourier transform
is taken, giving 'frequency slices' of a signal.
These slices are added onto each other with an overlap
depending on the window lengths and positions, thus
(re-)constructing the frequency side signal. In the
last step, an inverse Fourier transform brings the signal
back to the time side.

More information can be found at:
http://www.univie.ac.at/nonstatgab/

Edited by Nicki Holighaus 01.03.11
"""

import numpy as np
import time
import torch
from itertools import chain
from .fft import fftp, ifftp, irfftp
    

#@profile
def nsigtf_sl(cseq, gd, wins, nn, Ls=None, real=False, reducedform=0, matrixform=False, measurefft=False, multithreading=False, device="cpu"):
    dtype = gd[0].dtype

    fft = fftp(measure=measurefft, dtype=dtype)
    ifft = irfftp(measure=measurefft, dtype=dtype) if real else ifftp(measure=measurefft, dtype=dtype)

    if real:
        ln = len(gd)//2+1-reducedform*2
        if reducedform:
            sl = lambda x: chain(x[reducedform:len(gd)//2+1-reducedform],x[len(gd)//2+reducedform:len(gd)+1-reducedform])
        else:
            sl = lambda x: x
    else:
        ln = len(gd)
        sl = lambda x: x
        
    maxLg = max(len(gdii) for gdii in sl(gd))

    ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, maxLg-gdii.shape[0])) for gdii in sl(gd)]
    gdiis = torch.conj(torch.cat(ragged_gdiis))

    if not matrixform:
        #print(cseq)
        assert type(cseq) == list
        nfreqs = 0
        for i, cseq_tsor in enumerate(cseq):
            cseq_dtype = cseq_tsor.dtype
            cseq[i] = fft(cseq_tsor)
            nfreqs += cseq_tsor.shape[2]
        cseq_shape = (*cseq_tsor.shape[:2], nfreqs)
    else:
        assert type(cseq) == torch.Tensor
        cseq_shape = cseq.shape[:3]
        cseq_dtype = cseq.dtype
        fc = fft(cseq)

    fr = torch.zeros(*cseq_shape[:2], nn, dtype=cseq_dtype, device=torch.device(device))  # Allocate output
    temp0 = torch.empty(*cseq_shape[:2], maxLg, dtype=fr.dtype, device=torch.device(device))  # pre-allocation

    fbins = cseq_shape[2]

    loopparams = []
    for gdii,win_range in zip(sl(gd), sl(wins)):
        Lg = len(gdii)
        wr1 = win_range[:(Lg)//2]
        wr2 = win_range[-((Lg+1)//2):]
        p = (wr1,wr2,Lg)
        loopparams.append(p)

    # The overlap-add procedure including multiplication with the synthesis windows
    #tart=time.time()
    if matrixform:
        for i,(wr1,wr2,Lg) in enumerate(loopparams[:fbins]):
            t = fc[:, :, i]

            r = (Lg+1)//2
            l = (Lg//2)

            t1 = temp0[:, :, :r]
            t2 = temp0[:, :, Lg-l:Lg]

            t1[:, :, :] = t[:, :, :r]
            t2[:, :, :] = t[:, :, maxLg-l:maxLg]

            temp0[:, :, :Lg] *= gdiis[i, :Lg] 
            temp0[:, :, :Lg] *= maxLg

            fr[:, :, wr1] += t2
            fr[:, :, wr2] += t1
    else:
        # frequencies are bucketed by same time resolution
        fbin_ptr = 0
        for i, fc in enumerate(cseq):
            Lg_outer = fc.shape[-1]

            nb_fbins = fc.shape[2]
            for i,(wr1,wr2,Lg) in enumerate(loopparams[fbin_ptr:fbin_ptr+nb_fbins][:fbins]):
                freq_idx = fbin_ptr+i

                assert Lg == Lg_outer
                t = fc[:, :, i]

                r = (Lg+1)//2
                l = (Lg//2)

                t1 = temp0[:, :, :r]
                t2 = temp0[:, :, Lg-l:Lg]

                t1[:, :, :] = t[:, :, :r]
                t2[:, :, :] = t[:, :, Lg-l:Lg]

                temp0[:, :, :Lg] *= gdiis[freq_idx, :Lg] 
                temp0[:, :, :Lg] *= Lg

                fr[:, :, wr1] += t2
                fr[:, :, wr2] += t1
            fbin_ptr += nb_fbins

    #end=time.time()
    #rint("in for loop",end-start)
    ftr = fr[:, :, :nn//2+1] if real else fr
    sig = ifft(ftr, outn=nn)
    sig = sig[:, :, :Ls] # Truncate the signal to original length (if given)
    return sig


# non-sliced version
def nsigtf(c, gd, wins, nn, Ls=None, real=False, reducedform=0, measurefft=False, matrixform=False, multithreading=False, device="cpu"):
    #ret = nsigtf_sl(torch.unsqueeze(c[0], dim=0), gd, wins, nn, Ls=Ls, real=real, reducedform=reducedform, measurefft=measurefft,matrixform=matrixform, multithreading=multithreading, device=device)
        
    #tartinv=time.time()
    ret = nsigtf_sl(c[0], gd, wins, nn, Ls=Ls, real=real, reducedform=reducedform, measurefft=measurefft,matrixform=matrixform, multithreading=multithreading, device=device)
    #ndinv=time.time()
    #rint("whole inverse",endinv-startinv)
    return ret
