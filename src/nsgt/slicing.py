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
from .util import hannwin
from .reblock import reblock
from itertools import chain, cycle


def makewnd(sl_len, tr_area, device="cpu"):
    hhop = sl_len//4
    htr = tr_area//2
    # build window function within one slice (centered with transition areas around sl_len/4 and 3*sl_len/4    
    w = hannwin(2*tr_area, device=device)  # window is shifted
    tw = torch.empty(sl_len, dtype=torch.float32, device=torch.device(device))
    tw[:hhop-htr] = 0
    tw[hhop-htr:hhop+htr] = w[tr_area:]
    tw[hhop+htr:3*hhop-htr] = 1
    tw[3*hhop-htr:3*hhop+htr] = w[:tr_area]
    tw[3*hhop+htr:] = 0
    return tw


def slicing(f, sl_len, tr_area, device="cpu"):
    if tr_area%2 != 0:
        raise ValueError("Transition area 'tr_area' must be modulo 2")
    if sl_len%4 != 0:
        raise ValueError("Slice length 'sl_len' must be modulo 4")
    
    hhop = sl_len//4  # half hopsize

    tw = makewnd(sl_len, tr_area, device=device)
    # four parts of slice with centered window function
    tw = [tw[o:o+hhop] for o in range(0, sl_len, hhop)]
    
    # stream of hopsize/2 blocks with leading and trailing zero blocks
    fseq = reblock(f, hhop, dtype=torch.float32, fulllast=True, padding=0., multichannel=True, device=device)
    
    # get first block to deduce number of channels
    fseq0 = next(fseq)
    chns = len(fseq0)
    pad = torch.zeros((chns,hhop), dtype=fseq0.dtype, device=torch.device(device))
    # assemble a stream of front padding, already retrieved first block, the block stream and some tail padding
    fseq = chain((pad,pad,fseq0), fseq, (pad,pad,pad))

    slices = [[slice(hhop*((i+3-k*2)%4), hhop*((i+3-k*2)%4+1)) for i in range(4)] for k in range(2)]
    slices = cycle(slices)
    
    past = []
    for fi in fseq:
        past.append(fi)
        if len(past) == 4:
            f_slice = torch.empty((chns,sl_len), dtype=fi.dtype, device=torch.device(device))
            sl = next(slices)
            for sli,pi,twi in zip(sl, past, tw):
                f_slice[:,sli] = pi    # signal
                f_slice[:,sli] *= twi  # multiply with part of window function
            yield f_slice
            past = past[2:]  # pop the two oldest slices
