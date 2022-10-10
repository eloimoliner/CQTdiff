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
from math import exp, floor, ceil, pi


def hannwin(l, device="cpu"):
    r = torch.arange(l,dtype=float, device=torch.device(device))
    r *= np.pi*2./l
    r = torch.cos(r)
    r += 1.
    r *= 0.5
    return r


def blackharr(n, l=None, mod=True, device="cpu"):
    if l is None: 
        l = n
    nn = (n//2)*2
    k = torch.arange(n, device=torch.device(device))
    if not mod:
        bh = 0.35875 - 0.48829*torch.cos(k*(2*pi/nn)) + 0.14128*torch.cos(k*(4*pi/nn)) -0.01168*torch.cos(k*(6*pi/nn))
    else:
        bh = 0.35872 - 0.48832*torch.cos(k*(2*pi/nn)) + 0.14128*torch.cos(k*(4*pi/nn)) -0.01168*torch.cos(k*(6*pi/nn))
    bh = torch.hstack((bh,torch.zeros(l-n,dtype=bh.dtype,device=torch.device(device))))
    bh = torch.hstack((bh[-n//2:],bh[:-n//2]))
    return bh

def blackharrcw(bandwidth,corr_shift):
    flip = -1 if corr_shift < 0 else 1
    corr_shift *= flip
    
    M = np.ceil(bandwidth/2+corr_shift-1)*2
    win = np.concatenate((np.arange(M//2,M), np.arange(0,M//2)))-corr_shift
    win = (0.35872 - 0.48832*np.cos(win*(2*np.pi/bandwidth))+ 0.14128*np.cos(win*(4*np.pi/bandwidth)) -0.01168*np.cos(win*(6*np.pi/bandwidth)))*(win <= bandwidth)*(win >= 0)

    return win[::flip],M


def cont_tukey_win(n, sl_len, tr_area):
    g = np.arange(n)*(sl_len/float(n))
    g[np.logical_or(g < sl_len/4.-tr_area/2., g > 3*sl_len/4.+tr_area/2.)] = 0.
    g[np.logical_and(g > sl_len/4.+tr_area/2., g < 3*sl_len/4.-tr_area/2.)] = 1.
    #
    idxs = np.logical_and(g >= sl_len/4.-tr_area/2., g <= sl_len/4.+tr_area/2.)
    temp = g[idxs]
    temp -= sl_len/4.+tr_area/2.
    temp *= pi/tr_area
    g[idxs] = np.cos(temp)*0.5+0.5
    #
    idxs = np.logical_and(g >= 3*sl_len/4.-tr_area/2., g <= 3*sl_len/4.+tr_area/2.)
    temp = g[idxs]
    temp += -3*sl_len/4.+tr_area/2.
    temp *= pi/tr_area
    g[idxs] = np.cos(temp)*0.5+0.5
    #
    return g

def tgauss(ess_ln, ln=0):
    if ln < ess_ln: 
        ln = ess_ln
    #
    g = np.zeros(ln, dtype=float)
    sl1 = int(floor(ess_ln/2))
    sl2 = int(ceil(ess_ln/2))+1
    r = np.arange(-sl1, sl2) # (-floor(ess_len/2):ceil(ess_len/2)-1)
    r = np.exp((r*(3.8/ess_ln))**2*-pi)
    r -= exp(-pi*1.9**2)
    #
    g[-sl1:] = r[:sl1]
    g[:sl2] = r[-sl2:]
    return g

def _isseq(x):
    try:
        len(x)
    except TypeError:
        return False
    return True        

def chkM(M, g):
    if M is None:
        M = np.array(list(map(len, g)))
    elif not _isseq(M):
        M = np.ones(len(g), dtype=int)*M
    return M


def calcwinrange(g, rfbas, Ls, device="cpu"):
    shift = np.concatenate(((np.mod(-rfbas[-1],Ls),), rfbas[1:]-rfbas[:-1]))
    
    timepos = np.cumsum(shift)
    nn = timepos[-1]
    timepos -= shift[0] # Calculate positions from shift vector
    
    wins = []
    for gii,tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = torch.arange(-(Lg//2)+tpii, Lg-(Lg//2)+tpii, dtype=int, device=torch.device(device))
        win_range %= nn

        wins.append(win_range)
        
    return wins,nn

