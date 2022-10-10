# -*- coding: utf-8

"""
Thomas Grill, 2011-2015
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSGFWIN.M
---------------------------------------------------------------
 [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
 centers correspond to center frequencies to be
 used for the nonstationary Gabor transform with varying Q-factor. 
---------------------------------------------------------------

INPUT : fmin ...... Minimum frequency (in Hz)
        bins ...... Vector consisting of the number of bins per octave
        sr ........ Sampling rate (in Hz)
        Ls ........ Length of signal (in samples)

OUTPUT : g ......... Cell array of window functions.
         rfbas ..... Vector of positions of the center frequencies.
         M ......... Vector of lengths of the window functions.

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

EXTERNALS : firwin
"""

import numpy as np
from .util import hannwin,_isseq

def nsgfwin(fmin, fmax ,bins, sr, Ls, min_win=4, device="cpu"):

    nf = sr/2
    
    if fmax > nf:
        fmax = nf
    
    b = np.ceil(np.log2(fmax/fmin))+1

    if not _isseq(bins):
        bins = np.ones(b,dtype=int)*bins
    elif len(bins) < b:
        # TODO: test this branch!
        bins[bins <= 0] = 1
        bins = np.concatenate((bins, np.ones(b-len(bins), dtype=int)*np.min(bins)))
    
    fbas = []
    for kk,bkk in enumerate(bins):
        r = np.arange(kk*bkk, (kk+1)*bkk, dtype=float)
        # TODO: use N.logspace instead
        fbas.append(2**(r/bkk)*fmin)
    fbas = np.concatenate(fbas)

    if fbas[np.min(np.where(fbas>=fmax))] >= nf:
        fbas = fbas[:np.max(np.where(fbas<fmax))+1]
    else:
        # TODO: test this branch!
        fbas = fbas[:np.min(np.where(fbas>=fmax))+1]
    
    lbas = len(fbas)
    fbas = np.concatenate(((0.,), fbas, (nf,), sr-fbas[::-1]))
    fbas *= float(Ls)/sr
    
    # TODO: put together with array indexing
    M = np.empty(fbas.shape, dtype=int)
    M[0] = np.round(2.*fmin*Ls/sr)
    for k in range(1, 2*lbas+1):
        M[k] = np.round(fbas[k+1]-fbas[k-1])
    M[-1] = np.round(Ls-fbas[-2])
    
    M = np.clip(M, min_win, np.inf).astype(int)
    g = [hannwin(m, device=device) for m in M]
    
    fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
    fbas[lbas+2] = Ls-fbas[lbas]
    rfbas = np.round(fbas).astype(int)
    
    return g,rfbas,M
